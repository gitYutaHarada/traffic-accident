"""
2段階モデル + DAE特徴量統合パイプライン
========================================
Stage 1: LightGBM + 1:2 Under-sampling + 3-Seed Averaging
Stage 2: LightGBM + Focal Loss + DAE特徴量

DAE (Denoising Autoencoder) による特徴量抽出:
- CVの各Fold内でDAEを学習し、ボトルネック特徴量 (128次元) を抽出
- リーク防止: DAEは訓練データのみで学習し、検証/テストデータには変換のみ適用

注意:
- Focal Loss使用時の予測値は、実際のイベント発生確率とは乖離します。
  ビジネスで使用する場合は「スコア」として扱うか、Isotonic Regression等でキャリブレーションを行ってください。
"""

import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import lightgbm as lgb
from scipy.special import expit
import warnings

# DAE Feature Extractor (local import)
from dae_feature_extractor import DAEFeatureExtractor

warnings.filterwarnings('ignore')


def get_focal_loss_lgb(alpha: float = 0.75, gamma: float = 1.0):
    """
    LightGBM用 Focal Loss を生成するファクトリー関数
    
    Args:
        alpha: 正例(死亡事故)の重み (0.5より大きいと正例を重視)
        gamma: 難易度に応じた重み付けパラメータ (0で通常のCE, 大きいほど難しいサンプルを重視)
    
    Returns:
        focal_loss_lgb: LightGBM用カスタム損失関数
    """
    def focal_loss_lgb(y_true, preds):
        """
        LightGBM用 Focal Loss
        
        注意: LGBMClassifier (sklearn API) では引数の順序が (y_true, preds) となる
        preds: モデルの生出力 (Logits)
        y_true: 正解ラベル (numpy array)
        """
        # シグモイド変換
        p = expit(preds)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        # p_t: 正解クラスの確率
        p_t = y_true * p + (1 - y_true) * (1 - p)
        
        # alpha_t: クラスごとの重み
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** gamma
        
        # 勾配
        grad = alpha_t * focal_weight * (p - y_true)
        
        # ヘッセ行列（近似）
        # 注意: 厳密なFocal Lossの2階微分はより複雑な項を含みますが、
        # 数値安定性のため、grad * (1 - 2*p) の項を無視した近似を使用。
        # focal_weightは定数として扱っています（微分の連鎖律に含まれていない）。
        # この近似は実用上多くのケースで機能します。
        # 学習が不安定な場合は scale_pos_weight を使用した重み付けLogLossと比較検討してください。
        hess = alpha_t * focal_weight * p * (1 - p)
        hess = np.maximum(hess, 1e-7)
        
        return grad, hess
    
    return focal_loss_lgb


class TwoStageDAEPipeline:
    """2段階モデル + DAE特徴量統合パイプライン"""
    
    def __init__(
        self,
        data_path: str = "data/processed/honhyo_clean_with_features.csv",
        target_col: str = "死者数",
        n_folds: int = 5,
        random_state: int = 42,
        stage1_recall_target: float = 0.95,
        undersample_ratio: float = 2.0,
        n_seeds: int = 3,
        top_k_interactions: int = 5,
        test_size: float = 0.2,
        # Optuna最適化パラメータ
        focal_alpha: float = 0.6321,
        focal_gamma: float = 1.1495,
        # DAEパラメータ
        dae_bottleneck_dim: int = 128,
        dae_hidden_dim: int = 768,    # 高速化: 1500->768
        dae_epochs: int = 15,         # 高速化: 50->15
        dae_swap_noise: float = 0.15,
        dae_batch_size: int = 32768,  # GPU最適化 (RTX 5080用)
        # オプション
        use_prob_stage1: bool = True,
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.n_folds = n_folds
        self.random_state = random_state
        self.stage1_recall_target = stage1_recall_target
        self.undersample_ratio = undersample_ratio
        self.n_seeds = n_seeds
        self.top_k_interactions = top_k_interactions
        self.test_size = test_size
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # DAE parameters
        self.dae_bottleneck_dim = dae_bottleneck_dim
        self.dae_hidden_dim = dae_hidden_dim
        self.dae_epochs = dae_epochs
        self.dae_swap_noise = dae_swap_noise
        self.dae_batch_size = dae_batch_size
        self.use_prob_stage1 = use_prob_stage1
        
        self.output_dir = "results/two_stage_model/dae_pipeline"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Storage for models
        self.stage1_models = []
        self.stage2_models = []
        self.dae_models = []
        
        print("=" * 60)
        print("2段階モデル + DAE特徴量パイプライン")
        print(f"Stage 1: 1:{int(self.undersample_ratio)} Under-sampling, Recall {self.stage1_recall_target:.0%}")
        print(f"📝 Stage 1予測は Logits で保持し、Seed間で平均化してから Sigmoid 適用")
        print(f"Focal Loss: Alpha={self.focal_alpha:.4f}, Gamma={self.focal_gamma:.4f}")
        print(f"DAE: Bottleneck={self.dae_bottleneck_dim}, Epochs={self.dae_epochs}, Batch={self.dae_batch_size}")
        print(f"use_prob_stage1: {self.use_prob_stage1}")
        print(f"Test Set: {self.test_size:.0%}")
        print("=" * 60)
    
    def load_data(self):
        """データ読み込みとTrain/Test分割"""
        print("\n📂 データ読み込み中...")
        self.df = pd.read_csv(self.data_path)
        
        y_all = self.df[self.target_col].values
        X_all = self.df.drop(columns=[self.target_col])
        
        if '発生日時' in X_all.columns:
            X_all = X_all.drop(columns=['発生日時'])
        
        # Train/Test分割 (層化抽出)
        self.X, self.X_test, self.y, self.y_test = train_test_split(
            X_all, y_all, test_size=self.test_size, 
            random_state=self.random_state, stratify=y_all
        )
        
        print(f"\n📊 データ分割 (Train: {1-self.test_size:.0%} / Test: {self.test_size:.0%})")
        print(f"   Train: 正例 {self.y.sum():,} / {len(self.y):,}")
        print(f"   Test:  正例 {self.y_test.sum():,} / {len(self.y_test):,}")
        
        # カテゴリ変数の特定
        known_categoricals = [
            '都道府県コード', '市区町村コード', '警察署等コード',
            '昼夜', '天候', '地形', '路面状態', '道路形状', '信号機',
            '衝突地点', 'ゾーン規制', '中央分離帯施設等', '歩車道区分',
            '事故類型', '曜日(発生年月日)', '祝日(発生年月日)',
            'road_type', 'area_id', '地点コード'
        ]
        
        self.categorical_cols = []
        self.numeric_cols = []
        
        for col in self.X.columns:
            if col in known_categoricals or self.X[col].dtype == 'object':
                self.categorical_cols.append(col)
                self.X[col] = self.X[col].astype('category')
                self.X_test[col] = self.X_test[col].astype('category')
            else:
                self.numeric_cols.append(col)
                self.X[col] = self.X[col].astype(np.float32)
                self.X_test[col] = self.X_test[col].astype(np.float32)
        
        self.feature_names = list(self.X.columns)
        gc.collect()
    
    def train_stage1(self):
        """Stage 1: LightGBM + Under-sampling + Multi-Seed"""
        print(f"\n🌿 Stage 1: LightGBM + Under-sampling (1:{int(self.undersample_ratio)}) + {self.n_seeds}-Seed Averaging")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.oof_proba_stage1 = np.zeros(len(self.y))
        feature_importances = np.zeros(len(self.feature_names))
        
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'num_leaves': 31,
            'max_depth': 8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'is_unbalance': False,
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'n_jobs': -1
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            X_train_full = self.X.iloc[train_idx]
            X_val = self.X.iloc[val_idx]
            y_train_full = self.y[train_idx]
            y_val = self.y[val_idx]
            
            fold_models = []
            fold_logits = np.zeros(len(val_idx))  # 確率ではなくLogitsで平均化
            
            for seed in range(self.n_seeds):
                np.random.seed(self.random_state + seed)
                
                # Under-sampling
                pos_idx = np.where(y_train_full == 1)[0]
                neg_idx = np.where(y_train_full == 0)[0]
                n_pos = len(pos_idx)
                n_neg_sample = int(n_pos * self.undersample_ratio)
                neg_sample_idx = np.random.choice(neg_idx, size=min(n_neg_sample, len(neg_idx)), replace=False)
                
                train_idx_sampled = np.concatenate([pos_idx, neg_sample_idx])
                X_train = X_train_full.iloc[train_idx_sampled]
                y_train = y_train_full[train_idx_sampled]
                
                model = lgb.LGBMClassifier(**lgb_params, random_state=self.random_state + seed)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                
                # Logitsで取得して平均化（極端な予測に対してロバスト）
                raw_score = model.predict_proba(X_val)[:, 1]
                # predict_probaは確率を返すので、logit変換してから平均
                raw_score = np.clip(raw_score, 1e-15, 1 - 1e-15)
                logits = np.log(raw_score / (1 - raw_score))  # logit変換
                fold_logits += logits / self.n_seeds
                fold_models.append(model)
                feature_importances += model.feature_importances_ / (self.n_folds * self.n_seeds)
            
            # Logits平均からSigmoidで確率に変換
            self.oof_proba_stage1[val_idx] = expit(fold_logits)
            # Logitsも保存（Stage 2の特徴量として使用）
            if not hasattr(self, 'oof_logits_stage1'):
                self.oof_logits_stage1 = np.zeros(len(self.y))
            self.oof_logits_stage1[val_idx] = fold_logits
            self.stage1_models.append(fold_models)
            
            del X_train, X_val
            gc.collect()
        
        # Feature Importance
        self.feature_importance_df = pd.DataFrame({
            'feature': self.feature_names, 'importance': feature_importances
        }).sort_values('importance', ascending=False)
        self.top_features = self.feature_importance_df.head(10)['feature'].tolist()
        
        # OOF評価
        oof_pred = (self.oof_proba_stage1 >= 0.5).astype(int)
        oof_auc = roc_auc_score(self.y, self.oof_proba_stage1)
        print(f"   OOF (閾値0.5): Prec={precision_score(self.y, oof_pred):.4f}, "
              f"Rec={recall_score(self.y, oof_pred):.4f}, AUC={oof_auc:.4f}")
    
    def find_recall_threshold(self):
        """Recall目標を達成する閾値を探索"""
        # 0.5から下げていき、Recall目標を満たす最大の閾値を見つける
        # (以前の実装は0.001から上げていたため、最小の閾値で止まってしまっていた)
        for thresh in np.arange(0.5, 0.0, -0.001):
            y_pred = (self.oof_proba_stage1 >= thresh).astype(int)
            recall = recall_score(self.y, y_pred)
            if recall >= self.stage1_recall_target:
                self.threshold_stage1 = thresh
                break
        else:
            self.threshold_stage1 = 0.001
        
        y_pred_final = (self.oof_proba_stage1 >= self.threshold_stage1).astype(int)
        self.stage1_recall = recall_score(self.y, y_pred_final)
        n_candidates = y_pred_final.sum()
        self.filter_rate = 1 - (n_candidates / len(self.y))
        n_filtered = len(self.y) - n_candidates
        
        print(f"   閾値: {self.threshold_stage1:.4f}, Recall: {self.stage1_recall:.4f}")
        print(f"   [Result] フィルタリング: {n_filtered:,} 件除外 ({self.filter_rate:.2%})")
        print(f"   [Result] 残存データ: {n_candidates:,} 件 (Stage 2 候補)")
        print(f"   [Result] 正例残存: {self.y[self.oof_proba_stage1 >= self.threshold_stage1].sum():,} / {self.y.sum():,}")
        
        self.stage2_mask = self.oof_proba_stage1 >= self.threshold_stage1
    
    def generate_stage2_features(self, X_subset, logits_stage1_subset, fit_categories=True):
        """
        Stage 2用特徴量生成 (DAE特徴量なし、基本特徴量のみ)
        
        Args:
            logits_stage1_subset: Stage 1のLogits値（確率ではなく生スコア）
                                   Logitsは情報の解像度が高く、学習しやすい
        """
        X_out = X_subset.copy()
        
        # (a) logits_stage1 追加 (オプション)
        # 確率(0-1)ではなくLogitsを使用することで端の情報を保持
        if self.use_prob_stage1:
            X_out['logits_stage1'] = logits_stage1_subset
        
        # (b) Categorical Interaction Features
        top_cat_features = [f for f in self.top_features if f in self.categorical_cols]
        
        if fit_categories:
            self.interaction_categories = {}
        
        for i, f1 in enumerate(top_cat_features[:self.top_k_interactions]):
            for f2 in top_cat_features[i+1:self.top_k_interactions]:
                name = f"{f1}_{f2}"
                interaction_values = X_subset[f1].astype(str) + "_" + X_subset[f2].astype(str)
                
                if fit_categories:
                    # 学習時: カテゴリを作成して保存
                    cat_type = pd.CategoricalDtype(categories=list(interaction_values.unique()) + ['__UNKNOWN__'])
                    self.interaction_categories[name] = cat_type
                    X_out[name] = pd.Categorical(interaction_values, dtype=cat_type)
                else:
                    # テスト時: 保存済みカテゴリを使用、未知の組み合わせは __UNKNOWN__ にマップ
                    if hasattr(self, 'interaction_categories') and name in self.interaction_categories:
                        known_cats = set(self.interaction_categories[name].categories)
                        # 未知の組み合わせを __UNKNOWN__ に置換
                        interaction_values = interaction_values.apply(
                            lambda x: x if x in known_cats else '__UNKNOWN__'
                        )
                        X_out[name] = pd.Categorical(interaction_values, dtype=self.interaction_categories[name])
                    else:
                        X_out[name] = interaction_values.astype('category')
        
        return X_out
    
    def train_stage2_with_dae(self):
        """
        Stage 2: DAE特徴量を使用したLightGBM学習
        CVの各Fold内でDAEを学習し、特徴量を追加
        """
        print("\n🌿 Stage 2: LightGBM + DAE特徴量 (5-Fold CV)")
        print(f"   Focal Loss: Alpha={self.focal_alpha:.4f}, Gamma={self.focal_gamma:.4f}")
        print(f"   DAE: Bottleneck={self.dae_bottleneck_dim}, Epochs={self.dae_epochs}")
        
        # Stage 2用の全データ (基本特徴量のみ) - Logitsを使用
        X_s2_base = self.generate_stage2_features(
            self.X[self.stage2_mask].copy(),
            self.oof_logits_stage1[self.stage2_mask],  # 確率ではなくLogitsを使用
            fit_categories=True
        ).reset_index(drop=True)
        
        y_s2_full = self.y[self.stage2_mask]
        
        n_pos, n_neg = y_s2_full.sum(), len(y_s2_full) - y_s2_full.sum()
        print(f"   Stage 2 データ: {len(y_s2_full):,} (Pos: {n_pos:,}, Neg: {n_neg:,})")
        print(f"   Top Features for Interaction: {self.top_features[:5]}")
        
        # OOF予測値を格納
        self.oof_proba_stage2 = np.zeros(len(y_s2_full))
        self.stage2_models = []
        self.dae_models = []
        
        # CV設定
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # LightGBMパラメータ (Optuna最適化済み)
        focal_loss_fn = get_focal_loss_lgb(alpha=self.focal_alpha, gamma=self.focal_gamma)
        lgb_params = {
            'objective': focal_loss_fn,
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'num_leaves': 127,
            'max_depth': -1,
            'min_child_samples': 44,
            'reg_alpha': 2.3897,
            'reg_lambda': 2.2842,
            'colsample_bytree': 0.8646,
            'subsample': 0.6328,
            'learning_rate': 0.0477,
            'is_unbalance': False,
            'n_estimators': 1000,
            'n_jobs': -1
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_s2_base, y_s2_full)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            
            X_train_base = X_s2_base.iloc[train_idx].reset_index(drop=True)
            X_val_base = X_s2_base.iloc[val_idx].reset_index(drop=True)
            y_train = y_s2_full[train_idx]
            y_val = y_s2_full[val_idx]
            
            # === DAE学習 & 特徴量抽出 ===
            print(f"      📦 DAE学習中...")
            dae = DAEFeatureExtractor(
                numeric_cols=self.numeric_cols + (['logits_stage1'] if self.use_prob_stage1 else []),
                cat_cols=self.categorical_cols,
                bottleneck_dim=self.dae_bottleneck_dim,
                hidden_dim=self.dae_hidden_dim,
                epochs=self.dae_epochs,
                swap_noise_rate=self.dae_swap_noise,
                batch_size=self.dae_batch_size,
                verbose=True,  # ログを表示してGPU確認
                n_workers=4    # 高速化のためにワーカーを使用
            )
            
            # デバイス確認用ログ
            print(f"      🖥️  Device being used: {dae.device}")
            
            # DAEは訓練データのみで学習
            dae.fit(X_train_base)
            
            # 訓練・検証データの両方から特徴量抽出
            dae_train_features = dae.transform(X_train_base)
            dae_val_features = dae.transform(X_val_base)
            
            # DAE特徴量をDataFrameに変換
            dae_cols = [f'dae_{i}' for i in range(self.dae_bottleneck_dim)]
            dae_train_df = pd.DataFrame(dae_train_features, columns=dae_cols)
            dae_val_df = pd.DataFrame(dae_val_features, columns=dae_cols)
            
            # 基本特徴量とDAE特徴量を結合
            X_train_full = pd.concat([X_train_base.reset_index(drop=True), dae_train_df], axis=1)
            X_val_full = pd.concat([X_val_base.reset_index(drop=True), dae_val_df], axis=1)
            
            # === LightGBM学習 ===
            model = lgb.LGBMClassifier(**lgb_params, random_state=self.random_state)
            model.fit(
                X_train_full, y_train,
                eval_set=[(X_val_full, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            # OOF予測 (raw_scoreからシグモイド変換)
            raw_score = model.predict(X_val_full, raw_score=True)
            proba = 1.0 / (1.0 + np.exp(-raw_score))
            self.oof_proba_stage2[val_idx] = proba
            
            # モデル保存
            self.stage2_models.append(model)
            self.dae_models.append(dae)
            
            del X_train_full, X_val_full, dae_train_features, dae_val_features
            gc.collect()
        
        # Stage 2 OOF評価
        oof_auc = roc_auc_score(y_s2_full, self.oof_proba_stage2)
        print(f"   Stage 2 OOF AUC: {oof_auc:.4f}")
    
    def evaluate(self):
        """最終評価 (CV OOF)"""
        print("\n📈 最終評価 (Cross Validation OOF)")
        
        y_s2 = self.y[self.stage2_mask]
        
        # 最終予測確率
        self.final_proba = np.zeros(len(self.y))
        self.final_proba[self.stage2_mask] = self.oof_proba_stage2
        
        # 動的閾値評価
        precisions, recalls, thresholds = precision_recall_curve(y_s2, self.oof_proba_stage2)
        
        self.dynamic_results = {}
        target_recalls = [0.99, 0.98, 0.95]
        
        print("\n   📊 動的閾値評価:")
        for target_recall in target_recalls:
            idx = np.where(recalls >= target_recall)[0]
            if len(idx) > 0:
                idx = idx[-1]
                if idx < len(thresholds):
                    best_thresh = thresholds[idx]
                    best_prec = precisions[idx]
                else:
                    best_thresh = 0.0
                    best_prec = precisions[-1]
            else:
                best_thresh = 0.0
                best_prec = 0.0
            
            self.dynamic_results[target_recall] = {
                'threshold': best_thresh,
                'precision': best_prec
            }
            print(f"      Recall ~{target_recall:.0%}: 閾値={best_thresh:.4f}, Precision={best_prec:.4f}")
        
        # 固定閾値評価
        y_pred = (self.final_proba >= 0.5).astype(int)
        
        self.final_precision = precision_score(self.y, y_pred) if y_pred.sum() > 0 else 0
        self.final_recall = recall_score(self.y, y_pred)
        self.final_f1 = f1_score(self.y, y_pred)
        self.final_auc = roc_auc_score(self.y, self.final_proba)
        
        print(f"\n   [閾値0.5] Precision: {self.final_precision:.4f}, Recall: {self.final_recall:.4f}, F1: {self.final_f1:.4f}")
        
        # Baseline (Stage 1)
        y_pred_bl = (self.oof_proba_stage1 >= 0.5).astype(int)
        self.baseline_precision = precision_score(self.y, y_pred_bl)
        self.baseline_recall = recall_score(self.y, y_pred_bl)
        print(f"   [ベース(Stage1)] Precision: {self.baseline_precision:.4f}, Recall: {self.baseline_recall:.4f}")
        
        improvement = (self.final_precision - self.baseline_precision) / self.baseline_precision * 100 if self.baseline_precision > 0 else 0
        print(f"   Precision改善率 (閾値0.5): {improvement:+.2f}%")
        
        return {
            'stage1_threshold': self.threshold_stage1,
            'stage1_recall': self.stage1_recall,
            'filter_rate': self.filter_rate,
            'final_precision': self.final_precision,
            'final_recall': self.final_recall,
            'final_f1': self.final_f1,
            'final_auc': self.final_auc,
            'baseline_precision': self.baseline_precision,
            'baseline_recall': self.baseline_recall,
            'precision_improvement_pct': improvement,
            'dynamic_recall_99_precision': self.dynamic_results.get(0.99, {}).get('precision', 0),
            'dynamic_recall_98_precision': self.dynamic_results.get(0.98, {}).get('precision', 0),
        }
    
    def evaluate_test_set(self):
        """テストセットでの最終評価"""
        print("\n📈 テストセット評価 (Hold-Out)")
        
        # Stage 1: アンサンブル予測 (Logits平均)
        test_logits_stage1 = np.zeros(len(self.y_test))
        for fold_models in self.stage1_models:
            for model in fold_models:
                proba = model.predict_proba(self.X_test)[:, 1]
                proba = np.clip(proba, 1e-15, 1 - 1e-15)
                logits = np.log(proba / (1 - proba))
                test_logits_stage1 += logits
        test_logits_stage1 /= (self.n_folds * self.n_seeds)
        test_proba_stage1 = expit(test_logits_stage1)
        
        # Stage 1閾値適用
        test_stage2_mask = test_proba_stage1 >= self.threshold_stage1
        n_candidates = test_stage2_mask.sum()
        n_pos_in_candidates = self.y_test[test_stage2_mask].sum()
        
        print(f"   Stage 1 フィルタリング後: {n_candidates:,} / {len(self.y_test):,}")
        print(f"   正例残存: {n_pos_in_candidates:,} / {self.y_test.sum():,}")
        
        if n_candidates == 0:
            print("   ⚠️ Stage 2に進むデータがありません")
            self.test_results = {'error': 'No candidates after Stage 1'}
            return self.test_results
        
        # Stage 2用基本特徴量 (Logitsを使用)
        X_test_s2_base = self.generate_stage2_features(
            self.X_test[test_stage2_mask].copy(),
            test_logits_stage1[test_stage2_mask],  # 確率ではなくLogitsを使用
            fit_categories=False
        )
        y_test_s2 = self.y_test[test_stage2_mask]
        
        # Stage 2: 各FoldのDAE+LightGBMでアンサンブル予測
        test_proba_stage2 = np.zeros(len(y_test_s2))
        
        for fold, (dae, model) in enumerate(zip(self.dae_models, self.stage2_models)):
            # DAE特徴量抽出
            dae_features = dae.transform(X_test_s2_base)
            dae_cols = [f'dae_{i}' for i in range(self.dae_bottleneck_dim)]
            dae_df = pd.DataFrame(dae_features, columns=dae_cols)
            
            # 結合
            X_test_full = pd.concat([X_test_s2_base.reset_index(drop=True), dae_df], axis=1)
            
            # 予測
            raw_score = model.predict(X_test_full, raw_score=True)
            proba = 1.0 / (1.0 + np.exp(-raw_score))
            test_proba_stage2 += proba / self.n_folds
        
        # 動的閾値評価
        precisions, recalls, thresholds = precision_recall_curve(y_test_s2, test_proba_stage2)
        
        self.test_dynamic_results = {}
        target_recalls = [0.99, 0.98, 0.95]
        
        print("\n   📊 テストセット動的閾値評価:")
        for target_recall in target_recalls:
            idx = np.where(recalls >= target_recall)[0]
            if len(idx) > 0:
                idx = idx[-1]
                if idx < len(thresholds):
                    best_thresh = thresholds[idx]
                    best_prec = precisions[idx]
                else:
                    best_thresh = 0.0
                    best_prec = precisions[-1]
            else:
                best_thresh = 0.0
                best_prec = 0.0
            
            self.test_dynamic_results[target_recall] = {
                'threshold': best_thresh,
                'precision': best_prec
            }
            print(f"      Recall ~{target_recall:.0%}: 閾値={best_thresh:.4f}, Precision={best_prec:.4f}")
        
        # 固定閾値評価
        final_test_proba = np.zeros(len(self.y_test))
        final_test_proba[test_stage2_mask] = test_proba_stage2
        y_test_pred = (final_test_proba >= 0.5).astype(int)
        
        test_precision = precision_score(self.y_test, y_test_pred) if y_test_pred.sum() > 0 else 0
        test_recall = recall_score(self.y_test, y_test_pred)
        test_f1 = f1_score(self.y_test, y_test_pred)
        test_auc = roc_auc_score(self.y_test, final_test_proba)
        
        print(f"\n   [テスト閾値0.5] Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
        print(f"   [テストAUC]: {test_auc:.4f}")
        
        self.test_results = {
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_precision_at_recall99': self.test_dynamic_results.get(0.99, {}).get('precision', 0),
            'test_precision_at_recall98': self.test_dynamic_results.get(0.98, {}).get('precision', 0),
            'test_precision_at_recall95': self.test_dynamic_results.get(0.95, {}).get('precision', 0),
        }
        
        return self.test_results
    
    def generate_report(self, results: dict, elapsed_sec: float):
        """実験レポートをMarkdownで出力"""
        report_path = os.path.join(self.output_dir, "experiment_report.md")
        
        report_content = f"""# DAE特徴量統合実験レポート

**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**実行時間**: {elapsed_sec:.1f}秒

## パラメータ設定

| パラメータ | 値 |
|-----------|----| 
| Focal Alpha | {self.focal_alpha:.4f} |
| Focal Gamma | {self.focal_gamma:.4f} |
| DAE Bottleneck | {self.dae_bottleneck_dim} |
| DAE Epochs | {self.dae_epochs} |
| DAE Swap Noise | {self.dae_swap_noise:.2f} |
| Stage 1 Recall Target | {self.stage1_recall_target:.0%} |
| Test Set Ratio | {self.test_size:.0%} |

## 結果サマリ

### Stage 1
- **閾値**: {results['stage1_threshold']:.4f}
- **Recall**: {results['stage1_recall']:.4f}
- **フィルタリング率**: {results['filter_rate']*100:.2f}%

### Stage 2 (Focal Loss + DAE) - CV OOF評価

#### 固定閾値 (0.5) での評価
| 指標 | 値 |
|------|----| 
| Precision | {results['final_precision']:.4f} |
| Recall | {results['final_recall']:.4f} |
| F1 | {results['final_f1']:.4f} |
| AUC | {results['final_auc']:.4f} |

#### 動的閾値での評価 (CV OOF)
| Target Recall | Precision |
|---------------|----------|
| 99% | {results.get('dynamic_recall_99_precision', 0):.4f} |
| 98% | {results.get('dynamic_recall_98_precision', 0):.4f} |

### テストセット評価 (Hold-Out {self.test_size:.0%})

| 指標 | 値 |
|------|----| 
| Precision | {results.get('test_precision', 0):.4f} |
| Recall | {results.get('test_recall', 0):.4f} |
| F1 | {results.get('test_f1', 0):.4f} |
| AUC | {results.get('test_auc', 0):.4f} |

#### 動的閾値での評価 (Test Set)
| Target Recall | Precision |
|---------------|----------|
| 99% | {results.get('test_precision_at_recall99', 0):.4f} |
| 98% | {results.get('test_precision_at_recall98', 0):.4f} |
| 95% | {results.get('test_precision_at_recall95', 0):.4f} |

## 考察

- DAE特徴量 ({self.dae_bottleneck_dim}次元) により、LightGBMが苦手な非線形関係を捕捉
- Swap Noise ({self.dae_swap_noise:.0%}) によるノイズ除去効果
- CV OOF と Test Set の結果が近いほど、汎化性能が高い
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n   📄 レポート出力: {report_path}")
        return report_path
    
    def run(self):
        start = datetime.now()
        self.load_data()
        self.train_stage1()
        self.find_recall_threshold()
        self.train_stage2_with_dae()
        results = self.evaluate()
        
        # テストセット評価
        test_results = self.evaluate_test_set()
        results.update(test_results)
        
        elapsed_sec = (datetime.now() - start).total_seconds()
        results['elapsed_sec'] = elapsed_sec
        
        # 結果保存
        pd.DataFrame([results]).to_csv(os.path.join(self.output_dir, "final_results.csv"), index=False)
        self.feature_importance_df.to_csv(os.path.join(self.output_dir, "stage1_feature_importance.csv"), index=False)
        
        # Markdown レポート生成
        self.generate_report(results, elapsed_sec)
        
        print("\n" + "=" * 60)
        print("✅ 完了！")
        print(f"   結果CSV: {self.output_dir}/final_results.csv")
        print(f"   レポートMD: {self.output_dir}/experiment_report.md")
        print("=" * 60)
        
        return results


if __name__ == "__main__":
    pipeline = TwoStageDAEPipeline()
    pipeline.run()
