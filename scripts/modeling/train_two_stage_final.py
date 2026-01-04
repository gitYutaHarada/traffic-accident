"""
2段階モデル 最終パイプライン
============================
Implementation Plan v18

Stage 1: LightGBM + 1:2 Under-sampling + 3-Seed Averaging
Stage 2: High Complexity + Strong Regularization

特徴エンジニアリング:
- prob_stage1 (OOF予測値を使用、リーク防止)
- Categorical Interaction Features (文字列結合)
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
        p = np.clip(p, 1e-15, 1 - 1e-15)  # 数値安定性のためクリップ
        
        # p_t: 正解クラスの確率
        # y=1 の場合 p_t = p, y=0 の場合 p_t = 1-p
        p_t = y_true * p + (1 - y_true) * (1 - p)
        
        # alpha_t: クラスごとの重み
        # y=1 の場合 alpha, y=0 の場合 1-alpha
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** gamma
        
        # 簡略化した勾配計算
        # grad = alpha_t * focal_weight * (p - y_true)
        # これはクロスエントロピーの勾配 (p - y) に focal_weight と alpha_t を掛けたもの
        grad = alpha_t * focal_weight * (p - y_true)
        
        # ヘッセ行列（近似）
        # 標準的なログロスのヘッセ行列に focal_weight と alpha_t を掛ける
        hess = alpha_t * focal_weight * p * (1 - p)
        # 数値安定性のため、ヘッセ行列に最小値を設定
        hess = np.maximum(hess, 1e-7)
        
        return grad, hess
    
    return focal_loss_lgb



class TwoStageFinalPipeline:
    """2段階モデル最終パイプライン"""
    
    def __init__(
        self,
        data_path: str = "data/processed/honhyo_clean_with_features.csv",
        target_col: str = "死者数",
        n_folds: int = 5,
        random_state: int = 42,
        stage1_recall_target: float = 0.99,
        undersample_ratio: float = 2.0,  # 1:2
        n_seeds: int = 3,
        top_k_interactions: int = 5,
        test_size: float = 0.2,  # テストセット比率
        # Optuna最適化パラメータ (optuna_focal_loss_v2)
        focal_alpha: float = 0.6321,
        focal_gamma: float = 1.1495,
        output_dir: str = "results/two_stage_model/final_pipeline",  # 出力ディレクトリ
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
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 60)
        print("2段階モデル 最終パイプライン (Optuna最適化版)")
        print(f"Stage 1: 1:{int(self.undersample_ratio)} Under-sampling, Recall {self.stage1_recall_target:.0%}")
        print(f"Focal Loss: Alpha={self.focal_alpha:.4f}, Gamma={self.focal_gamma:.4f}")
        print(f"Test Set: {self.test_size:.0%}")
        print("=" * 60)
    
    def load_data(self):
        """データ読み込みとTrain/Test分割"""
        print("\n📂 データ読み込み中...")
        df = pd.read_csv(self.data_path)
        y_all = df[self.target_col].values
        X_all = df.drop(columns=[self.target_col])
        
        if '発生日時' in X_all.columns:
            X_all = X_all.drop(columns=['発生日時'])
        
        known_categoricals = [
            '都道府県コード', '市区町村コード', '警察署等コード',
            '昼夜', '天候', '地形', '路面状態', '道路形状', '信号機',
            '衝突地点', 'ゾーン規制', '中央分離帯施設等', '歩車道区分',
            '事故類型', '曜日(発生年月日)', '祝日(発生年月日)',
            'road_type', 'area_id', '地点コード'
        ]
        
        self.categorical_cols = []
        for col in X_all.columns:
            if col in known_categoricals or X_all[col].dtype == 'object':
                self.categorical_cols.append(col)
                X_all[col] = X_all[col].astype('category')
            else:
                X_all[col] = X_all[col].astype(np.float32)
        
        self.feature_names = list(X_all.columns)
        
        # Train/Test分割
        print(f"\n📊 データ分割 (Train: {1-self.test_size:.0%} / Test: {self.test_size:.0%})")
        self.X, self.X_test, self.y, self.y_test = train_test_split(
            X_all, y_all,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_all
        )
        self.X = self.X.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        
        print(f"   Train: 正例 {self.y.sum():,} / {len(self.y):,}")
        print(f"   Test:  正例 {self.y_test.sum():,} / {len(self.y_test):,}")
        gc.collect()

    
    def undersample(self, X, y, seed):
        """負例をアンダーサンプリング"""
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_neg_sample = int(len(pos_idx) * self.undersample_ratio)
        np.random.seed(seed)
        sampled_neg_idx = np.random.choice(neg_idx, size=min(n_neg_sample, len(neg_idx)), replace=False)
        sampled_idx = np.concatenate([pos_idx, sampled_neg_idx])
        np.random.shuffle(sampled_idx)
        return X.iloc[sampled_idx], y[sampled_idx]
    
    def train_stage1(self):
        """Stage 1: OOF学習 + Feature Importance取得"""
        print("\n🌿 Stage 1: LightGBM + Under-sampling (1:2) + 3-Seed Averaging")
        
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
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'n_jobs': -1
        }
        
        self.stage1_models = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            X_train_full = self.X.iloc[train_idx]
            y_train_full = self.y[train_idx]
            X_val = self.X.iloc[val_idx]
            y_val = self.y[val_idx]
            
            fold_proba = np.zeros(len(val_idx))
            fold_models = []
            
            for seed_offset in range(self.n_seeds):
                seed = self.random_state + fold * 100 + seed_offset
                X_train_under, y_train_under = self.undersample(X_train_full, y_train_full, seed)
                
                model = lgb.LGBMClassifier(**lgb_params, random_state=seed)
                model.fit(
                    X_train_under, y_train_under,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                
                fold_proba += model.predict_proba(X_val)[:, 1] / self.n_seeds
                feature_importances += model.feature_importances_ / (self.n_folds * self.n_seeds)
                fold_models.append(model)
                
                del model
                gc.collect()
            
            self.oof_proba_stage1[val_idx] = fold_proba
            self.stage1_models.append(fold_models)
        
        # Feature Importance
        self.feature_importance_df = pd.DataFrame({
            'feature': self.feature_names, 'importance': feature_importances
        }).sort_values('importance', ascending=False)
        self.top_features = self.feature_importance_df.head(self.top_k_interactions)['feature'].tolist()
        
        # OOF精度
        oof_pred = (self.oof_proba_stage1 >= 0.5).astype(int)
        print(f"   OOF (閾値0.5): Prec={precision_score(self.y, oof_pred):.4f}, Rec={recall_score(self.y, oof_pred):.4f}, AUC={roc_auc_score(self.y, self.oof_proba_stage1):.4f}")
    
    def find_recall_threshold(self):
        """Recall目標閾値探索"""
        for thresh in np.arange(0.50, 0.001, -0.005):
            y_pred = (self.oof_proba_stage1 >= thresh).astype(int)
            recall = recall_score(self.y, y_pred)
            if recall >= self.stage1_recall_target:
                self.threshold_stage1 = thresh
                break
        else:
            self.threshold_stage1 = 0.001
        
        y_pred_final = (self.oof_proba_stage1 >= self.threshold_stage1).astype(int)
        self.stage1_recall = recall_score(self.y, y_pred_final)
        self.stage1_precision = precision_score(self.y, y_pred_final)
        n_candidates = y_pred_final.sum()
        self.filter_rate = 1 - (n_candidates / len(self.y))
        
        print(f"   閾値: {self.threshold_stage1:.4f}, Recall: {self.stage1_recall:.4f}")
        print(f"   フィルタリング率: {self.filter_rate*100:.2f}% 除外, 候補: {n_candidates:,}")
        
        self.stage2_mask = self.oof_proba_stage1 >= self.threshold_stage1
        
        # OOF結果保存（XGBoostとの相関分析用）
        oof_df = pd.DataFrame({
            'y_true': self.y,
            'oof_proba': self.oof_proba_stage1
        })
        oof_path = "results/two_stage_model/lightgbm_stage1_oof.csv"
        os.makedirs(os.path.dirname(oof_path), exist_ok=True)
        oof_df.to_csv(oof_path, index=False)
        print(f"   💾 OOF結果保存: {oof_path}")
    
    def generate_stage2_features(self, X_subset, prob_stage1_subset, fit_categories=True):
        """
        Stage 2用特徴量生成
        
        Args:
            X_subset: 入力特徴量DataFrame
            prob_stage1_subset: Stage 1の予測確率
            fit_categories: Trueの場合、カテゴリマッピングを学習して保存。
                           Falseの場合、保存済みのマッピングを適用（テスト時用）。
        """
        X_out = X_subset.copy()
        
        # (a) prob_stage1 追加
        X_out['prob_stage1'] = prob_stage1_subset
        
        # (b) Categorical Interaction Features
        top_cat_features = [f for f in self.top_features if f in self.categorical_cols]
        
        if fit_categories:
            # 学習時: カテゴリマッピングを保存
            self.interaction_categories = {}
        
        for i, f1 in enumerate(top_cat_features[:self.top_k_interactions]):
            for f2 in top_cat_features[i+1:self.top_k_interactions]:
                name = f"{f1}_{f2}"
                interaction_values = X_subset[f1].astype(str) + "_" + X_subset[f2].astype(str)
                
                if fit_categories:
                    # 学習時: カテゴリを作成して保存
                    cat_type = pd.CategoricalDtype(categories=interaction_values.unique())
                    self.interaction_categories[name] = cat_type
                    X_out[name] = pd.Categorical(interaction_values, dtype=cat_type)
                else:
                    # テスト時: 保存済みカテゴリを使用（未知のカテゴリはNaNになる）
                    if hasattr(self, 'interaction_categories') and name in self.interaction_categories:
                        X_out[name] = pd.Categorical(interaction_values, dtype=self.interaction_categories[name])
                    else:
                        X_out[name] = interaction_values.astype('category')
        
        return X_out
    
    def get_stage2_data(self):
        """
        Optuna等の外部スクリプト用: Stage 2データを生成して返す
        
        Returns:
            X_s2: Stage 2用の特徴量DataFrame
            y_s2: Stage 2用のラベルarray
        """
        self.load_data()
        self.train_stage1()
        self.find_recall_threshold()
        
        X_s2 = self.generate_stage2_features(
            self.X[self.stage2_mask].copy(),
            self.oof_proba_stage1[self.stage2_mask]
        )
        y_s2 = self.y[self.stage2_mask]
        
        print(f"\n📦 Stage 2用データ生成完了:")
        print(f"   データ数: {len(y_s2):,} (Pos: {y_s2.sum():,}, Neg: {len(y_s2)-y_s2.sum():,})")
        
        return X_s2, y_s2
    
    def train_stage2(self):
        """
        Stage 2: Cross Validationによる学習と評価
        (学習データに対する過学習を防ぎ、真の汎化性能を測定する)
        """
        print("\n🌿 Stage 2: High Complexity + Strong Regularization (5-Fold CV)")
        print(f"   Focal Loss: Alpha={self.focal_alpha:.4f}, Gamma={self.focal_gamma:.4f}")
        
        # Stage 2用の全データ
        X_s2_full = self.generate_stage2_features(
            self.X[self.stage2_mask].copy(),
            self.oof_proba_stage1[self.stage2_mask]
        ).reset_index(drop=True)
        
        y_s2_full = self.y[self.stage2_mask]
        
        n_pos, n_neg = y_s2_full.sum(), len(y_s2_full) - y_s2_full.sum()
        print(f"   Stage 2 データ: {len(y_s2_full):,} (Pos: {n_pos:,}, Neg: {n_neg:,})")
        print(f"   Top Features for Interaction: {self.top_features}")
        
        # Stage 2のOOF予測値を格納する配列
        self.oof_proba_stage2 = np.zeros(len(y_s2_full))
        self.stage2_models = []
        
        # CV設定
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # ハイパーパラメータ (Optuna最適化 - optuna_focal_loss_v2)
        focal_loss_fn = get_focal_loss_lgb(alpha=self.focal_alpha, gamma=self.focal_gamma)
        lgb_params = {
            'objective': focal_loss_fn,  # カスタムFocal Loss (動的パラメータ)
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'num_leaves': 127,        # Optuna最適化
            'max_depth': -1,          # num_leaves=127を活かすため制限なし
            'min_child_samples': 44,  # Optuna最適化
            'reg_alpha': 2.3897,      # Optuna最適化
            'reg_lambda': 2.2842,     # Optuna最適化
            'colsample_bytree': 0.8646,  # Optuna最適化
            'subsample': 0.6328,      # Optuna最適化
            'learning_rate': 0.0477,  # Optuna最適化
            'is_unbalance': False,
            'n_estimators': 1000,
            'n_jobs': -1
        }

        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_s2_full, y_s2_full)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            X_train, y_train = X_s2_full.iloc[train_idx], y_s2_full[train_idx]
            X_val, y_val = X_s2_full.iloc[val_idx], y_s2_full[val_idx]
            
            model = lgb.LGBMClassifier(**lgb_params, random_state=self.random_state + fold)
            
            # Early Stoppingを利用して過学習抑制
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            # Focal Loss使用時はraw_score=Trueでlogitを取得し、シグモイド変換
            y_pred_raw = model.predict(X_val, raw_score=True)
            y_pred_proba = 1.0 / (1.0 + np.exp(-y_pred_raw))
            
            self.oof_proba_stage2[val_idx] = y_pred_proba
            self.stage2_models.append(model)
            
            del model
            gc.collect()
        
        # OOF精度（Stage 2のみ）
        oof_auc = roc_auc_score(y_s2_full, self.oof_proba_stage2)
        print(f"   Stage 2 OOF AUC: {oof_auc:.4f}")
        
        self.stage2_feature_names = list(X_s2_full.columns)
    
    def evaluate(self):
        """最終評価（CVのOOF予測値を用いた公平な評価）"""
        print("\n📈 最終評価 (Cross Validation OOF)")
        
        # Stage 2のOOF予測確率を使用（train_stage2で生成済み）
        y_prob_s2 = self.oof_proba_stage2
        
        # Stage 2 のスコア分布を表示
        print("\n   📊 予測スコア分布 (Stage 2 OOF):")
        prob_series = pd.Series(y_prob_s2)
        print(f"      mean={prob_series.mean():.4f}, std={prob_series.std():.4f}")
        print(f"      min={prob_series.min():.4f}, 25%={prob_series.quantile(0.25):.4f}, 50%={prob_series.quantile(0.5):.4f}, 75%={prob_series.quantile(0.75):.4f}, max={prob_series.max():.4f}")
        
        # Stage 2対象外のデータは確率0として全体の配列を作成
        final_proba = np.zeros(len(self.y))
        final_proba[self.stage2_mask] = y_prob_s2
        
        # 動的閾値探索: Stage 2対象データのみで計算
        y_s2_true = self.y[self.stage2_mask]
        precisions, recalls, thresholds = precision_recall_curve(y_s2_true, y_prob_s2)
        
        target_recalls = [0.99, 0.98, 0.95]
        self.dynamic_results = {}
        
        print("\n   📊 動的閾値評価:")
        for target_recall in target_recalls:
            # recalls は降順なので、target_recall 以上の最初のインデックスを探す
            idx = np.where(recalls >= target_recall)[0]
            if len(idx) > 0:
                idx = idx[-1]  # recallsは降順なので最後のインデックス
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
        
        # 固定閾値0.5での評価（従来との比較用）
        y_pred = (final_proba >= 0.5).astype(int)
        
        self.final_precision = precision_score(self.y, y_pred) if y_pred.sum() > 0 else 0
        self.final_recall = recall_score(self.y, y_pred)
        self.final_f1 = f1_score(self.y, y_pred)
        self.final_auc = roc_auc_score(self.y, final_proba)
        self.final_proba = final_proba  # レポート用に保持
        
        # Baseline (Stage 1 単独 閾値0.5)
        y_pred_bl = (self.oof_proba_stage1 >= 0.5).astype(int)
        self.baseline_precision = precision_score(self.y, y_pred_bl)
        self.baseline_recall = recall_score(self.y, y_pred_bl)
        
        print(f"\n   [閾値0.5] Precision: {self.final_precision:.4f}, Recall: {self.final_recall:.4f}, F1: {self.final_f1:.4f}")
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
            'dynamic_recall_98_precision': self.dynamic_results.get(0.98, {}).get('precision', 0),
            'dynamic_recall_98_threshold': self.dynamic_results.get(0.98, {}).get('threshold', 0),
        }
    
    def evaluate_test_set(self):
        """
        テストセットでの最終評価
        学習に使用していない完全に独立したデータで汎化性能を確認する
        """
        print("\n📈 テストセット評価 (Hold-Out)")
        
        # Stage 1: 全Foldのモデルでアンサンブル予測
        test_proba_stage1 = np.zeros(len(self.y_test))
        for fold_models in self.stage1_models:
            for model in fold_models:
                test_proba_stage1 += model.predict_proba(self.X_test)[:, 1]
        test_proba_stage1 /= (self.n_folds * self.n_seeds)
        
        # Stage 1閾値を適用してフィルタリング
        test_stage2_mask = test_proba_stage1 >= self.threshold_stage1
        n_candidates = test_stage2_mask.sum()
        n_pos_in_candidates = self.y_test[test_stage2_mask].sum()
        
        print(f"   Stage 1 フィルタリング後: {n_candidates:,} / {len(self.y_test):,}")
        print(f"   正例残存: {n_pos_in_candidates:,} / {self.y_test.sum():,}")
        
        if n_candidates == 0:
            print("   ⚠️ Stage 2に進むデータがありません")
            self.test_results = {'error': 'No candidates after Stage 1'}
            return self.test_results
        
        # Stage 2用の特徴量生成 (テスト時は保存済みカテゴリを使用)
        X_test_s2 = self.generate_stage2_features(
            self.X_test[test_stage2_mask].copy(),
            test_proba_stage1[test_stage2_mask],
            fit_categories=False  # テスト時は学習時のカテゴリマッピングを使用
        )
        y_test_s2 = self.y_test[test_stage2_mask]
        
        # Stage 2: 全Foldのモデルでアンサンブル予測 (Focal Loss使用時はraw_score)
        test_proba_stage2 = np.zeros(len(y_test_s2))
        for model in self.stage2_models:
            raw_score = model.predict(X_test_s2, raw_score=True)
            proba = 1.0 / (1.0 + np.exp(-raw_score))
            test_proba_stage2 += proba
        test_proba_stage2 /= self.n_folds
        
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
        
        # 固定閾値0.5での評価
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
        
        report_content = f"""# Focal Loss 実験レポート (Optuna最適化版)

**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**実行時間**: {elapsed_sec:.1f}秒

## パラメータ設定 (Optuna最適化)

| パラメータ | 値 |
|-----------|----| 
| Focal Alpha | {self.focal_alpha:.4f} |
| Focal Gamma | {self.focal_gamma:.4f} |
| num_leaves | 127 |
| max_depth | 6 |
| min_child_samples | 44 |
| reg_alpha | 2.3897 |
| reg_lambda | 2.2842 |
| colsample_bytree | 0.8646 |
| subsample | 0.6328 |
| learning_rate | 0.0477 |
| Stage 1 Recall Target | {self.stage1_recall_target:.0%} |
| Under-sampling Ratio | 1:{int(self.undersample_ratio)} |
| Test Set Ratio | {self.test_size:.0%} |

## 結果サマリ

### Stage 1
- **閾値**: {results['stage1_threshold']:.4f}
- **Recall**: {results['stage1_recall']:.4f}
- **フィルタリング率**: {results['filter_rate']*100:.2f}%

### Stage 2 (Focal Loss) - CV OOF評価

#### 固定閾値 (0.5) での評価
| 指標 | 値 |
|------|----| 
| Precision | {results['final_precision']:.4f} |
| Recall | {results['final_recall']:.4f} |
| F1 | {results['final_f1']:.4f} |
| AUC | {results['final_auc']:.4f} |

#### 動的閾値での評価 (CV OOF)
| Target Recall | 閾値 | Precision |
|---------------|------|----------|
| 99% | {self.dynamic_results.get(0.99, {}).get('threshold', 0):.4f} | {self.dynamic_results.get(0.99, {}).get('precision', 0):.4f} |
| 98% | {self.dynamic_results.get(0.98, {}).get('threshold', 0):.4f} | {self.dynamic_results.get(0.98, {}).get('precision', 0):.4f} |
| 95% | {self.dynamic_results.get(0.95, {}).get('threshold', 0):.4f} | {self.dynamic_results.get(0.95, {}).get('precision', 0):.4f} |

## Baseline との比較 (CV OOF)

| 指標 | Baseline (Stage1) | Focal Loss (固定閾値) | 変化 |
|------|-------------------|----------------------|------|
| Precision | {results['baseline_precision']:.4f} | {results['final_precision']:.4f} | {results['precision_improvement_pct']:+.2f}% |
| Recall | {results['baseline_recall']:.4f} | {results['final_recall']:.4f} | - |

## 予測スコア分布

```
mean={pd.Series(self.final_proba[self.stage2_mask]).mean():.4f}
std={pd.Series(self.final_proba[self.stage2_mask]).std():.4f}
min={pd.Series(self.final_proba[self.stage2_mask]).min():.4f}
max={pd.Series(self.final_proba[self.stage2_mask]).max():.4f}
```

## 考察

- Focal Alpha={self.focal_alpha:.4f} は正例（死亡事故）の重みを調整
- Focal Gamma={self.focal_gamma:.4f} は難易度に応じた重み付け
- Optuna最適化により、Recall 99%時のPrecisionを最大化するパラメータを探索
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
        self.train_stage2()
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
    pipeline = TwoStageFinalPipeline()
    pipeline.run()
