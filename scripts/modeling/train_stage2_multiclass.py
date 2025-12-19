"""
Stage 2 二値分類（負傷 vs 死亡）パイプライン
============================================
Stage 1: LightGBM (Binary) + Under-sampling + 3-Seed Averaging  (既存と同じ)
Stage 2: LightGBM (Binary: 0=負傷, 1=死亡)

目的変数:
  0: 負傷 (死者数==0)
  1: 死亡 (死者数>0)

データリーク防止:
  - 特徴量データ (X) には一切手を加えない
  - 負傷者数は生データから**ラベル生成用にのみ**抽出
  - Xに負傷者数を含めないことをアサートで保証
"""

import pandas as pd
import numpy as np
import joblib
import os
import gc
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, 
    precision_recall_curve, accuracy_score, confusion_matrix,
    classification_report
)
import lightgbm as lgb
from scipy.special import expit, softmax
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# リーク防止チェック関数
# ============================================================================
FORBIDDEN_COLUMNS = [
    '事故内容',
    '人身損傷程度（当事者A）', '人身損傷程度（当事者B）',
    '負傷者数',  # これはラベル生成用、特徴量にしてはいけない
    '車両の損壊程度（当事者A）', '車両の損壊程度（当事者B）',
    '車両の衝突部位（当事者A）', '車両の衝突部位（当事者B）',
    'エアバッグの装備（当事者A）', 'エアバッグの装備（当事者B）',
    'サイドエアバッグの装備（当事者A）', 'サイドエアバッグの装備（当事者B）',
]

def check_no_leakage(X: pd.DataFrame, context: str = ""):
    """
    特徴量データフレームにリーク列が含まれていないことを確認
    含まれていたらAssertionErrorを発生
    """
    leaked = [col for col in FORBIDDEN_COLUMNS if col in X.columns]
    assert len(leaked) == 0, f"[LEAKAGE ERROR] {context}: リーク列が検出されました: {leaked}"
    print(f"   ✅ リークチェック通過 ({context}): {len(X.columns)}列, リーク列なし")


# ============================================================================
# メインパイプライン
# ============================================================================
class TwoStageBinaryPipeline:
    """2段階モデル + 二値分類 Stage 2 パイプライン (負傷 vs 死亡)"""
    
    def __init__(
        self,
        features_path: str = "data/processed/honhyo_clean_with_features.csv",
        raw_data_path: str = "honhyo_all/csv/honhyo_all_with_datetime.csv",
        target_col: str = "死者数",
        n_folds: int = 5,
        random_state: int = 42,
        stage1_recall_target: float = 0.95,
        undersample_ratio: float = 2.0,
        n_seeds: int = 3,
        top_k_interactions: int = 5,
        test_size: float = 0.2,
        # オプション
        use_logits_stage1: bool = True,
    ):
        self.features_path = features_path
        self.raw_data_path = raw_data_path
        self.target_col = target_col
        self.n_folds = n_folds
        self.random_state = random_state
        self.stage1_recall_target = stage1_recall_target
        self.undersample_ratio = undersample_ratio
        self.n_seeds = n_seeds
        self.top_k_interactions = top_k_interactions
        self.test_size = test_size
        self.use_logits_stage1 = use_logits_stage1
        
        self.output_dir = "results/two_stage_model/binary_pipeline"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # モデル保存用
        self.stage1_models = []
        self.stage2_models = []
        
        print("=" * 60)
        print("2段階モデル + 二値分類パイプライン (負傷 vs 死亡)")
        print(f"Stage 1: 1:{int(self.undersample_ratio)} Under-sampling, Recall {self.stage1_recall_target:.0%}")
        print(f"Stage 2: Binary (0=負傷, 1=死亡)")
        print(f"Test Set: {self.test_size:.0%}")
        print("=" * 60)
    
    def load_data(self):
        """
        データ読み込みと多クラスラベル生成
        
        重要: 特徴量データには一切手を加えず、負傷者数は生データからラベル作成用にのみ取得
        """
        print("\n📂 データ読み込み中...")
        
        # 1. 特徴量データ読み込み
        df_features = pd.read_csv(self.features_path)
        print(f"   特徴量データ: {len(df_features):,}件, {len(df_features.columns)}列")
        
        # 2. 生データから負傷者数のみ読み込み (他の列はリークの危険があるため読み込まない)
        df_raw = pd.read_csv(self.raw_data_path, usecols=['負傷者数'])
        print(f"   生データ（負傷者数のみ）: {len(df_raw):,}件")
        
        # 3. 整合性チェック
        assert len(df_features) == len(df_raw), \
            f"[ERROR] 行数不一致: 特徴量={len(df_features)}, 生データ={len(df_raw)}"
        print("   ✅ 行数一致確認完了")
        
        # 4. 多クラスラベル生成
        y_fatal = (df_features[self.target_col] > 0).astype(int)
        y_injury = (df_raw['負傷者数'] > 0).astype(int)
        
        # 3クラスに分類
        # 2: 死亡, 1: 負傷(死亡なし), 0: 無傷
        y_multiclass = np.zeros(len(df_features), dtype=np.int32)
        y_multiclass[y_fatal == 1] = 2  # 死亡
        y_multiclass[(y_fatal == 0) & (y_injury == 1)] = 1  # 負傷
        # 残りは0（無傷）のまま
        
        self.y_multiclass = y_multiclass
        
        # クラス分布表示
        print("\n📊 多クラスラベル分布:")
        for cls in [0, 1, 2]:
            count = (y_multiclass == cls).sum()
            pct = count / len(y_multiclass) * 100
            label = {0: "無傷/軽微", 1: "負傷", 2: "死亡"}[cls]
            print(f"   クラス {cls} ({label}): {count:,} ({pct:.2f}%)")
        
        # 5. 二値ラベル (Stage 1用: 死亡 vs その他)
        self.y_binary = (df_features[self.target_col] > 0).astype(int).values
        
        # 6. 特徴量抽出
        X_all = df_features.drop(columns=[self.target_col])
        if '発生日時' in X_all.columns:
            X_all = X_all.drop(columns=['発生日時'])
        
        # 7. リークチェック（最重要）
        check_no_leakage(X_all, "特徴量データ読み込み後")
        
        # Train/Test分割 (層化: 多クラスラベルで分割)
        self.X, self.X_test, self.y_mc, self.y_mc_test, \
        self.y_bin, self.y_bin_test = train_test_split(
            X_all, self.y_multiclass, self.y_binary,
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=self.y_multiclass
        )
        
        print(f"\n📊 データ分割 (Train: {1-self.test_size:.0%} / Test: {self.test_size:.0%})")
        print(f"   Train: {len(self.y_mc):,} (死亡: {(self.y_mc==2).sum():,})")
        print(f"   Test:  {len(self.y_mc_test):,} (死亡: {(self.y_mc_test==2).sum():,})")
        
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
        
        # 最終リークチェック
        check_no_leakage(self.X, "Train/Test分割後 (Train)")
        check_no_leakage(self.X_test, "Train/Test分割後 (Test)")
        
        gc.collect()
    
    def train_stage1(self):
        """Stage 1: LightGBM (Binary) + Under-sampling + Multi-Seed"""
        print(f"\n🌿 Stage 1: LightGBM (Binary) + Under-sampling (1:{int(self.undersample_ratio)}) + {self.n_seeds}-Seed Averaging")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.oof_proba_stage1 = np.zeros(len(self.y_bin))
        self.oof_logits_stage1 = np.zeros(len(self.y_bin))
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
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y_bin)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            X_train_full = self.X.iloc[train_idx]
            X_val = self.X.iloc[val_idx]
            y_train_full = self.y_bin[train_idx]
            y_val = self.y_bin[val_idx]
            
            fold_models = []
            fold_logits = np.zeros(len(val_idx))
            
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
                
                # Logits取得
                raw_score = model.predict_proba(X_val)[:, 1]
                raw_score = np.clip(raw_score, 1e-15, 1 - 1e-15)
                logits = np.log(raw_score / (1 - raw_score))
                fold_logits += logits / self.n_seeds
                fold_models.append(model)
                feature_importances += model.feature_importances_ / (self.n_folds * self.n_seeds)
            
            self.oof_logits_stage1[val_idx] = fold_logits
            self.oof_proba_stage1[val_idx] = expit(fold_logits)
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
        oof_auc = roc_auc_score(self.y_bin, self.oof_proba_stage1)
        print(f"   OOF (閾値0.5): Prec={precision_score(self.y_bin, oof_pred):.4f}, "
              f"Rec={recall_score(self.y_bin, oof_pred):.4f}, AUC={oof_auc:.4f}")
    
    def find_recall_threshold(self):
        """Recall目標を達成する閾値を探索（高い閾値から下げていく）"""
        for thresh in np.arange(0.5, 0.0, -0.001):
            y_pred = (self.oof_proba_stage1 >= thresh).astype(int)
            recall = recall_score(self.y_bin, y_pred)
            if recall >= self.stage1_recall_target:
                self.threshold_stage1 = thresh
                break
        else:
            self.threshold_stage1 = 0.001
        
        y_pred_final = (self.oof_proba_stage1 >= self.threshold_stage1).astype(int)
        self.stage1_recall = recall_score(self.y_bin, y_pred_final)
        n_candidates = y_pred_final.sum()
        self.filter_rate = 1 - (n_candidates / len(self.y_bin))
        n_filtered = len(self.y_bin) - n_candidates
        
        print(f"   閾値: {self.threshold_stage1:.4f}, Recall: {self.stage1_recall:.4f}")
        print(f"   [Result] フィルタリング: {n_filtered:,} 件除外 ({self.filter_rate:.2%})")
        print(f"   [Result] 残存データ: {n_candidates:,} 件 (Stage 2 候補)")
        print(f"   [Result] 死亡事例残存: {self.y_bin[self.oof_proba_stage1 >= self.threshold_stage1].sum():,} / {self.y_bin.sum():,}")
        
        self.stage2_mask = self.oof_proba_stage1 >= self.threshold_stage1
        
        # [Check] Stage 1 通過率 (クラス別) - 負傷事故が過剰に除外されていないか確認
        passed_counts = np.bincount(self.y_mc[self.stage2_mask], minlength=3)
        total_counts = np.bincount(self.y_mc, minlength=3)
        
        print("\n   [Check] Stage 1 通過率 (クラス別):")
        class_labels = {0: "無傷/軽微", 1: "負傷", 2: "死亡"}
        self.class_pass_rates = {}
        for cls in [0, 1, 2]:
            ratio = passed_counts[cls] / total_counts[cls] if total_counts[cls] > 0 else 0
            self.class_pass_rates[cls] = ratio
            print(f"     Class {cls} ({class_labels[cls]}): {passed_counts[cls]:,} / {total_counts[cls]:,} ({ratio:.1%})")
    
    def generate_stage2_features(self, X_subset, logits_stage1_subset, fit_categories=True):
        """Stage 2用特徴量生成"""
        X_out = X_subset.copy()
        
        # logits_stage1 追加 (オプション)
        if self.use_logits_stage1:
            X_out['logits_stage1'] = logits_stage1_subset
        
        # Categorical Interaction Features
        top_cat_features = [f for f in self.top_features if f in self.categorical_cols]
        
        if fit_categories:
            self.interaction_categories = {}
        
        for i, f1 in enumerate(top_cat_features[:self.top_k_interactions]):
            for f2 in top_cat_features[i+1:self.top_k_interactions]:
                name = f"{f1}_{f2}"
                interaction_values = X_subset[f1].astype(str) + "_" + X_subset[f2].astype(str)
                
                if fit_categories:
                    cat_type = pd.CategoricalDtype(categories=list(interaction_values.unique()) + ['__UNKNOWN__'])
                    self.interaction_categories[name] = cat_type
                    X_out[name] = pd.Categorical(interaction_values, dtype=cat_type)
                else:
                    if hasattr(self, 'interaction_categories') and name in self.interaction_categories:
                        known_cats = set(self.interaction_categories[name].categories)
                        interaction_values = interaction_values.apply(
                            lambda x: x if x in known_cats else '__UNKNOWN__'
                        )
                        X_out[name] = pd.Categorical(interaction_values, dtype=self.interaction_categories[name])
                    else:
                        X_out[name] = interaction_values.astype('category')
        
        return X_out
    
    def train_stage2_binary(self):
        """Stage 2: 二値分類 LightGBM (0=負傷, 1=死亡)
        
        Stage 1を通過した「紛らわしい負傷（Hard Negatives）」と「死亡」を識別する。
        データセットに無傷（Class 0）が存在しないため、純粋な二値分類として定式化。
        """
        print("\n🌿 Stage 2: LightGBM Binary Classification (5-Fold CV)")
        print(f"   クラス: 0=負傷 (Hard Negatives), 1=死亡")
        
        # Stage 2用データ (Stage 1でフィルタリング後)
        X_s2_base = self.generate_stage2_features(
            self.X[self.stage2_mask].copy(),
            self.oof_logits_stage1[self.stage2_mask],
            fit_categories=True
        ).reset_index(drop=True)
        
        # 二値ラベル: 死亡(1) vs 負傷(0)
        # y_mc から: Class 2 -> 1 (死亡), Class 1 -> 0 (負傷), Class 0 -> 除外 (存在しない)
        y_s2_mc = self.y_mc[self.stage2_mask]
        y_s2_binary = (y_s2_mc == 2).astype(int)  # 死亡=1, 負傷=0
        
        # クラス分布
        n_pos = y_s2_binary.sum()
        n_neg = len(y_s2_binary) - n_pos
        print(f"   Stage 2 データ: {len(y_s2_binary):,}")
        print(f"      負傷 (Class 0): {n_neg:,}")
        print(f"      死亡 (Class 1): {n_pos:,}")
        print(f"      正例比率: {n_pos / len(y_s2_binary) * 100:.2f}%")
        
        # OOF予測保存 (1次元: 死亡確率)
        self.oof_proba_stage2 = np.zeros(len(y_s2_binary))
        self.stage2_models = []
        self.stage2_feature_names = list(X_s2_base.columns)
        stage2_feature_importances = np.zeros(len(self.stage2_feature_names))
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # LightGBM パラメータ (scale_pos_weightは各Fold内で計算)
        lgb_base_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'num_leaves': 63,
            'max_depth': -1,
            'min_child_samples': 50,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'colsample_bytree': 0.8,
            'subsample': 0.7,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'n_jobs': -1
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_s2_base, y_s2_binary)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            
            X_train = X_s2_base.iloc[train_idx]
            X_val = X_s2_base.iloc[val_idx]
            y_train = y_s2_binary[train_idx]
            y_val = y_s2_binary[val_idx]
            
            # 各Foldの訓練データに基づいてscale_pos_weightを計算 (Best Practice)
            n_pos_fold = y_train.sum()
            n_neg_fold = len(y_train) - n_pos_fold
            scale_pos_weight = n_neg_fold / n_pos_fold if n_pos_fold > 0 else 1.0
            
            
            # モデル保存用パス
            model_dir = "results/models/lgb_stage2"
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"lgb_fold{fold+1}.pkl")
            
            # 途中再開ロジック
            if os.path.exists(model_path):
                print(f"   📥 既存のモデルが見つかりました、学習をスキップしてロードします: {model_path}")
                model = joblib.load(model_path)
            else:
                model = lgb.LGBMClassifier(
                    **lgb_base_params, 
                    scale_pos_weight=scale_pos_weight,
                    random_state=self.random_state
                )
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                joblib.dump(model, model_path)
                print(f"   💾 モデルを保存しました: {model_path}")
            
            # OOF予測 (死亡確率)
            proba = model.predict_proba(X_val)[:, 1]
            self.oof_proba_stage2[val_idx] = proba
            
            # 特徴量重要度を集計
            stage2_feature_importances += model.feature_importances_ / self.n_folds
            
            self.stage2_models.append(model)
            
            del X_train, X_val
            gc.collect()
        
        # Stage 2 OOF評価
        oof_pred = (self.oof_proba_stage2 >= 0.5).astype(int)
        oof_acc = accuracy_score(y_s2_binary, oof_pred)
        print(f"\n   Stage 2 OOF Accuracy: {oof_acc:.4f}")
        print(f"   Confusion Matrix:\n{confusion_matrix(y_s2_binary, oof_pred)}")
        
        # AUC評価
        auc_fatal = roc_auc_score(y_s2_binary, self.oof_proba_stage2)
        print(f"   Fatal AUC: {auc_fatal:.4f}")
        print(f"   Precision(0.5): {precision_score(y_s2_binary, oof_pred):.4f}")
        print(f"   Recall(0.5): {recall_score(y_s2_binary, oof_pred):.4f}")
        
        # 二値ラベルを保存 (evaluateで使用)
        self.y_s2_binary = y_s2_binary
        
        # Stage 2 特徴量重要度を保存
        self.stage2_feature_importance_df = pd.DataFrame({
            'feature': self.stage2_feature_names,
            'importance': stage2_feature_importances
        }).sort_values('importance', ascending=False)
        
        print(f"\n   📊 Stage 2 Top 10 特徴量:")
        for i, row in self.stage2_feature_importance_df.head(10).iterrows():
            print(f"      {row['feature']}: {row['importance']:.0f}")
    
    def evaluate(self):
        """最終評価 (CV OOF) - Binary Classification with Dynamic Threshold"""
        print("\n📈 最終評価 (Cross Validation OOF)")
        
        y_s2_bin = self.y_s2_binary
        prob_fatal = self.oof_proba_stage2
        
        # Precision-Recall Curve
        precisions, recalls, thresholds = precision_recall_curve(y_s2_bin, prob_fatal)
        
        # 1. Best F1 Score 探索
        numerator = 2 * precisions * recalls
        denominator = precisions + recalls
        f1_scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        
        best_f1_idx = np.argmax(f1_scores)
        self.best_f1_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_f1_idx]
        best_prec = precisions[best_f1_idx]
        best_rec = recalls[best_f1_idx]
        
        print(f"\n   🏆 Best F1 Score: {best_f1:.4f} (閾値: {self.best_f1_threshold:.4f})")
        print(f"      Precision: {best_prec:.4f}, Recall: {best_rec:.4f}")
        
        # Confusion Matrix at Best Threshold
        y_pred_best = (prob_fatal >= self.best_f1_threshold).astype(int)
        conf_mat = confusion_matrix(y_s2_bin, y_pred_best)
        print(f"      Confusion Matrix:\n{conf_mat}")
        
        # 2. Recall Oriented Thresholds
        self.dynamic_results = {}
        target_recalls = [0.99, 0.98, 0.95]
        
        print("\n   📊 Recall重視の評価:")
        for target_recall in target_recalls:
            idx = np.where(recalls >= target_recall)[0]
            if len(idx) > 0:
                idx = idx[-1]
                thresh = thresholds[idx] if idx < len(thresholds) else 0.0
                prec = precisions[idx]
            else:
                thresh = 0.0
                prec = 0.0
            
            self.dynamic_results[target_recall] = {'threshold': thresh, 'precision': prec}
            print(f"      Recall ~{target_recall:.0%}: 閾値={thresh:.4f}, Precision={prec:.4f}")
        
        # Global Metrics (Best F1 Threshold)
        y_bin_all = (self.y_mc == 2).astype(int)  # 全体に対するラベル
        final_proba = np.zeros(len(self.y_mc))
        final_proba[self.stage2_mask] = prob_fatal
        y_pred_global = (final_proba >= self.best_f1_threshold).astype(int)
        
        self.final_precision = precision_score(y_bin_all, y_pred_global)
        self.final_recall = recall_score(y_bin_all, y_pred_global)
        self.final_f1 = f1_score(y_bin_all, y_pred_global)
        self.final_auc = roc_auc_score(y_bin_all, final_proba)
        
        print(f"\n   [全体評価 @ Best Thresh] Precision: {self.final_precision:.4f}, Recall: {self.final_recall:.4f}, F1: {self.final_f1:.4f}")
        print(f"   [全体AUC]: {self.final_auc:.4f}")
        
        # アンサンブル用OOF保存
        oof_df = pd.DataFrame({
            'index': self.X[self.stage2_mask].index,
            'true_label': y_binary_stage2,
            'prob': prob_fatal
        })
        os.makedirs('results/oof', exist_ok=True)
        oof_df.to_csv('results/oof/oof_stage2_lightgbm.csv', index=False)
        print("\n   💾 OOF予測を保存しました: results/oof/oof_stage2_lightgbm.csv")
        
        return {
            'stage1_threshold': self.threshold_stage1,
            'stage1_recall': self.stage1_recall,
            'filter_rate': self.filter_rate,
            'best_f1_threshold': self.best_f1_threshold,
            'best_f1': best_f1,
            'best_f1_precision': best_prec,
            'best_f1_recall': best_rec,
            'final_precision': self.final_precision,
            'final_recall': self.final_recall,
            'final_f1': self.final_f1,
            'final_auc': self.final_auc,
            'recall_99_precision': self.dynamic_results[0.99]['precision'],
            'recall_95_precision': self.dynamic_results[0.95]['precision'],
        }
    
    def evaluate_test_set(self):
        """テストセットでの最終評価"""
        print("\n📈 テストセット評価 (Hold-Out)")
        
        # Stage 1: アンサンブル予測
        test_logits_stage1 = np.zeros(len(self.y_mc_test))
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
        n_fatal_in_candidates = (self.y_mc_test[test_stage2_mask] == 2).sum()
        
        print(f"   Stage 1 フィルタリング後: {n_candidates:,} / {len(self.y_mc_test):,}")
        print(f"   死亡事例残存: {n_fatal_in_candidates:,} / {(self.y_mc_test==2).sum():,}")
        
        if n_candidates == 0:
            print("   ⚠️ Stage 2に進むデータがありません")
            return {'error': 'No candidates after Stage 1'}
        
        # Stage 2用特徴量
        X_test_s2 = self.generate_stage2_features(
            self.X_test[test_stage2_mask].copy(),
            test_logits_stage1[test_stage2_mask],
            fit_categories=False
        )
        y_test_s2_mc = self.y_mc_test[test_stage2_mask]
        y_test_bin = (y_test_s2_mc == 2).astype(int)  # 死亡=1, 負傷=0
        
        # Stage 2: アンサンブル予測 (Binary)
        test_proba_stage2 = np.zeros(len(y_test_bin))
        for model in self.stage2_models:
            proba = model.predict_proba(X_test_s2)[:, 1]  # 死亡確率
            test_proba_stage2 += proba / self.n_folds
        
        # テスト評価
        prob_fatal = test_proba_stage2
        
        precisions, recalls, thresholds = precision_recall_curve(y_test_bin, prob_fatal)
        
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
        
        # CVでの最適閾値を適用
        cv_best_thresh = self.best_f1_threshold
        
        # Testセットでの Best F1 も探索（比較用）
        precisions, recalls, thresholds = precision_recall_curve(y_test_bin, prob_fatal)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-15)
        test_best_idx = np.argmax(f1_scores)
        test_best_f1 = f1_scores[test_best_idx]
        test_best_thresh = thresholds[test_best_idx] if test_best_idx < len(thresholds) else 0.5
        
        print(f"   🏆 Test Best F1: {test_best_f1:.4f} (Ideal Threshold: {test_best_thresh:.4f})")
        
        # CV閾値での評価
        y_test_pred_cv = (prob_fatal >= cv_best_thresh).astype(int)
        conf_mat = confusion_matrix(y_test_bin, y_test_pred_cv)
        print(f"\n   [CV閾値適用 ({cv_best_thresh:.4f})] Confusion Matrix:\n{conf_mat}")
        
        # 全体メトリクス (CV閾値)
        final_test_proba = np.zeros(len(self.y_mc_test))
        final_test_proba[test_stage2_mask] = prob_fatal
        y_test_pred_global = (final_test_proba >= cv_best_thresh).astype(int)
        y_test_all_bin = (self.y_mc_test == 2).astype(int)
        
        test_precision = precision_score(y_test_all_bin, y_test_pred_global)
        test_recall = recall_score(y_test_all_bin, y_test_pred_global)
        test_f1 = f1_score(y_test_all_bin, y_test_pred_global)
        test_auc = roc_auc_score(y_test_all_bin, final_test_proba)
        
        print(f"   [全体評価] Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
        print(f"   [全体AUC]: {test_auc:.4f}")
        
        # アンサンブル用Test予測保存
        test_df = pd.DataFrame({
            'index': self.X_test[test_stage2_mask].index,
            'true_label': y_test_bin,
            'prob': prob_fatal
        })
        os.makedirs('results/test_preds', exist_ok=True)
        test_df.to_csv('results/test_preds/test_stage2_lightgbm.csv', index=False)
        print("\n   💾 Test予測を保存しました: results/test_preds/test_stage2_lightgbm.csv")
        
        return {
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'cv_threshold_used': cv_best_thresh,
            'ideal_test_best_f1': test_best_f1
        }
    
    def generate_report(self, results: dict, elapsed_sec: float):
        """実験レポートをMarkdownで出力"""
        report_path = os.path.join(self.output_dir, "experiment_report.md")
        
        report_content = f"""# 多クラス分類 Stage 2 実験レポート

**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**実行時間**: {elapsed_sec:.1f}秒

## モデル構成
- **Stage 1**: Binary Classification (死亡 vs その他)
- **Stage 2**: Multiclass Classification (0=無傷, 1=負傷, 2=死亡)
- **Objective**: multiclass (class_weight使用)

## 結果サマリ
        
### Stage 1 (Recall {self.stage1_recall_target:.0%})
- **閾値**: {results['stage1_threshold']:.4f}
- **Recall**: {results['stage1_recall']:.4f}
- **フィルタリング率**: {results['filter_rate']*100:.2f}%
- **負傷事故(Class 1) 通過率**: {self.class_pass_rates.get(1, 0)*100:.1f}%

### Stage 2 Binary Classification (CV OOF)

**Best F1 閾値 ({results['best_f1_threshold']:.4f}) での評価**:
| 指標 | 値 |
|------|----| 
| F1 Score | {results['best_f1']:.4f} |
| Precision | {results['best_f1_precision']:.4f} |
| Recall | {results['best_f1_recall']:.4f} |

**Overall Metrics (全体に対する評価)**:
| 指標 | 値 |
|------|----| 
| Final Precision | {results['final_precision']:.4f} |
| Final Recall | {results['final_recall']:.4f} |
| Final F1 | {results['final_f1']:.4f} |
| AUC | {results['final_auc']:.4f} |

### テストセット評価 (Hold-Out {self.test_size:.0%})

**CV最適閾値 ({results.get('cv_threshold_used', 0):.4f}) を適用**:
| 指標 | 値 |
|------|----| 
| Precision | {results.get('test_precision', 0):.4f} |
| Recall | {results.get('test_recall', 0):.4f} |
| F1 | {results.get('test_f1', 0):.4f} |
| AUC | {results.get('test_auc', 0):.4f} |

**参考: Test Ideal F1**: {results.get('ideal_test_best_f1', 0):.4f}

## 考察

- 多クラス分類により、モデルは「無傷」「負傷」「死亡」の3段階の重大性を学習
- P(Injury+) スコアで「明らかに無害な事故」を除外することで、Precision向上の余地あり
- Binary分類と比較して、死亡事故の特定精度が向上しているか要検証
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n   📄 レポート出力: {report_path}")
        return report_path
    
    def run(self):
        """パイプライン実行"""
        start = datetime.now()
        self.load_data()
        self.train_stage1()
        self.find_recall_threshold()
        self.train_stage2_binary()
        results = self.evaluate()
        
        # テストセット評価
        test_results = self.evaluate_test_set()
        results.update(test_results)
        
        elapsed_sec = (datetime.now() - start).total_seconds()
        results['elapsed_sec'] = elapsed_sec
        
        # 結果保存
        pd.DataFrame([results]).to_csv(os.path.join(self.output_dir, "final_results.csv"), index=False)
        self.feature_importance_df.to_csv(os.path.join(self.output_dir, "stage1_feature_importance.csv"), index=False)
        self.stage2_feature_importance_df.to_csv(os.path.join(self.output_dir, "stage2_feature_importance.csv"), index=False)
        
        # Markdown レポート生成
        self.generate_report(results, elapsed_sec)
        
        print("\n" + "=" * 60)
        print("✅ 完了！")
        print(f"   結果CSV: {self.output_dir}/final_results.csv")
        print(f"   レポートMD: {self.output_dir}/experiment_report.md")
        print("=" * 60)
        
        return results


if __name__ == "__main__":
    pipeline = TwoStageBinaryPipeline()
    pipeline.run()
