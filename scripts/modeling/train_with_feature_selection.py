"""
特徴量選択後の再学習スクリプト
=============================
Permutation Importanceにより特定された「ノイズ特徴量」を除外して
Two-Stageモデル（Stage 1 + Stage 2 w/ DAE）を学習・評価します。

ベースモデル: train_two_stage_with_dae.py
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

# 削除する特徴量リスト (Importance <= 0.0)
DROP_FEATURES = [
    '地形', '昼夜', '車道幅員', 'hour', '道路線形', '中央分離帯施設等', '道路形状', 
    '地点コード', '祝日(発生年月日)', '衝突地点', 'ゾーン規制', 
    '一時停止規制　標識（当事者A）', '一時停止規制　表示（当事者A）', '信号機', 
    '一時停止規制　表示（当事者B）', '曜日(発生年月日)', '歩車道区分', 
    '用途別（当事者A）', '年齢（当事者A）', '地点　緯度（北緯）', '市区町村コード', 
    '天候', '地点　経度（東経）', 'day', 'month', 'road_type', 'year', 
    '路面状態', '速度規制（指定のみ）（当事者A）'
]

def get_focal_loss_lgb(alpha: float = 0.75, gamma: float = 1.0):
    """
    LightGBM用 Focal Loss を生成するファクトリー関数
    """
    def focal_loss_lgb(y_true, preds):
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
        hess = alpha_t * focal_weight * p * (1 - p)
        hess = np.maximum(hess, 1e-7)
        
        return grad, hess
    
    return focal_loss_lgb


class TwoStageSelectedFeaturesPipeline:
    """特徴量選択済み Two-Stage Model パイプライン"""
    
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
        dae_hidden_dim: int = 768,
        dae_epochs: int = 15,
        dae_swap_noise: float = 0.15,
        dae_batch_size: int = 32768,
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
        
        self.output_dir = "results/two_stage_model/selected_features"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Storage for models
        self.stage1_models = []
        self.stage2_models = []
        self.dae_models = []
        
        print("=" * 60)
        print("特徴量選択済み Two-Stage モデル学習")
        print(f"削除特徴量数: {len(DROP_FEATURES)}")
        print(f"Stage 1 Recall Target: {self.stage1_recall_target:.0%}")
        print("=" * 60)
    
    def load_data(self):
        """データ読み込みと特徴量削除"""
        print("\n📂 データ読み込み中...")
        self.df = pd.read_csv(self.data_path)
        
        y_all = self.df[self.target_col].values
        X_all = self.df.drop(columns=[self.target_col])
        
        if '発生日時' in X_all.columns:
            X_all = X_all.drop(columns=['発生日時'])
            
        # 特徴量削除
        available_drop_features = [c for c in DROP_FEATURES if c in X_all.columns]
        if available_drop_features:
            print(f"   🗑️  {len(available_drop_features)} 個の特徴量を削除します...")
            X_all = X_all.drop(columns=available_drop_features)
        
        self.feature_names = list(X_all.columns)
        print(f"   残存特徴量数: {len(self.feature_names)}")
        print(f"   例: {self.feature_names[:5]} ...")
        
        # Train/Test分割 (層化抽出)
        self.X, self.X_test, self.y, self.y_test = train_test_split(
            X_all, y_all, test_size=self.test_size, 
            random_state=self.random_state, stratify=y_all
        )
        
        print(f"\n📊 データ分割 (Train: {1-self.test_size:.0%} / Test: {self.test_size:.0%})")
        
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
        
        gc.collect()
    
    def train_stage1(self):
        """Stage 1: LightGBM + Under-sampling + Multi-Seed"""
        print(f"\n🌿 Stage 1: LightGBM + Under-sampling (1:{int(self.undersample_ratio)})")
        
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
            fold_logits = np.zeros(len(val_idx))
            
            for seed in range(self.n_seeds):
                np.random.seed(self.random_state + seed)
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
                
                raw_score = model.predict_proba(X_val)[:, 1]
                raw_score = np.clip(raw_score, 1e-15, 1 - 1e-15)
                logits = np.log(raw_score / (1 - raw_score))
                fold_logits += logits / self.n_seeds
                fold_models.append(model)
                feature_importances += model.feature_importances_ / (self.n_folds * self.n_seeds)
            
            self.oof_proba_stage1[val_idx] = expit(fold_logits)
            if not hasattr(self, 'oof_logits_stage1'):
                self.oof_logits_stage1 = np.zeros(len(self.y))
            self.oof_logits_stage1[val_idx] = fold_logits
            self.stage1_models.append(fold_models)
            
            del X_train, X_val
            gc.collect()
        
        self.feature_importance_df = pd.DataFrame({
            'feature': self.feature_names, 'importance': feature_importances
        }).sort_values('importance', ascending=False)
        self.top_features = self.feature_importance_df.head(10)['feature'].tolist()
        
        oof_auc = roc_auc_score(self.y, self.oof_proba_stage1)
        print(f"   Stage 1 OOF AUC: {oof_auc:.4f}")

    def find_recall_threshold(self):
        """Recall目標を達成する閾値を探索"""
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
        
        print(f"   閾値: {self.threshold_stage1:.4f}, Recall: {self.stage1_recall:.4f}")
        print(f"   フィルタリング: {self.filter_rate:.2%}")
        self.stage2_mask = self.oof_proba_stage1 >= self.threshold_stage1

    def generate_stage2_features(self, X_subset, logits_stage1_subset, fit_categories=True):
        X_out = X_subset.copy()
        if self.use_prob_stage1:
            X_out['logits_stage1'] = logits_stage1_subset
        
        # Interaction Features (if any top features remain)
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
                        interaction_values = interaction_values.apply(lambda x: x if x in known_cats else '__UNKNOWN__')
                        X_out[name] = pd.Categorical(interaction_values, dtype=self.interaction_categories[name])
                    else:
                        X_out[name] = interaction_values.astype('category')
        return X_out

    def train_stage2_with_dae(self):
        """Stage 2: LightGBM + DAE"""
        print("\n🌿 Stage 2: LightGBM + DAE特徴量 (5-Fold CV)")
        
        X_s2_base = self.generate_stage2_features(
            self.X[self.stage2_mask].copy(),
            self.oof_logits_stage1[self.stage2_mask],
            fit_categories=True
        ).reset_index(drop=True)
        
        y_s2_full = self.y[self.stage2_mask]
        
        self.oof_proba_stage2 = np.zeros(len(y_s2_full))
        self.stage2_models = []
        self.dae_models = []
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
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
            
            # DAE
            dae_numeric_cols = self.numeric_cols + (['logits_stage1'] if self.use_prob_stage1 else [])
            # Filter cols that actually exist in X_train_base
            dae_numeric_cols = [c for c in dae_numeric_cols if c in X_train_base.columns]
            dae_cat_cols = [c for c in self.categorical_cols if c in X_train_base.columns]
            
            dae = DAEFeatureExtractor(
                numeric_cols=dae_numeric_cols,
                cat_cols=dae_cat_cols,
                bottleneck_dim=self.dae_bottleneck_dim,
                hidden_dim=self.dae_hidden_dim,
                epochs=self.dae_epochs,
                swap_noise_rate=self.dae_swap_noise,
                batch_size=self.dae_batch_size,
                verbose=False,
                n_workers=4
            )
            dae.fit(X_train_base)
            
            dae_train_features = dae.transform(X_train_base)
            dae_val_features = dae.transform(X_val_base)
            
            dae_cols = [f'dae_{i}' for i in range(self.dae_bottleneck_dim)]
            dae_train_df = pd.DataFrame(dae_train_features, columns=dae_cols)
            dae_val_df = pd.DataFrame(dae_val_features, columns=dae_cols)
            
            X_train_full = pd.concat([X_train_base.reset_index(drop=True), dae_train_df], axis=1)
            X_val_full = pd.concat([X_val_base.reset_index(drop=True), dae_val_df], axis=1)
            
            model = lgb.LGBMClassifier(**lgb_params, random_state=self.random_state)
            model.fit(
                X_train_full, y_train,
                eval_set=[(X_val_full, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            raw_score = model.predict(X_val_full, raw_score=True)
            self.oof_proba_stage2[val_idx] = 1.0 / (1.0 + np.exp(-raw_score))
            
            self.stage2_models.append(model)
            self.dae_models.append(dae)
            
            gc.collect()
            
        print(f"   Stage 2 OOF AUC: {roc_auc_score(y_s2_full, self.oof_proba_stage2):.4f}")

    def evaluate(self):
        """最終評価"""
        y_s2 = self.y[self.stage2_mask]
        self.final_proba = np.zeros(len(self.y))
        self.final_proba[self.stage2_mask] = self.oof_proba_stage2
        
        precisions, recalls, thresholds = precision_recall_curve(y_s2, self.oof_proba_stage2)
        
        self.dynamic_results = {}
        for target_recall in [0.99, 0.98, 0.95]:
            idx = np.where(recalls >= target_recall)[0]
            if len(idx) > 0:
                idx = idx[-1]
                best_thresh = thresholds[idx] if idx < len(thresholds) else 0.0
                best_prec = precisions[idx]
            else:
                best_thresh = 0.0
                best_prec = 0.0
            self.dynamic_results[target_recall] = {'threshold': best_thresh, 'precision': best_prec}
            
        y_pred = (self.final_proba >= 0.5).astype(int)
        self.final_precision = precision_score(self.y, y_pred) if y_pred.sum() > 0 else 0
        self.final_recall = recall_score(self.y, y_pred)
        self.final_f1 = f1_score(self.y, y_pred)
        self.final_auc = roc_auc_score(self.y, self.final_proba)
        
        return {
            'final_precision': self.final_precision,
            'final_recall': self.final_recall,
            'final_f1': self.final_f1,
            'final_auc': self.final_auc,
            'dynamic_recall_99_precision': self.dynamic_results.get(0.99, {}).get('precision', 0),
        }

    def generate_report(self, results):
        report_path = os.path.join(self.output_dir, "experiment_report.md")
        content = f"""# 特徴量選択後 Two-Stage モデル実験レポート
        
**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 設定
- 削除特徴量: {len(DROP_FEATURES)}件
- Stage 1 Recall Target: {self.stage1_recall_target:.0%}

## 結果
### CV OOF (閾値 0.5)
- Precision: {results['final_precision']:.4f}
- Recall: {results['final_recall']:.4f}
- F1: {results['final_f1']:.4f}
- AUC: {results['final_auc']:.4f}

### 動的閾値 (CV OOF)
- Recall 99%: Precision {results.get('dynamic_recall_99_precision', 0):.4f}
"""
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return report_path

    def run(self):
        self.load_data()
        self.train_stage1()
        self.find_recall_threshold()
        self.train_stage2_with_dae()
        results = self.evaluate()
        self.generate_report(results)
        return results

if __name__ == "__main__":
    pipeline = TwoStageSelectedFeaturesPipeline()
    pipeline.run()
