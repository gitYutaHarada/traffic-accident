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
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


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
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.n_folds = n_folds
        self.random_state = random_state
        self.stage1_recall_target = stage1_recall_target
        self.undersample_ratio = undersample_ratio
        self.n_seeds = n_seeds
        self.top_k_interactions = top_k_interactions
        
        self.output_dir = "results/two_stage_model/final_pipeline"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 60)
        print("2段階モデル 最終パイプライン")
        print(f"Stage 1: 1:{int(self.undersample_ratio)} Under-sampling, Recall {self.stage1_recall_target:.0%}")
        print("=" * 60)
    
    def load_data(self):
        """データ読み込み"""
        print("\n📂 データ読み込み中...")
        self.df = pd.read_csv(self.data_path)
        self.y = self.df[self.target_col].values
        self.X = self.df.drop(columns=[self.target_col])
        
        if '発生日時' in self.X.columns:
            self.X = self.X.drop(columns=['発生日時'])
        
        known_categoricals = [
            '都道府県コード', '市区町村コード', '警察署等コード',
            '昼夜', '天候', '地形', '路面状態', '道路形状', '信号機',
            '衝突地点', 'ゾーン規制', '中央分離帯施設等', '歩車道区分',
            '事故類型', '曜日(発生年月日)', '祝日(発生年月日)',
            'road_type', 'area_id', '地点コード'
        ]
        
        self.categorical_cols = []
        for col in self.X.columns:
            if col in known_categoricals or self.X[col].dtype == 'object':
                self.categorical_cols.append(col)
                self.X[col] = self.X[col].astype('category')
            else:
                self.X[col] = self.X[col].astype(np.float32)
        
        self.feature_names = list(self.X.columns)
        print(f"   正例: {self.y.sum():,} / {len(self.y):,}")
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
    
    def generate_stage2_features(self, X_subset, prob_stage1_subset):
        """Stage 2用特徴量生成"""
        X_out = X_subset.copy()
        
        # (a) prob_stage1 追加
        X_out['prob_stage1'] = prob_stage1_subset
        
        # (b) Categorical Interaction Features
        top_cat_features = [f for f in self.top_features if f in self.categorical_cols]
        
        for i, f1 in enumerate(top_cat_features[:self.top_k_interactions]):
            for f2 in top_cat_features[i+1:self.top_k_interactions]:
                name = f"{f1}_{f2}"
                X_out[name] = (X_subset[f1].astype(str) + "_" + X_subset[f2].astype(str)).astype('category')
        
        return X_out
    
    def train_stage2(self):
        """Stage 2: 高複雑度 + 強正則化"""
        print("\n🌿 Stage 2: High Complexity + Strong Regularization")
        
        X_s2 = self.generate_stage2_features(
            self.X[self.stage2_mask].copy(),
            self.oof_proba_stage1[self.stage2_mask]
        )
        y_s2 = self.y[self.stage2_mask]
        
        n_pos, n_neg = y_s2.sum(), len(y_s2) - y_s2.sum()
        scale_pos = n_neg / n_pos if n_pos > 0 else 1.0
        
        print(f"   Stage 2 データ: {len(y_s2):,} (Pos: {n_pos:,}, Neg: {n_neg:,})")
        print(f"   Top Features for Interaction: {self.top_features}")
        
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'num_leaves': 118,
            'max_depth': 8,
            'min_child_samples': 32,
            'reg_alpha': 7.289881162161227,
            'reg_lambda': 0.7394666125185072,
            'colsample_bytree': 0.7059404335368878,
            'subsample': 0.5385972873574277,
            'learning_rate': 0.048867878885592735,
            'scale_pos_weight': 1.1345607321720075,
            'is_unbalance': False,  # scale_pos_weightを使用するためFalse
            'n_estimators': 1000,   # 増やしておく（Early Stoppingはないが多めでOK）
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        self.stage2_model = lgb.LGBMClassifier(**lgb_params)
        self.stage2_model.fit(X_s2, y_s2)
        
        self.stage2_feature_names = list(X_s2.columns)
    
    def evaluate(self):
        """最終評価"""
        print("\n📈 最終評価")
        
        X_s2 = self.generate_stage2_features(
            self.X[self.stage2_mask].copy(),
            self.oof_proba_stage1[self.stage2_mask]
        )
        y_prob = self.stage2_model.predict_proba(X_s2)[:, 1]
        
        final_proba = np.zeros(len(self.y))
        final_proba[self.stage2_mask] = y_prob
        
        y_pred = (final_proba >= 0.5).astype(int)
        
        self.final_precision = precision_score(self.y, y_pred) if y_pred.sum() > 0 else 0
        self.final_recall = recall_score(self.y, y_pred)
        self.final_f1 = f1_score(self.y, y_pred)
        self.final_auc = roc_auc_score(self.y, final_proba)
        
        # Baseline (Stage 1 単独 閾値0.5)
        y_pred_bl = (self.oof_proba_stage1 >= 0.5).astype(int)
        self.baseline_precision = precision_score(self.y, y_pred_bl)
        self.baseline_recall = recall_score(self.y, y_pred_bl)
        
        print(f"   [最終] Precision: {self.final_precision:.4f}, Recall: {self.final_recall:.4f}, F1: {self.final_f1:.4f}")
        print(f"   [ベース(Stage1)] Precision: {self.baseline_precision:.4f}, Recall: {self.baseline_recall:.4f}")
        
        improvement = (self.final_precision - self.baseline_precision) / self.baseline_precision * 100 if self.baseline_precision > 0 else 0
        print(f"   Precision改善率: {improvement:+.2f}%")
        
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
            'precision_improvement_pct': improvement
        }
    
    def run(self):
        start = datetime.now()
        self.load_data()
        self.train_stage1()
        self.find_recall_threshold()
        self.train_stage2()
        results = self.evaluate()
        results['elapsed_sec'] = (datetime.now() - start).total_seconds()
        
        # 結果保存
        pd.DataFrame([results]).to_csv(os.path.join(self.output_dir, "final_results.csv"), index=False)
        self.feature_importance_df.to_csv(os.path.join(self.output_dir, "stage1_feature_importance.csv"), index=False)
        
        print("\n" + "=" * 60)
        print("✅ 完了！")
        print(f"   結果: {self.output_dir}/final_results.csv")
        print("=" * 60)
        
        return results


if __name__ == "__main__":
    pipeline = TwoStageFinalPipeline()
    pipeline.run()
