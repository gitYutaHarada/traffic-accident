"""
2段階モデル LightGBM + Under-sampling 版
========================================
Implementation Plan v11

変更点:
1. Stage 1学習時に1:10アンダーサンプリング
2. Validationデータは元の不均衡分布（1:118）
3. Early Stopping指標: AUC（不均衡対応）
4. シード平均化（3回）で安定化
5. 閾値は動的決定（確率インフレ対応）
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


class TwoStageLGBMUnderPipeline:
    """2段階モデル（LightGBM + Under-sampling版）"""
    
    def __init__(
        self,
        data_path: str = "data/processed/honhyo_clean_with_features.csv",
        target_col: str = "死者数",
        n_folds: int = 5,
        random_state: int = 42,
        stage1_recall_target: float = 0.95,
        undersample_ratio: float = 10.0,  # 負例 = 正例 × この値
        n_seeds: int = 3,  # シード平均化の回数
        top_k_features: int = 10,
        output_suffix: str = ""
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.n_folds = n_folds
        self.random_state = random_state
        self.stage1_recall_target = stage1_recall_target
        self.undersample_ratio = undersample_ratio
        self.n_seeds = n_seeds
        self.top_k_features = top_k_features
        
        suffix = output_suffix if output_suffix else f"under_recall_{int(stage1_recall_target*100)}"
        self.output_dir = f"results/two_stage_model/{suffix}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.stage1_models = []
        self.stage2_model = None
        self.threshold_stage1 = None
        self.top_features = None
        self.interaction_names = []
        
        print(f"\n{'='*60}")
        print(f"LightGBM + Under-sampling: Recall目標 = {self.stage1_recall_target:.0%}")
        print(f"Under-sampling比率 = 1:{int(self.undersample_ratio)}")
        print(f"{'='*60}")
    
    def load_data(self):
        """データ読み込み"""
        print("📂 データ読み込み中...")
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
        self.numeric_cols = []
        
        for col in self.X.columns:
            if col in known_categoricals or self.X[col].dtype == 'object':
                self.categorical_cols.append(col)
                self.X[col] = self.X[col].astype('category')
            else:
                self.numeric_cols.append(col)
                self.X[col] = self.X[col].astype(np.float32)
        
        self.feature_names = list(self.X.columns)
        print(f"   正例: {self.y.sum():,} / {len(self.y):,}")
        gc.collect()
    
    def undersample(self, X, y, seed):
        """負例を間引いてアンダーサンプリング"""
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        
        n_neg_sample = int(len(pos_idx) * self.undersample_ratio)
        np.random.seed(seed)
        sampled_neg_idx = np.random.choice(neg_idx, size=min(n_neg_sample, len(neg_idx)), replace=False)
        
        sampled_idx = np.concatenate([pos_idx, sampled_neg_idx])
        np.random.shuffle(sampled_idx)
        
        return X.iloc[sampled_idx], y[sampled_idx]
    
    def train_stage1_oof(self):
        """Stage 1: LightGBM OOF学習（アンダーサンプリング + シード平均化）"""
        print(f"🌿 Stage 1: LightGBM OOF学習 (Under-sampling + {self.n_seeds}シード平均化)...")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.oof_proba_stage1 = np.zeros(len(self.y))
        feature_importances = np.zeros(len(self.feature_names))
        
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',  # 重要: 不均衡データにはAUC
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
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            
            X_train_full = self.X.iloc[train_idx]
            y_train_full = self.y[train_idx]
            X_val = self.X.iloc[val_idx]  # 元の不均衡データ
            y_val = self.y[val_idx]
            
            fold_proba = np.zeros(len(val_idx))
            
            for seed_offset in range(self.n_seeds):
                seed = self.random_state + fold * 100 + seed_offset
                
                # アンダーサンプリング（学習データのみ）
                X_train_under, y_train_under = self.undersample(X_train_full, y_train_full, seed)
                
                model = lgb.LGBMClassifier(**lgb_params, random_state=seed)
                model.fit(
                    X_train_under, y_train_under,
                    eval_set=[(X_val, y_val)],  # Valは元の不均衡データ
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                
                fold_proba += model.predict_proba(X_val)[:, 1] / self.n_seeds
                feature_importances += model.feature_importances_ / (self.n_folds * self.n_seeds)
                
                del model
                gc.collect()
            
            self.oof_proba_stage1[val_idx] = fold_proba
        
        self.feature_importance_df = pd.DataFrame({
            'feature': self.feature_names, 'importance': feature_importances
        }).sort_values('importance', ascending=False)
        self.top_features = self.feature_importance_df.head(self.top_k_features)['feature'].tolist()
        
        # OOF精度（閾値0.5）
        oof_pred = (self.oof_proba_stage1 >= 0.5).astype(int)
        print(f"   OOF (閾値0.5): Prec={precision_score(self.y, oof_pred):.4f}, Rec={recall_score(self.y, oof_pred):.4f}, AUC={roc_auc_score(self.y, self.oof_proba_stage1):.4f}")
    
    def find_recall_threshold(self):
        """Recall目標閾値探索"""
        # アンダーサンプリングにより確率がインフレしているため、より高い閾値で探索
        for thresh in np.arange(0.50, 0.01, -0.01):  # 高い方から探索
            y_pred = (self.oof_proba_stage1 >= thresh).astype(int)
            recall = recall_score(self.y, y_pred)
            if recall >= self.stage1_recall_target:
                self.threshold_stage1 = thresh
                break
        else:
            self.threshold_stage1 = 0.01
        
        y_pred_final = (self.oof_proba_stage1 >= self.threshold_stage1).astype(int)
        self.stage1_recall = recall_score(self.y, y_pred_final)
        self.stage1_precision = precision_score(self.y, y_pred_final)
        n_candidates = y_pred_final.sum()
        self.filter_rate = 1 - (n_candidates / len(self.y))
        
        print(f"   閾値: {self.threshold_stage1:.3f}, Recall: {self.stage1_recall:.4f}")
        print(f"   フィルタリング率: {self.filter_rate*100:.2f}% 除外, 候補: {n_candidates:,}")
        
        self.stage2_mask = self.oof_proba_stage1 >= self.threshold_stage1
    
    def generate_interaction_features(self, X_subset):
        """相互作用特徴量生成"""
        X_out = X_subset.copy()
        self.interaction_names = []
        
        top_feature_list = [f for f in self.top_features if f in X_subset.columns]
        
        for i, f1 in enumerate(top_feature_list):
            for f2 in top_feature_list[i+1:]:
                if f1 in self.numeric_cols and f2 in self.numeric_cols:
                    name = f"{f1}*{f2}"
                    X_out[name] = X_subset[f1] * X_subset[f2]
                    self.interaction_names.append(name)
        
        return X_out
    
    def train_stage2(self):
        """Stage 2: LightGBM学習"""
        print(f"🌿 Stage 2: LightGBM 学習...")
        
        X_s2 = self.generate_interaction_features(self.X[self.stage2_mask].copy())
        y_s2 = self.y[self.stage2_mask]
        
        n_pos, n_neg = y_s2.sum(), len(y_s2) - y_s2.sum()
        scale_pos = n_neg / n_pos if n_pos > 0 else 1.0
        
        print(f"   Stage 2 データ: {len(y_s2):,} (Pos: {n_pos:,}, Neg: {n_neg:,})")
        
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'num_leaves': 31,
            'max_depth': 8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'scale_pos_weight': scale_pos,
            'n_estimators': 500,
            'learning_rate': 0.05,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        self.stage2_model = lgb.LGBMClassifier(**lgb_params)
        self.stage2_model.fit(X_s2, y_s2)
    
    def evaluate(self):
        """評価"""
        X_s2 = self.generate_interaction_features(self.X[self.stage2_mask].copy())
        y_prob = self.stage2_model.predict_proba(X_s2)[:, 1]
        
        final_proba = np.zeros(len(self.y))
        final_proba[self.stage2_mask] = y_prob
        
        y_pred = (final_proba >= 0.5).astype(int)
        
        self.final_precision = precision_score(self.y, y_pred) if y_pred.sum() > 0 else 0
        self.final_recall = recall_score(self.y, y_pred)
        self.final_f1 = f1_score(self.y, y_pred)
        
        # Baseline (Stage 1 単独 閾値0.5)
        y_pred_bl = (self.oof_proba_stage1 >= 0.5).astype(int)
        self.baseline_precision = precision_score(self.y, y_pred_bl)
        self.baseline_recall = recall_score(self.y, y_pred_bl)
        
        print(f"\n📈 結果: Prec={self.final_precision:.4f}, Rec={self.final_recall:.4f}")
        
        return {
            'recall_target': self.stage1_recall_target,
            'stage1_threshold': self.threshold_stage1,
            'stage1_recall': self.stage1_recall,
            'filter_rate': self.filter_rate,
            'final_precision': self.final_precision,
            'final_recall': self.final_recall,
            'final_f1': self.final_f1,
            'baseline_precision': self.baseline_precision,
            'baseline_recall': self.baseline_recall,
            'precision_improvement_pct': (self.final_precision - self.baseline_precision) / self.baseline_precision * 100 if self.baseline_precision > 0 else 0
        }
    
    def run(self):
        start = datetime.now()
        self.load_data()
        self.train_stage1_oof()
        self.find_recall_threshold()
        self.train_stage2()
        results = self.evaluate()
        results['elapsed_sec'] = (datetime.now() - start).total_seconds()
        pd.DataFrame([results]).to_csv(os.path.join(self.output_dir, "results.csv"), index=False)
        return results


def run_batch_experiments():
    """バッチ実験実行"""
    print("=" * 70)
    print("LightGBM + Under-sampling バッチ実験")
    print("Recall目標: 99%, 98%, 97%, 96%, 95%")
    print("=" * 70)
    
    recall_targets = [0.99, 0.98, 0.97, 0.96, 0.95]
    all_results = []
    
    for target in recall_targets:
        pipeline = TwoStageLGBMUnderPipeline(stage1_recall_target=target)
        result = pipeline.run()
        all_results.append(result)
        del pipeline
        gc.collect()
    
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("results/two_stage_model/under_batch_comparison.csv", index=False)
    
    print("\n" + "=" * 70)
    print("📊 Under-sampling バッチ実験結果比較")
    print("=" * 70)
    print(df_results[['recall_target', 'stage1_threshold', 'filter_rate', 
                       'final_precision', 'final_recall', 'precision_improvement_pct']].to_string(index=False))
    print("\n✅ 完了！ 結果: results/two_stage_model/under_batch_comparison.csv")
    
    return df_results


if __name__ == "__main__":
    run_batch_experiments()
