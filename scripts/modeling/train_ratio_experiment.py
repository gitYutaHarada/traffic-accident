"""
アンダーサンプリング比率探索実験
================================
比率: 1:9, 1:8, 1:7, 1:6, 1:5, 1:4, 1:3, 1:2, 1:1 (9パターン)
Recall目標: 99%固定

目的: より過激なサンプリングでRecallを高める閾値を発見する
"""

import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


class RatioExperimentPipeline:
    """サンプリング比率探索実験"""
    
    def __init__(
        self,
        data_path: str = "data/processed/honhyo_clean_with_features.csv",
        target_col: str = "死者数",
        n_folds: int = 5,
        random_state: int = 42,
        stage1_recall_target: float = 0.99,
        undersample_ratio: float = 10.0,
        n_seeds: int = 3,
        output_suffix: str = ""
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.n_folds = n_folds
        self.random_state = random_state
        self.stage1_recall_target = stage1_recall_target
        self.undersample_ratio = undersample_ratio
        self.n_seeds = n_seeds
        
        suffix = output_suffix if output_suffix else f"ratio_{int(undersample_ratio)}"
        self.output_dir = f"results/two_stage_model/ratio_exp/{suffix}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"\n{'='*50}")
        print(f"比率 1:{int(self.undersample_ratio)}, Recall目標 {self.stage1_recall_target:.0%}")
        print(f"{'='*50}")
    
    def load_data(self):
        """データ読み込み"""
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
        
        for col in self.X.columns:
            if col in known_categoricals or self.X[col].dtype == 'object':
                self.X[col] = self.X[col].astype('category')
            else:
                self.X[col] = self.X[col].astype(np.float32)
        
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
    
    def train_stage1_oof(self):
        """Stage 1 OOF学習"""
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.oof_proba = np.zeros(len(self.y))
        
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
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            X_train_full = self.X.iloc[train_idx]
            y_train_full = self.y[train_idx]
            X_val = self.X.iloc[val_idx]
            y_val = self.y[val_idx]
            
            fold_proba = np.zeros(len(val_idx))
            
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
                del model
                gc.collect()
            
            self.oof_proba[val_idx] = fold_proba
        
        # OOF精度（閾値0.5）
        oof_pred = (self.oof_proba >= 0.5).astype(int)
        self.oof_precision = precision_score(self.y, oof_pred)
        self.oof_recall = recall_score(self.y, oof_pred)
        self.oof_auc = roc_auc_score(self.y, self.oof_proba)
        
        print(f"   OOF: Prec={self.oof_precision:.4f}, Rec={self.oof_recall:.4f}, AUC={self.oof_auc:.4f}")
    
    def find_recall_threshold(self):
        """Recall目標閾値探索"""
        for thresh in np.arange(0.50, 0.001, -0.005):
            y_pred = (self.oof_proba >= thresh).astype(int)
            recall = recall_score(self.y, y_pred)
            if recall >= self.stage1_recall_target:
                self.threshold = thresh
                break
        else:
            self.threshold = 0.001
        
        y_pred_final = (self.oof_proba >= self.threshold).astype(int)
        self.stage1_recall = recall_score(self.y, y_pred_final)
        self.stage1_precision = precision_score(self.y, y_pred_final)
        n_candidates = y_pred_final.sum()
        self.filter_rate = 1 - (n_candidates / len(self.y))
        
        print(f"   閾値: {self.threshold:.4f}, Recall: {self.stage1_recall:.4f}, フィルタ率: {self.filter_rate*100:.1f}%")
    
    def run(self):
        start = datetime.now()
        self.load_data()
        self.train_stage1_oof()
        self.find_recall_threshold()
        elapsed = (datetime.now() - start).total_seconds()
        
        return {
            'ratio': self.undersample_ratio,
            'recall_target': self.stage1_recall_target,
            'threshold': self.threshold,
            'stage1_recall': self.stage1_recall,
            'stage1_precision': self.stage1_precision,
            'filter_rate': self.filter_rate,
            'oof_auc': self.oof_auc,
            'elapsed_sec': elapsed
        }


def run_ratio_experiments():
    """比率探索実験実行"""
    print("=" * 70)
    print("アンダーサンプリング比率探索実験")
    print("比率: 1:9, 1:8, 1:7, 1:6, 1:5, 1:4, 1:3, 1:2, 1:1")
    print("Recall目標: 99%")
    print("=" * 70)
    
    ratios = [9, 8, 7, 6, 5, 4, 3, 2, 1]
    all_results = []
    
    for ratio in ratios:
        pipeline = RatioExperimentPipeline(undersample_ratio=ratio)
        result = pipeline.run()
        all_results.append(result)
        del pipeline
        gc.collect()
    
    df_results = pd.DataFrame(all_results)
    os.makedirs("results/two_stage_model/ratio_exp", exist_ok=True)
    df_results.to_csv("results/two_stage_model/ratio_exp/ratio_comparison.csv", index=False)
    
    print("\n" + "=" * 70)
    print("📊 比率探索実験結果")
    print("=" * 70)
    print(df_results[['ratio', 'threshold', 'stage1_recall', 'stage1_precision', 'filter_rate', 'oof_auc']].to_string(index=False))
    print("\n✅ 完了！ 結果: results/two_stage_model/ratio_exp/ratio_comparison.csv")
    
    return df_results


if __name__ == "__main__":
    run_ratio_experiments()
