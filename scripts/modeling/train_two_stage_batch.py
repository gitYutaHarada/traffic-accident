"""
2段階モデル バッチ実験スクリプト
================================
Recall目標を98%〜95%まで変化させた4回の実験を自動実行し、結果を比較する。

推定実行時間: 約8分
"""

import pandas as pd
import numpy as np
import os
import gc
import joblib
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
import lightgbm as lgb
import warnings
import argparse

warnings.filterwarnings('ignore')


class TwoStagePipeline:
    """2段階モデルのパイプライン"""
    
    def __init__(
        self,
        data_path: str = "data/processed/honhyo_clean_with_features.csv",
        target_col: str = "死者数",
        n_folds: int = 5,
        random_state: int = 42,
        stage1_recall_target: float = 0.98,
        top_k_features: int = 10,
        output_suffix: str = ""
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.n_folds = n_folds
        self.random_state = random_state
        self.stage1_recall_target = stage1_recall_target
        self.top_k_features = top_k_features
        
        suffix = output_suffix if output_suffix else f"recall_{int(stage1_recall_target*100)}"
        self.output_dir = f"results/two_stage_model/{suffix}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.stage1_models = []
        self.stage2_model = None
        self.threshold_stage1 = None
        self.top_features = None
        self.interaction_names = None
        
        print(f"\n{'='*60}")
        print(f"実験: Recall目標 = {self.stage1_recall_target:.0%}")
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
                self.X[col] = self.X[col].astype(str).fillna('Missing')
            else:
                self.numeric_cols.append(col)
                self.X[col] = self.X[col].fillna(self.X[col].median()).astype(np.float32)
        
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.ordinal_encoder.fit(self.X[self.categorical_cols])
        self.feature_names = self.numeric_cols + self.categorical_cols
        
        X_cat_enc = self.ordinal_encoder.transform(self.X[self.categorical_cols])
        self.X_encoded = np.hstack([self.X[self.numeric_cols].values, X_cat_enc])
        gc.collect()
    
    def train_stage1_oof(self):
        """Stage 1: RF OOF学習"""
        print(f"🌲 Stage 1: RF OOF学習...")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.oof_proba_stage1 = np.zeros(len(self.y))
        feature_importances = np.zeros(len(self.feature_names))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_encoded, self.y)):
            X_train, X_val = self.X_encoded[train_idx], self.X_encoded[val_idx]
            y_train = self.y[train_idx]
            
            model = RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_leaf=20,
                class_weight='balanced', random_state=self.random_state, n_jobs=-1
            )
            model.fit(X_train, y_train)
            self.oof_proba_stage1[val_idx] = model.predict_proba(X_val)[:, 1]
            feature_importances += model.feature_importances_ / self.n_folds
            self.stage1_models.append(model)
            del model
            gc.collect()
        
        self.feature_importance_df = pd.DataFrame({
            'feature': self.feature_names, 'importance': feature_importances
        }).sort_values('importance', ascending=False)
        self.top_features = self.feature_importance_df.head(self.top_k_features)['feature'].tolist()
    
    def find_recall_threshold(self):
        """Recall目標閾値探索"""
        for thresh in np.arange(0.01, 0.5, 0.005):
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
        print(f"   フィルタリング率: {self.filter_rate*100:.2f}% 除外")
        
        self.stage2_mask = self.oof_proba_stage1 >= self.threshold_stage1
    
    def generate_interaction_features(self, X_subset):
        """相互作用特徴量生成"""
        top_indices = [self.feature_names.index(f) for f in self.top_features if f in self.feature_names]
        
        interactions = []
        self.interaction_names = []
        for i, idx1 in enumerate(top_indices):
            for idx2 in top_indices[i+1:]:
                interactions.append(X_subset[:, idx1] * X_subset[:, idx2])
                self.interaction_names.append(f"{self.feature_names[idx1]}*{self.feature_names[idx2]}")
        
        if interactions:
            return np.hstack([X_subset, np.column_stack(interactions)])
        return X_subset
    
    def train_stage2(self):
        """Stage 2: LightGBM学習"""
        print(f"🌿 Stage 2: LightGBM 学習...")
        
        X_s2 = self.generate_interaction_features(self.X_encoded[self.stage2_mask])
        y_s2 = self.y[self.stage2_mask]
        
        n_pos, n_neg = y_s2.sum(), len(y_s2) - y_s2.sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        self.stage2_model = lgb.LGBMClassifier(
            objective='binary', metric='binary_logloss', boosting_type='gbdt',
            verbosity=-1, n_estimators=500, learning_rate=0.05, num_leaves=31,
            max_depth=8, scale_pos_weight=scale_pos_weight,
            random_state=self.random_state, n_jobs=-1
        )
        self.stage2_model.fit(X_s2, y_s2)
    
    def evaluate(self):
        """評価"""
        X_s2 = self.generate_interaction_features(self.X_encoded[self.stage2_mask])
        y_prob = self.stage2_model.predict_proba(X_s2)[:, 1]
        
        final_proba = np.zeros(len(self.y))
        final_proba[self.stage2_mask] = y_prob
        
        y_pred = (final_proba >= 0.5).astype(int)
        
        self.final_precision = precision_score(self.y, y_pred) if y_pred.sum() > 0 else 0
        self.final_recall = recall_score(self.y, y_pred)
        self.final_f1 = f1_score(self.y, y_pred)
        
        # Baseline
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
    print("2段階モデル バッチ実験")
    print("Recall目標: 98%, 97%, 96%, 95%")
    print("=" * 70)
    
    recall_targets = [0.98, 0.97, 0.96, 0.95]
    all_results = []
    
    for target in recall_targets:
        pipeline = TwoStagePipeline(stage1_recall_target=target)
        result = pipeline.run()
        all_results.append(result)
        del pipeline
        gc.collect()
    
    # 結果比較表
    df_results = pd.DataFrame(all_results)
    df_results.to_csv("results/two_stage_model/batch_comparison.csv", index=False)
    
    print("\n" + "=" * 70)
    print("📊 バッチ実験結果比較")
    print("=" * 70)
    print(df_results[['recall_target', 'stage1_threshold', 'filter_rate', 
                       'final_precision', 'final_recall', 'precision_improvement_pct']].to_string(index=False))
    print("\n✅ 完了！ 結果: results/two_stage_model/batch_comparison.csv")
    
    return df_results


if __name__ == "__main__":
    run_batch_experiments()
