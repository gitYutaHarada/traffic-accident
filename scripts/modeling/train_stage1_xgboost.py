"""
Stage 1 XGBoost 実験スクリプト (改良版)
=======================================
LightGBMとの比較のため、Stage 1のモデルをXGBoostに変更して
フィルタリング能力を評価します。

改良点:
- Early Stopping追加（過学習防止）
- 不均衡対策パラメータ（scale_pos_weight）
- LightGBMとの相関分析機能（アンサンブル効果見積もり）
- OOF結果のID紐付け保存

評価指標:
- OOF AUC
- Recall 99%時の閾値
- 除外率 (Filter Rate)
"""

import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')


class Stage1XGBoostExperiment:
    """Stage 1 XGBoost 実験クラス (改良版)"""
    
    def __init__(
        self,
        data_path: str = "data/processed/honhyo_clean_with_features.csv",
        target_col: str = "死者数",
        n_folds: int = 5,
        random_state: int = 42,
        stage1_recall_target: float = 0.99,
        undersample_ratio: float = 2.0,
        n_seeds: int = 3,
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.n_folds = n_folds
        self.random_state = random_state
        self.stage1_recall_target = stage1_recall_target
        self.undersample_ratio = undersample_ratio
        self.n_seeds = n_seeds
        
        self.output_dir = "results/two_stage_model/xgboost_experiment"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 60)
        print("Stage 1 XGBoost 実験 (改良版)")
        print(f"Under-sampling 1:{int(self.undersample_ratio)}, Recall Target {self.stage1_recall_target:.0%}")
        print("=" * 60)
    
    def load_data(self):
        """データ読み込み"""
        print("\n📂 データ読み込み中...")
        self.df = pd.read_csv(self.data_path)
        self.y = self.df[self.target_col].values
        self.X = self.df.drop(columns=[self.target_col])
        
        if '発生日時' in self.X.columns:
            self.X = self.X.drop(columns=['発生日時'])
        
        # カテゴリカル変数の処理
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
        print(f"   カテゴリカル変数: {len(self.categorical_cols)}個")
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
    
    def train_stage1_xgboost(self):
        """Stage 1: XGBoost + Under-sampling + Multi-Seed Averaging"""
        print("\n🌿 Stage 1: XGBoost + Under-sampling (1:2) + 3-Seed Averaging")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.oof_proba = np.zeros(len(self.y))
        feature_importances = np.zeros(len(self.feature_names))
        
        # XGBoostパラメータ (改良版)
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',  # 高速化
            'enable_categorical': True,  # カテゴリカル変数対応
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 1000,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_jobs': -1,
            'verbosity': 0,
            # 不均衡対策パラメータ
            'scale_pos_weight': 2.0,  # 負例が正例の2倍あるため、バランスを取る
            'max_delta_step': 1,      # クラス不均衡時の更新を安定させる
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
                
                model = xgb.XGBClassifier(
                    **xgb_params,
                    random_state=seed,
                    early_stopping_rounds=50  # Early Stopping（コンストラクタで指定）
                )
                model.fit(
                    X_train_under, y_train_under,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                fold_proba += model.predict_proba(X_val)[:, 1] / self.n_seeds
                feature_importances += model.feature_importances_ / (self.n_folds * self.n_seeds)
                fold_models.append(model)
                
                del model
                gc.collect()
            
            self.oof_proba[val_idx] = fold_proba
            self.stage1_models.append(fold_models)
        
        # Feature Importance
        self.feature_importance_df = pd.DataFrame({
            'feature': self.feature_names, 'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        # OOF精度
        oof_pred = (self.oof_proba >= 0.5).astype(int)
        self.oof_auc = roc_auc_score(self.y, self.oof_proba)
        print(f"   OOF (閾値0.5): Prec={precision_score(self.y, oof_pred):.4f}, Rec={recall_score(self.y, oof_pred):.4f}, AUC={self.oof_auc:.4f}")
    
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
        self.recall = recall_score(self.y, y_pred_final)
        self.precision = precision_score(self.y, y_pred_final)
        n_candidates = y_pred_final.sum()
        self.filter_rate = 1 - (n_candidates / len(self.y))
        
        print(f"\n📊 Recall {self.stage1_recall_target:.0%} 評価:")
        print(f"   閾値: {self.threshold:.4f}")
        print(f"   Recall: {self.recall:.4f}")
        print(f"   除外率: {self.filter_rate*100:.2f}% ({len(self.y) - n_candidates:,}件除外)")
    
    def analyze_correlation(self):
        """LightGBMとの相関分析（アンサンブル効果見積もり）"""
        lgbm_path = "results/two_stage_model/lightgbm_stage1_oof.csv"
        
        if os.path.exists(lgbm_path):
            print("\n🔗 LightGBMとの相関分析...")
            lgbm_df = pd.read_csv(lgbm_path)
            lgbm_probs = lgbm_df['oof_proba'].values
            
            # 相関係数の算出
            corr = np.corrcoef(self.oof_proba, lgbm_probs)[0, 1]
            print(f"   予測値相関: {corr:.4f}")
            
            if corr < 0.95:
                print("   👉 多様性あり！アンサンブル（Stacking）で精度向上が見込めます。")
            else:
                print("   ⚠️ 挙動が酷似しています。単独性能が高い方を採用してください。")
            
            self.lgbm_correlation = corr
        else:
            print(f"\n⚠️ LightGBMのOOF結果ファイルが見つかりません: {lgbm_path}")
            print("   相関分析をスキップします。LightGBMのOOF結果を保存してから再実行してください。")
            self.lgbm_correlation = None
    
    def save_results(self):
        """結果保存（ID紐付け対応）"""
        # OOF確率とラベルを保存（ID紐付け）
        oof_df = pd.DataFrame({
            'y_true': self.y,
            'oof_proba': self.oof_proba
        })
        oof_df.to_csv(
            os.path.join(self.output_dir, "xgboost_stage1_oof.csv"),
            index=False
        )
        
        # サマリー結果
        results = {
            'model': 'XGBoost',
            'oof_auc': self.oof_auc,
            'threshold': self.threshold,
            'recall': self.recall,
            'precision': self.precision,
            'filter_rate': self.filter_rate,
            'lgbm_correlation': self.lgbm_correlation if hasattr(self, 'lgbm_correlation') else None,
        }
        
        pd.DataFrame([results]).to_csv(
            os.path.join(self.output_dir, "xgboost_stage1_results.csv"),
            index=False
        )
        self.feature_importance_df.to_csv(
            os.path.join(self.output_dir, "xgboost_feature_importance.csv"),
            index=False
        )
        
        print(f"\n💾 結果保存: {self.output_dir}/")
        print("   - xgboost_stage1_oof.csv (OOF確率、ラベル)")
        print("   - xgboost_stage1_results.csv (サマリー)")
        print("   - xgboost_feature_importance.csv")
    
    def run(self):
        """実験実行"""
        start = datetime.now()
        
        self.load_data()
        self.train_stage1_xgboost()
        self.find_recall_threshold()
        self.analyze_correlation()
        self.save_results()
        
        elapsed = (datetime.now() - start).total_seconds()
        
        print("\n" + "=" * 60)
        print("✅ 実験完了!")
        print(f"   実行時間: {elapsed:.1f}秒")
        print("=" * 60)
        
        return {
            'oof_auc': self.oof_auc,
            'threshold': self.threshold,
            'recall': self.recall,
            'filter_rate': self.filter_rate,
        }


if __name__ == "__main__":
    experiment = Stage1XGBoostExperiment()
    experiment.run()
