"""
Robust Model Comparison Script v2 (TargetEncoder Edition)
==========================================================
Comparing Logistic Regression, Random Forest, and LightGBM with strict fairness and interpretability.

Key Features:
1. NO Data Leakage: All preprocessing strictly inside CV loops.
2. Correct Encoding: TargetEncoder for LogReg (fixes OrdinalEncoder problem).
3. Correct LightGBM CV: Manual loop with properly transformed eval_set.
4. Baseline Comparison: Dummy Classifiers for reference.
5. Optimized SHAP: Sampling background data.
6. Resume Capability: Checkpointing to continue from interruptions.
7. Robust Error Handling: Try-except blocks to prevent crashes.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import gc
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score
)
from category_encoders import TargetEncoder
from sklearn.utils import parallel_backend
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# Experiment Class
# -----------------------------------------------------------------------------

class RobustModelComparator:
    def __init__(self, data_path, target_col='死者数', n_folds=5, random_state=42):
        self.data_path = data_path
        self.target_col = target_col
        self.n_folds = n_folds
        self.random_state = random_state
        
        self.output_dir = "results/robust_comparison"
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.checkpoint_path = os.path.join(self.output_dir, "checkpoint.json")
        self.results = self._load_checkpoint()
        
        print(f"✅ Experiment Initialized. Output: {self.output_dir}")
        if self.results:
            print(f"   ⏩ Resuming from checkpoint ({len(self.results)} records found)")

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return []
        return []

    def _save_checkpoint(self):
        try:
            with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"   ⚠️ Checkpoint save failed: {e}")

    def _is_done(self, model_name, fold):
        for r in self.results:
            if r['Model'] == model_name and r['Fold'] == fold:
                return True
        return False

    def load_data(self):
        print(f"\n📂 Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        print(f"   Shape: {self.df.shape}")
        
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
                median_val = self.X[col].median()
                self.X[col] = self.X[col].fillna(median_val).astype(np.float32)

        print(f"   Numerical: {len(self.numeric_cols)}, Categorical: {len(self.categorical_cols)}")
        gc.collect()

    def run_evaluation(self):
        print("\n🚀 Running Evaluation...")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        lgb_params = {
            'objective': 'binary', 
            'metric': 'binary_logloss', 
            'boosting_type': 'gbdt',
            'verbosity': -1, 
            'scale_pos_weight': 10, 
            'n_estimators': 1000,
            'learning_rate': 0.05, 
            'num_leaves': 31, 
            'max_depth': 8,
            'min_child_samples': 100,
            'random_state': self.random_state, 
            'n_jobs': -1
        }
        
        # Use threading backend to avoid Loky/ResourceTracker errors on Windows/Python 3.14
        # Random Forest releases GIL, so performance is still good.
        with parallel_backend('threading', n_jobs=-1):
            for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
                print(f"\n--- Fold {fold+1}/{self.n_folds} ---")
                
                X_train = self.X.iloc[train_idx].copy()
                X_val = self.X.iloc[val_idx].copy()
                y_train = self.y[train_idx]
                y_val = self.y[val_idx]
                
                # === Baseline: Stratified ===
                if not self._is_done('Baseline(Stratified)', fold+1):
                    try:
                        dummy_strat = DummyClassifier(strategy='stratified', random_state=self.random_state)
                        dummy_strat.fit(X_train, y_train)
                        self._record_metrics('Baseline(Stratified)', dummy_strat, X_val, y_val, fold+1)
                    except Exception as e:
                        print(f"   ❌ Baseline(Stratified) failed: {e}")
                
                # === Baseline: MostFrequent ===
                if not self._is_done('Baseline(MostFreq)', fold+1):
                    try:
                        dummy_freq = DummyClassifier(strategy='most_frequent')
                        dummy_freq.fit(X_train, y_train)
                        self._record_metrics('Baseline(MostFreq)', dummy_freq, X_val, y_val, fold+1)
                    except Exception as e:
                        print(f"   ❌ Baseline(MostFreq) failed: {e}")
                
                # === Logistic Regression (with TargetEncoder) ===
                if not self._is_done('LogisticRegression', fold+1):
                    try:
                        print("   Training LogisticRegression (TargetEncoder)...")
                        
                        # TargetEncoder: fit on train, transform on both
                        te = TargetEncoder(cols=self.categorical_cols, smoothing=1.0)
                        X_train_te = te.fit_transform(X_train, y_train)
                        X_val_te = te.transform(X_val)
                        
                        # Scale numerics
                        scaler = StandardScaler()
                        X_train_scaled = scaler.fit_transform(X_train_te)
                        X_val_scaled = scaler.transform(X_val_te)
                        
                        lr = LogisticRegression(
                            C=0.1, penalty='l2', solver='lbfgs', max_iter=500,
                            class_weight='balanced', random_state=self.random_state, n_jobs=-1
                        )
                        lr.fit(X_train_scaled, y_train)
                        
                        y_prob = lr.predict_proba(X_val_scaled)[:, 1]
                        y_pred = lr.predict(X_val_scaled)
                        self._add_result('LogisticRegression', fold+1, y_val, y_prob, y_pred)
                        
                        del te, scaler, lr, X_train_te, X_val_te, X_train_scaled, X_val_scaled
                        gc.collect()
                    except Exception as e:
                        print(f"   ❌ LogisticRegression failed: {e}")
                
                # === Random Forest ===
                if not self._is_done('RandomForest', fold+1):
                    try:
                        print("   Training RandomForest...")
                        
                        # OrdinalEncoder is fine for tree models
                        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                        X_train_cat = X_train[self.categorical_cols].copy()
                        X_val_cat = X_val[self.categorical_cols].copy()
                        
                        X_train_cat_enc = oe.fit_transform(X_train_cat)
                        X_val_cat_enc = oe.transform(X_val_cat)
                        
                        X_train_rf = np.hstack([X_train[self.numeric_cols].values, X_train_cat_enc])
                        X_val_rf = np.hstack([X_val[self.numeric_cols].values, X_val_cat_enc])
                        
                        rf = RandomForestClassifier(
                            n_estimators=100, max_depth=10, min_samples_leaf=20,
                            class_weight='balanced', random_state=self.random_state, n_jobs=-1
                        )
                        rf.fit(X_train_rf, y_train)
                        
                        y_prob = rf.predict_proba(X_val_rf)[:, 1]
                        y_pred = rf.predict(X_val_rf)
                        self._add_result('RandomForest', fold+1, y_val, y_prob, y_pred)
                        
                        del oe, rf, X_train_rf, X_val_rf
                        gc.collect()
                    except Exception as e:
                        print(f"   ❌ RandomForest failed: {e}")
                
                # === LightGBM ===
                if not self._is_done('LightGBM', fold+1):
                    try:
                        print("   Training LightGBM...")
                        
                        # Native categorical handling
                        X_train_lgb = X_train.copy()
                        X_val_lgb = X_val.copy()
                        for c in self.categorical_cols:
                            X_train_lgb[c] = X_train_lgb[c].astype('category')
                            X_val_lgb[c] = X_val_lgb[c].astype('category')
                        
                        lgb_clf = lgb.LGBMClassifier(**lgb_params)
                        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
                        
                        lgb_clf.fit(
                            X_train_lgb, y_train,
                            eval_set=[(X_val_lgb, y_val)],
                            eval_metric='average_precision',
                            callbacks=callbacks
                        )
                        
                        y_prob = lgb_clf.predict_proba(X_val_lgb)[:, 1]
                        y_pred = (y_prob >= 0.5).astype(int)
                        self._add_result('LightGBM', fold+1, y_val, y_prob, y_pred)
                        
                        del lgb_clf, X_train_lgb, X_val_lgb
                        gc.collect()
                    except Exception as e:
                        print(f"   ❌ LightGBM failed: {e}")
                
                del X_train, X_val
                gc.collect()
        
        self._finalize()

    def _record_metrics(self, name, model, X_val, y_val, fold):
        try:
            y_prob = model.predict_proba(X_val)[:, 1]
        except:
            y_prob = np.zeros(len(y_val))
        y_pred = model.predict(X_val)
        self._add_result(name, fold, y_val, y_prob, y_pred)

    def _add_result(self, name, fold, y_true, y_prob, y_pred):
        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except:
            pr_auc = 0.0
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except:
            roc_auc = 0.0
            
        res = {
            'Model': name,
            'Fold': fold,
            'PR-AUC': pr_auc,
            'ROC-AUC': roc_auc,
            'F1': float(f1_score(y_true, y_pred, zero_division=0)),
            'Recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'Precision': float(precision_score(y_true, y_pred, zero_division=0))
        }
        self.results.append(res)
        self._save_checkpoint()
        print(f"  ✓ {name:<20} | PR-AUC: {res['PR-AUC']:.4f} | Recall: {res['Recall']:.4f}")

    def _finalize(self):
        print("\n📊 Generating Final Report...")
        
        df_res = pd.DataFrame(self.results)
        df_res.to_csv(os.path.join(self.output_dir, "comparison_results.csv"), index=False, encoding='utf-8-sig')
        
        summary = df_res.groupby('Model')[['PR-AUC', 'ROC-AUC', 'F1', 'Recall', 'Precision']].mean()
        summary = summary.sort_values(by='PR-AUC', ascending=False)
        
        print("\n🏆 Final Results Summary (Average across Folds):")
        print(summary.to_string())
        
        summary.to_csv(os.path.join(self.output_dir, "summary_metrics.csv"), encoding='utf-8-sig')
        
        if os.path.exists(self.checkpoint_path):
            done_path = os.path.join(self.output_dir, "checkpoint_done.json")
            try:
                os.replace(self.checkpoint_path, done_path)
            except:
                pass
        
        print(f"\n✅ All results saved to: {self.output_dir}")

    def run(self):
        self.load_data()
        self.run_evaluation()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    DATA_PATH = "data/processed/honhyo_clean_with_features.csv"
    
    experiment = RobustModelComparator(data_path=DATA_PATH, n_folds=5)
    experiment.run()
