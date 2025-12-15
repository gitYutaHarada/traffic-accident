"""
Stage 2 Optuna再最適化（Recall 99%固定 Precision最大化）
======================================================
目的関数: Recall 99%を達成する閾値でのPrecision

PR-AUCではなく、ユーザーの真の目標に直接最適化する。
"""

import pandas as pd
import numpy as np
import os
import pickle
import gc
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score
import lightgbm as lgb
import optuna
import warnings

warnings.filterwarnings('ignore')


def find_threshold_for_recall(y_true, y_prob, target_recall):
    """指定Recallを達成する閾値を探索"""
    for thresh in np.arange(0.99, 0.001, -0.005):
        y_pred = (y_prob >= thresh).astype(int)
        recall = recall_score(y_true, y_pred)
        if recall >= target_recall:
            return thresh
    return 0.001


class Stage2ObjectiveRecallFixed:
    """Recall 99%固定でのPrecision最大化用Objective"""
    
    def __init__(self, X, y, target_recall=0.99, n_folds=5, random_state=42):
        self.X = X
        self.y = y
        self.target_recall = target_recall
        self.n_folds = n_folds
        self.random_state = random_state
        self.trial_count = 0
        self.best_score = 0.0
        self.start_time = datetime.now()
    
    def __call__(self, trial):
        self.trial_count += 1
        
        # 探索パラメータ
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': self.random_state,
            'n_estimators': 500,
            
            # 探索対象
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', 6, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 50.0),
        }
        
        # Cross-Validation
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        precisions_at_recall = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            X_train = self.X.iloc[train_idx]
            y_train = self.y[train_idx]
            X_val = self.X.iloc[val_idx]
            y_val = self.y[val_idx]
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            y_prob = model.predict_proba(X_val)[:, 1]
            
            # Recall 99%を達成する閾値を探索
            thresh = find_threshold_for_recall(y_val, y_prob, self.target_recall)
            y_pred = (y_prob >= thresh).astype(int)
            precision = precision_score(y_val, y_pred) if y_pred.sum() > 0 else 0
            
            precisions_at_recall.append(precision)
            
            del model
            gc.collect()
        
        mean_precision = np.mean(precisions_at_recall)
        
        # 進捗表示
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if mean_precision > self.best_score:
            self.best_score = mean_precision
            print(f"   🏆 Trial {self.trial_count}: Prec@Rec99%={mean_precision:.4f} (NEW BEST!) [{elapsed/60:.1f}min]")
        else:
            print(f"   Trial {self.trial_count}: Prec@Rec99%={mean_precision:.4f} [{elapsed/60:.1f}min]")
        
        return mean_precision


def run_optuna_recall_fixed(
    data_path: str = "results/two_stage_model/optuna_data/stage2_train_data.pkl",
    target_recall: float = 0.99,
    n_trials: int = 50,
    n_folds: int = 5,
    random_state: int = 42,
    output_dir: str = "results/two_stage_model/optuna_recall_fixed"
):
    """Recall固定Optuna最適化を実行"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print(f"Stage 2 Optuna最適化（Recall {int(target_recall*100)}%固定 Precision最大化）")
    print(f"試行回数: {n_trials}")
    print("=" * 70)
    
    # データ読み込み
    print("\n📂 Stage 2用データ読み込み...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_s2 = data['X_s2']
    y_s2 = data['y_s2']
    
    n_pos = y_s2.sum()
    n_neg = len(y_s2) - n_pos
    print(f"   データ数: {len(y_s2):,} (Pos: {n_pos:,}, Neg: {n_neg:,})")
    
    # Optuna Study作成
    print("\n🔍 最適化開始...")
    print("-" * 70)
    
    study = optuna.create_study(
        direction='maximize',
        study_name=f'stage2_precision_at_recall{int(target_recall*100)}',
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    
    objective = Stage2ObjectiveRecallFixed(
        X_s2, y_s2, 
        target_recall=target_recall,
        n_folds=n_folds, 
        random_state=random_state
    )
    
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True
    )
    
    # 結果表示
    print("\n" + "=" * 70)
    print("✅ 最適化完了！")
    print("=" * 70)
    
    best = study.best_trial
    print(f"\n🏆 ベストスコア: Precision@Recall{int(target_recall*100)}% = {best.value:.4f}")
    print(f"\n📋 ベストパラメータ:")
    for key, value in best.params.items():
        print(f"   {key}: {value}")
    
    # 結果保存
    results_df = study.trials_dataframe()
    results_df.to_csv(os.path.join(output_dir, "optuna_trials.csv"), index=False)
    
    best_params_df = pd.DataFrame([best.params])
    best_params_df['precision_at_recall99'] = best.value
    best_params_df.to_csv(os.path.join(output_dir, "best_params.csv"), index=False)
    
    print(f"\n💾 結果保存:")
    print(f"   - {output_dir}/optuna_trials.csv")
    print(f"   - {output_dir}/best_params.csv")
    
    return study


if __name__ == "__main__":
    import sys
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    
    run_optuna_recall_fixed(n_trials=n_trials)
