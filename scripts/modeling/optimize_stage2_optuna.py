"""
Stage 2 Optuna ハイパーパラメータ最適化
======================================
Implementation Plan v22

- 評価指標: PR-AUC (カスタム関数)
- Pruning: 見込みのない試行を早期打ち切り
- 探索: num_leaves, reg, scale_pos_weight 等
- 進捗表示: tqdmとOptuna標準ログで確認可能
"""

import pandas as pd
import numpy as np
import os
import pickle
import gc
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_score, recall_score
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# カスタム評価関数（PR-AUC）
# ============================================================
def pr_auc_metric(preds, train_data):
    """LightGBM用カスタムPR-AUC評価関数"""
    y_true = train_data.get_label()
    score = average_precision_score(y_true, preds)
    return 'pr_auc', score, True  # higher_is_better=True


# ============================================================
# Optuna Objective関数
# ============================================================
class Stage2Objective:
    """Stage 2のハイパーパラメータ最適化用Objective"""
    
    def __init__(self, X, y, n_folds=5, random_state=42):
        self.X = X
        self.y = y
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
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': self.random_state,
            
            # 探索対象
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', 8, 15),
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
        pr_auc_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            X_train = self.X.iloc[train_idx]
            y_train = self.y[train_idx]
            X_val = self.X.iloc[val_idx]
            y_val = self.y[val_idx]
            
            # LightGBM Dataset
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            
            # Pruning Callback
            pruning_callback = LightGBMPruningCallback(trial, 'pr_auc')
            
            try:
                model = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=500,
                    valid_sets=[dval],
                    feval=pr_auc_metric,
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),
                        pruning_callback
                    ]
                )
                
                y_prob = model.predict(X_val)
                pr_auc = average_precision_score(y_val, y_prob)
                pr_auc_scores.append(pr_auc)
                
            except optuna.TrialPruned:
                raise
            
            del model
            gc.collect()
        
        mean_pr_auc = np.mean(pr_auc_scores)
        
        # 進捗表示
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if mean_pr_auc > self.best_score:
            self.best_score = mean_pr_auc
            print(f"   🏆 Trial {self.trial_count}: PR-AUC={mean_pr_auc:.4f} (NEW BEST!) [{elapsed/60:.1f}min]")
        else:
            print(f"   Trial {self.trial_count}: PR-AUC={mean_pr_auc:.4f} [{elapsed/60:.1f}min]")
        
        return mean_pr_auc


# ============================================================
# メイン
# ============================================================
def run_optuna_optimization(
    data_path: str = "results/two_stage_model/optuna_data/stage2_train_data.pkl",
    n_trials: int = 50,
    n_folds: int = 5,
    random_state: int = 42,
    output_dir: str = "results/two_stage_model/optuna_results"
):
    """Optuna最適化を実行"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Stage 2 Optuna ハイパーパラメータ最適化")
    print(f"評価指標: PR-AUC (Precision-Recall AUC)")
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
    print(f"   データ数: {len(y_s2):,} (Pos: {n_pos:,}, Neg: {n_neg:,}, 比率 1:{n_neg//n_pos})")
    
    # Optuna Study作成
    print("\n🔍 最適化開始...")
    print("-" * 70)
    
    study = optuna.create_study(
        direction='maximize',
        study_name='stage2_pr_auc_optimization',
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    
    objective = Stage2Objective(X_s2, y_s2, n_folds=n_folds, random_state=random_state)
    
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
    print(f"\n🏆 ベストスコア: PR-AUC = {best.value:.4f}")
    print(f"\n📋 ベストパラメータ:")
    for key, value in best.params.items():
        print(f"   {key}: {value}")
    
    # 結果保存
    results_df = study.trials_dataframe()
    results_df.to_csv(os.path.join(output_dir, "optuna_trials.csv"), index=False)
    
    best_params_df = pd.DataFrame([best.params])
    best_params_df['pr_auc'] = best.value
    best_params_df.to_csv(os.path.join(output_dir, "best_params.csv"), index=False)
    
    print(f"\n💾 結果保存:")
    print(f"   - {output_dir}/optuna_trials.csv")
    print(f"   - {output_dir}/best_params.csv")
    
    return study


if __name__ == "__main__":
    # 引数でn_trialsを調整可能
    import sys
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    
    run_optuna_optimization(n_trials=n_trials)
