"""
Stage 2 Optuna ハイパーパラメータ最適化 (Focal Loss対応)
=========================================================
Implementation Plan v23 - Focal Loss + Recall 99% Precision最大化

- Focal Loss: Alpha, Gamma を探索
- 評価指標: Recall 99%時のPrecision
- Pruning: カスタム指標で早期打ち切り
- 安定性: boost_from_average=False, Hessian近似, 勾配スケーリング
"""

import pandas as pd
import numpy as np
import os
import pickle
import gc
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, average_precision_score
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# Focal Loss 実装 (クロージャ)
# ============================================================
def get_focal_loss(alpha, gamma):
    """
    Focal Lossを生成するクロージャ
    
    Args:
        alpha: 正例の重み (0.0~1.0)
        gamma: 難易度の重み (0.0~5.0)
    
    Returns:
        focal_loss_fixed: lgb.trainのfobj引数に渡す関数
    """
    def focal_loss_fixed(preds, train_data):
        y_true = train_data.get_label()
        
        # Logits -> Probability (数値安定性のためクリップ)
        p = 1.0 / (1.0 + np.exp(-preds))
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        # p_t: 正解クラスの確率
        p_t = y_true * p + (1 - y_true) * (1 - p)
        
        # alpha_t: クラスごとの重み
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        
        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** gamma
        
        # 1. 勾配 (Gradient) - 厳密解
        grad = alpha_t * focal_weight * (p - y_true)
        
        # 2. ヘッセ行列 (Hessian) - 近似解 (安定性重視)
        hess = alpha_t * focal_weight * p * (1 - p)
        hess = np.maximum(hess, 1e-7)  # 数値安定性
        
        # 3. 勾配スケーリング (学習進行促進)
        factor = 10.0
        return grad * factor, hess * factor
    
    return focal_loss_fixed


# ============================================================
# カスタム評価関数 (Recall 99%時のPrecision)
# ============================================================
def custom_eval_metric(preds, train_data):
    """
    LightGBM用カスタム評価関数
    Recall >= 98.5% を満たす最大Precisionを返す
    """
    y_true = train_data.get_label()
    
    # Logits -> Probability
    p = 1.0 / (1.0 + np.exp(-preds))
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, p)
    
    # Recall >= 0.985 の最大Precisionを探す
    target_recall = 0.985
    valid_indices = recall >= target_recall
    
    if valid_indices.sum() > 0:
        score = precision[valid_indices].max()
    else:
        score = 0.0
    
    return 'prec_at_rec99', score, True  # name, value, higher_is_better


# ============================================================
# Optuna Objective関数 (Focal Loss対応)
# ============================================================
class Stage2FocalLossObjective:
    """Stage 2のFocal Loss + ハイパーパラメータ最適化用Objective"""
    
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
        
        # Focal Lossパラメータ探索
        focal_alpha = trial.suggest_float('focal_alpha', 0.1, 0.9)
        focal_gamma = trial.suggest_float('focal_gamma', 0.0, 5.0)
        
        # LightGBMパラメータ探索
        # Focal Loss関数を生成 (paramsに設定するため先に作成)
        fobj = get_focal_loss(focal_alpha, focal_gamma)
        
        params = {
            'objective': fobj,  # LightGBM v4+: params内にカスタム目的関数を設定
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': self.random_state,
            
            # カスタム損失関数使用時の必須設定
            'boost_from_average': False,
            'is_unbalance': False,
            # scale_pos_weightは除外 (Focal LossのAlphaで制御)
            
            # 探索対象
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', 8, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'subsample': trial.suggest_float('subsample', 0.5, 0.9),
            'learning_rate': trial.suggest_float('learning_rate', 0.02, 0.2, log=True),
        }
        
        # Cross-Validation
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            X_train = self.X.iloc[train_idx]
            y_train = self.y[train_idx]
            X_val = self.X.iloc[val_idx]
            y_val = self.y[val_idx]
            
            # LightGBM Dataset
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
            
            # Pruning Callback (カスタム指標を監視)
            pruning_callback = LightGBMPruningCallback(trial, 'prec_at_rec99')
            
            try:
                model = lgb.train(
                    params,
                    dtrain,
                    num_boost_round=500,
                    valid_sets=[dval],
                    feval=custom_eval_metric,  # カスタム評価
                    callbacks=[
                        lgb.early_stopping(50, verbose=False),
                        pruning_callback
                    ]
                )
                
                # 評価 (Logits -> Probability)
                y_pred_logits = model.predict(X_val)
                y_pred_prob = 1.0 / (1.0 + np.exp(-y_pred_logits))
                
                # Recall 99%時のPrecisionを計算
                precision, recall, _ = precision_recall_curve(y_val, y_pred_prob)
                valid_indices = recall >= 0.985
                if valid_indices.sum() > 0:
                    fold_score = precision[valid_indices].max()
                else:
                    fold_score = 0.0
                
                scores.append(fold_score)
                
            except optuna.TrialPruned:
                raise
            
            del model
            gc.collect()
        
        mean_score = np.mean(scores)
        
        # 進捗表示
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if mean_score > self.best_score:
            self.best_score = mean_score
            print(f"   🏆 Trial {self.trial_count}: Prec@Rec99={mean_score:.4f} "
                  f"(α={focal_alpha:.2f}, γ={focal_gamma:.2f}) [NEW BEST!] [{elapsed/60:.1f}min]")
        else:
            print(f"   Trial {self.trial_count}: Prec@Rec99={mean_score:.4f} "
                  f"(α={focal_alpha:.2f}, γ={focal_gamma:.2f}) [{elapsed/60:.1f}min]")
        
        return mean_score


# ============================================================
# メイン
# ============================================================
def run_optuna_optimization(
    data_path: str = "results/two_stage_model/optuna_data/stage2_train_data.pkl",
    n_trials: int = 50,
    n_folds: int = 5,
    random_state: int = 42,
    output_dir: str = "results/two_stage_model/optuna_focal_loss_results"
):
    """Optuna最適化を実行 (Focal Loss対応)"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Stage 2 Optuna ハイパーパラメータ最適化 (Focal Loss)")
    print("評価指標: Precision @ Recall 99%")
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
        study_name='stage2_focal_loss_optimization',
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    
    objective = Stage2FocalLossObjective(X_s2, y_s2, n_folds=n_folds, random_state=random_state)
    
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
    print(f"\n🏆 ベストスコア: Precision@Recall99% = {best.value:.4f}")
    print(f"\n📋 ベストパラメータ:")
    for key, value in best.params.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    
    # 結果保存
    results_df = study.trials_dataframe()
    results_df.to_csv(os.path.join(output_dir, "optuna_trials.csv"), index=False)
    
    best_params_df = pd.DataFrame([best.params])
    best_params_df['prec_at_rec99'] = best.value
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
