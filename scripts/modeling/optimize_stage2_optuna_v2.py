"""
Stage 2 Optuna ハイパーパラメータ最適化 v2 (修正版)
=====================================================
boost_from_average=True に戻し、探索範囲を絞った再挑戦

- Focal Loss: Alpha, Gamma を狭い範囲で探索 (手動成功パラメータ付近)
- 評価指標: Recall 98.5%時のPrecision
- LGBMClassifier (sklearn API) 使用
"""

import pandas as pd
import numpy as np
import os
import sys
import gc
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, average_precision_score
import lightgbm as lgb
import optuna
from scipy.special import expit
import warnings

# パイプラインをインポートするためにパスを追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.modeling.train_two_stage_final import TwoStageFinalPipeline

warnings.filterwarnings('ignore')


# ============================================================
# Focal Loss 実装 (sklearn API用: y_true, preds の順序)
# ============================================================
def get_focal_loss_sklearn(alpha, gamma):
    """
    Focal Lossを生成するクロージャ (sklearn API用)
    
    Args:
        alpha: 正例の重み (0.0~1.0)
        gamma: 難易度の重み (0.0~5.0)
    
    Returns:
        focal_loss_fn: LGBMClassifierのobjective引数に渡す関数
    """
    def focal_loss_fn(y_true, preds):
        # sklearn APIでは (y_true, preds) の順序
        p = expit(preds)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        p_t = y_true * p + (1 - y_true) * (1 - p)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_weight = (1 - p_t) ** gamma
        
        grad = alpha_t * focal_weight * (p - y_true)
        hess = alpha_t * focal_weight * p * (1 - p)
        hess = np.maximum(hess, 1e-7)
        
        # 【追加】スケーリング係数 (Factor)
        # 勾配が小さすぎて学習が進まないのを防ぐため、値を大きくする
        factor = 10.0
        return grad * factor, hess * factor
    
    return focal_loss_fn


# ============================================================
# Optuna Objective関数 (修正版)
# ============================================================
class Stage2ObjectiveV2:
    """Stage 2のFocal Loss + ハイパーパラメータ最適化 (修正版)"""
    
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
        
        # Focal Lossパラメータ探索 (狭い範囲に絞る)
        # 手動設定 (Alpha=0.75, Gamma=1.0) の成功付近
        focal_alpha = trial.suggest_float('focal_alpha', 0.60, 0.90)
        focal_gamma = trial.suggest_float('focal_gamma', 0.5, 2.0)
        
        # Focal Loss関数を生成
        fobj = get_focal_loss_sklearn(focal_alpha, focal_gamma)
        
        # LightGBMパラメータ (探索範囲を絞る)
        # objective は params から削除し、LGBMClassifier のインスタンス化時に渡す
        params = {
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': self.random_state,
            
            # boost_from_average=True (デフォルト) を維持！
            # is_unbalance, scale_pos_weight は使わない (Focal LossのAlphaで制御)
            'is_unbalance': False,
            
            # 探索対象 (範囲を絞る)
            'num_leaves': trial.suggest_int('num_leaves', 64, 192),
            'max_depth': trial.suggest_int('max_depth', 6, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 50),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
            'subsample': trial.suggest_float('subsample', 0.5, 0.8),
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.1, log=True),
            'n_estimators': 1000,
        }
        
        # Cross-Validation
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            X_train = self.X.iloc[train_idx]
            y_train = self.y[train_idx]
            X_val = self.X.iloc[val_idx]
            y_val = self.y[val_idx]
            
            try:
                # objective は インスタンス化時に指定 (sklearn API の推奨方法)
                model = lgb.LGBMClassifier(objective=fobj, **params)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                
                # 評価 (Logits -> Probability)
                y_pred_raw = model.predict(X_val, raw_score=True)
                y_pred_prob = expit(y_pred_raw)
                
                # PR-AUC (Average Precision) を計算
                # 閾値に依存せず、モデルの分離能力全体を評価する最もロバストな指標
                from sklearn.metrics import average_precision_score
                fold_score = average_precision_score(y_val, y_pred_prob)
                
                scores.append(fold_score)
                
            except Exception as e:
                print(f"      [Fold {fold+1}] Error: {e}")
                scores.append(0.0)
            
            del model
            gc.collect()
        
        mean_score = np.mean(scores)
        
        # 進捗表示
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if mean_score > self.best_score:
            self.best_score = mean_score
            print(f"   🏆 Trial {self.trial_count}: PR-AUC={mean_score:.4f} "
                  f"(α={focal_alpha:.2f}, γ={focal_gamma:.2f}) [NEW BEST!] [{elapsed/60:.1f}min]")
        else:
            print(f"   Trial {self.trial_count}: PR-AUC={mean_score:.4f} "
                  f"(α={focal_alpha:.2f}, γ={focal_gamma:.2f}) [{elapsed/60:.1f}min]")
        
        return mean_score


# ============================================================
# メイン
# ============================================================
def run_optuna_optimization_v2(
    n_trials: int = 50,
    n_folds: int = 5,
    random_state: int = 42,
    output_dir: str = "results/two_stage_model/optuna_focal_loss_v2_results"
):
    """Optuna最適化を実行 (修正版: パイプライン統合)"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Stage 2 Optuna ハイパーパラメータ最適化 v2 (パイプライン統合版)")
    print("評価指標: PR-AUC (Average Precision)")
    print("設定: boost_from_average=True, 探索範囲絞り込み")
    print(f"試行回数: {n_trials}")
    print("=" * 70)
    
    # データ生成 (パイプライン実行)
    print("\n🚀 Pipelineを実行して最新データを生成中...")
    pipeline = TwoStageFinalPipeline()
    X_s2, y_s2 = pipeline.get_stage2_data()
    
    n_pos = y_s2.sum()
    n_neg = len(y_s2) - n_pos
    print(f"   比率 (Neg:Pos) = 1:{n_neg//n_pos}")
    
    # Optuna Study作成
    print("\n🔍 最適化開始...")
    print("-" * 70)
    
    study = optuna.create_study(
        direction='maximize',
        study_name='stage2_focal_loss_v2_optimization',
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    
    objective = Stage2ObjectiveV2(X_s2, y_s2, n_folds=n_folds, random_state=random_state)
    
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
    print(f"\n🏆 ベストスコア: Precision@Recall98.5% = {best.value:.4f}")
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
    import sys
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    
    run_optuna_optimization_v2(n_trials=n_trials)
