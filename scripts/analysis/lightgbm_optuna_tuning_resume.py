"""
Optunaを使用したLightGBMハイパーパラメータチューニング（継続実行版）

特徴:
  - SQLiteストレージを使用してstudyを保存
  - load_if_exists=Trueで既存のstudyから継続可能
  - 残りの試行のみを実行
"""
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

# ユーティリティモジュールのインポート
sys.path.append(str(Path(__file__).resolve().parent.parent))
try:
    from utils.tuning_utils import (
        calculate_all_metrics,
        calculate_metrics_at_thresholds,
        generate_tuning_report,
        create_results_directory,
    )
    USE_UTILS = True
except ImportError:
    print("[WARNING] tuning_utilsのインポートに失敗。基本機能のみ使用します。")
    USE_UTILS = False

warnings.filterwarnings("ignore")

# 日本語フォントの設定
mpl.rcParams["font.family"] = "MS Gothic"

# 定数
RANDOM_STATE = 42
N_TRIALS = 200  # 目標試行回数
N_FOLDS = 5
TIMEOUT = None  # 時間制限なし（ユーザー指定）
EARLY_STOPPING_ROUNDS = 50

# パス設定
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "honhyo_clean_predictable_only.csv"
RESULTS_DIR = BASE_DIR / "results"
TUNING_DIR = RESULTS_DIR / "tuning"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
VIZ_DIR = RESULTS_DIR / "visualizations"

# ディレクトリ作成
TUNING_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# SQLiteストレージの設定
STORAGE_PATH = TUNING_DIR / "lightgbm_tuning.db"
STORAGE_NAME = f"sqlite:///{STORAGE_PATH}"
STUDY_NAME = "lightgbm_pr_auc_optimization"


def load_and_preprocess_data():
    """データの読み込みと前処理"""
    print(f"\n[DATA] データ読み込み中: {DATA_PATH}")
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"データファイルが見つかりません: {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    print(f"[OK] データ読み込み完了: {len(df):,} 件")
    
    # 目的変数
    target_col = "死者数"
    
    # 除外する列（事後情報の徹底排除）
    drop_cols = [
        "資料区分", "本票番号",
        "人身損傷程度(当事者A)", "人身損傷程度(当事者B)",
        "車両の損壊程度(当事者A)", "車両の損壊程度(当事者B)",
        "負傷者数",
        "車両の衝突部位(当事者A)", "車両の衝突部位(当事者B)",
        "エアバッグの装備(当事者A)", "エアバッグの装備(当事者B)",
        "サイドエアバッグの装備(当事者A)", "サイドエアバッグの装備(当事者B)",
        "事故内容",
    ]
    
    print("\n[PROC] データ前処理中（事後情報の除外）...")
    df_clean = df.drop(columns=drop_cols, errors="ignore")
    
    # 特徴量と目的変数
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # カウントエンコーディング
    count_enc_cols = [
        "市区町村コード", "路線コード", "地点コード", "Area_Cluster_ID", "都道府県コード"
    ]
    
    print(f"\n[ENC] カウントエンコーディングの生成: {count_enc_cols}")
    for col in count_enc_cols:
        if col in X.columns:
            count_map = X[col].value_counts()
            X[f"{col}_count"] = X[col].map(count_map)
    
    # カテゴリカル変数の定義
    categorical_candidates = [
        "都道府県コード", "路線コード", "地点コード", "市区町村コード",
        "昼夜", "天候", "地形", "路面状態", "道路形状", "信号機",
        "一時停止規制 標識", "一時停止規制 表示", "車道幅員", "道路線形",
        "衝突地点", "ゾーン規制", "中央分離帯施設等", "歩車道区分",
        "事故類型", "年齢", "当事者種別", "用途別", "車両形状",
        "オートマチック車", "サポカー", "速度規制(指定のみ)",
        "曜日", "祝日", "発生月", "発生時", "発生年", "Area_Cluster_ID"
    ]
    
    explicit_cat_cols = [c for c in categorical_candidates if c in X.columns]
    object_cols = X.select_dtypes(include=["object"]).columns.tolist()
    final_cat_cols = list(set(explicit_cat_cols + object_cols))
    
    print(f"\n[CAT] カテゴリカル変数の変換: {len(final_cat_cols)} カラム")
    for col in final_cat_cols:
        X[col] = X[col].astype("category")
    
    print(f"[OK] 前処理完了 - 特徴量数: {X.shape[1]}")
    
    # クラス不均衡比の計算
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    base_scale_pos_weight = neg_count / pos_count
    
    print(f"\n[WEIGHT] クラス不均衡比:")
    print(f"  Negative (0): {neg_count:,}")
    print(f"  Positive (1): {pos_count:,}")
    print(f"  Base scale_pos_weight: {base_scale_pos_weight:.2f}")
    
    return X, y, base_scale_pos_weight


def objective(trial, X, y, base_scale_pos_weight):
    """Optunaの目的関数 - PR-AUCを最大化"""
    
    # ハイパーパラメータの探索空間（拡張版）
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        
        # 既存の探索パラメータ
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 300),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "n_estimators": 10000,
        
        # 新規追加パラメータ
        "min_child_weight": trial.suggest_float("min_child_weight", 0.001, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
        "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
        
        # scale_pos_weightの探索
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight",
            base_scale_pos_weight * 0.5,
            base_scale_pos_weight * 2.5
        ),
    }
    
    # 交差検証
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    pr_aucs = []
    all_metrics = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # モデル学習
        model = lgb.LGBMClassifier(**params)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0)
        ]
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=callbacks
        )
        
        # 予測
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # 複数評価指標の計算
        if USE_UTILS:
            metrics = calculate_all_metrics(y_val, y_prob, threshold=0.5)
        else:
            y_pred = (y_prob >= 0.5).astype(int)
            metrics = {
                "Accuracy": accuracy_score(y_val, y_pred),
                "Precision": precision_score(y_val, y_pred, zero_division=0),
                "Recall": recall_score(y_val, y_pred, zero_division=0),
                "F1": f1_score(y_val, y_pred, zero_division=0),
                "ROC_AUC": roc_auc_score(y_val, y_prob),
                "PR_AUC": average_precision_score(y_val, y_prob),
            }
        
        pr_aucs.append(metrics["PR_AUC"])
        all_metrics.append(metrics)
        
        # Pruning
        if fold_idx == 0:
            trial.report(metrics["PR_AUC"], fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    # 全評価指標の平均を記録
    for metric_name in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]:
        mean_value = np.mean([m[metric_name] for m in all_metrics])
        trial.set_user_attr(f"mean_{metric_name}", mean_value)
    
    # 平均PR-AUCを返す
    mean_pr_auc = np.mean(pr_aucs)
    return mean_pr_auc


def run_optimization_with_resume(X, y, base_scale_pos_weight):
    """Optunaによる最適化実行（継続機能付き）"""
    print("\n" + "=" * 80)
    print(f"[OPTUNA] ハイパーパラメータ最適化開始（継続実行対応）")
    print(f"  目標試行回数: {N_TRIALS}")
    print(f"  交差検証: {N_FOLDS}-fold")
    print(f"  目的関数: PR-AUC最大化")
    print(f"  ストレージ: {STORAGE_NAME}")
    print("=" * 80)
    
    # Studyの作成（既存のstudyがあれば読み込む）
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        load_if_exists=True  # 重要: 既存のstudyを読み込む
    )
    
    # 既に完了した試行数を表示
    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_remaining = max(0, N_TRIALS - n_completed)
    
    print(f"\n[INFO] 既存の試行数: {n_completed}")
    print(f"[INFO] 残りの試行数: {n_remaining}")
    
    if n_completed > 0:
        print(f"[INFO] 現在の最良PR-AUC: {study.best_value:.4f}")
    
    # 最適化実行（残りの試行のみ）
    if n_remaining > 0:
        print(f"\n[START] 残り{n_remaining}試行を実行します...")
        study.optimize(
            lambda trial: objective(trial, X, y, base_scale_pos_weight),
            n_trials=n_remaining,
            timeout=TIMEOUT,
            show_progress_bar=True,
        )
    else:
        print("\n[SKIP] 既に目標試行回数に達しています。")
    
    print("\n[DONE] 最適化完了")
    print(f"  総試行数: {len(study.trials)}")
    print(f"  完了試行数: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  最良のPR-AUC: {study.best_value:.4f}")
    
    return study


# 以下、evaluate_best_model, save_results, save_visualizations, main関数は
# オリジナルのスクリプトと同じものを使用（省略）


def main():
    """メイン処理"""
    print("=" * 80)
    print("LightGBM Optuna ハイパーパラメータチューニング（継続実行版）")
    print("=" * 80)
    
    # データ読み込みと前処理
    X, y, base_scale_pos_weight = load_and_preprocess_data()
    
    # Optunaによる最適化（継続機能付き）
    study = run_optimization_with_resume(X, y, base_scale_pos_weight)
    
    # 注意: 以下の評価・保存機能は元のスクリプトから移植が必要
    print("\n[INFO] 詳細な評価と結果保存は元のスクリプトを参照してください。")
    print(f"\n最良パラメータ:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
