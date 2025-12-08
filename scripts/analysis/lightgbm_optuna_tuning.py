"""
Optunaを使用したLightGBMハイパーパラメータチューニング

目的: PR-AUCを最大化するハイパーパラメータの探索
"""
import json
import os
import warnings
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

warnings.filterwarnings("ignore")

# 日本語フォントの設定
mpl.rcParams["font.family"] = "MS Gothic"

# 定数
RANDOM_STATE = 42
N_TRIALS = 100  # Optuna試行回数
N_FOLDS = 5  # 交差検証のfold数
TIMEOUT = 3600 * 2  # 最大2時間でタイムアウト

# パス設定
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "honhyo_model_ready.csv"
RESULTS_DIR = BASE_DIR / "results"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
VIZ_DIR = RESULTS_DIR / "visualizations"

# ディレクトリ作成
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)


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
    
    # ハイパーパラメータの探索空間
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        
        # 探索するパラメータ
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "n_estimators": trial.suggest_int("n_estimators", 500, 1500),
        
        # scale_pos_weightの微調整（ベース値の±20%）
        "scale_pos_weight": trial.suggest_float(
            "scale_pos_weight",
            base_scale_pos_weight * 0.8,
            base_scale_pos_weight * 1.2
        ),
    }
    
    # 交差検証
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    pr_aucs = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # モデル学習
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        # 予測
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # PR-AUCの計算
        pr_auc = average_precision_score(y_val, y_prob)
        pr_aucs.append(pr_auc)
        
        # Pruning: 最初のfoldで性能が悪い場合は早期終了
        if fold_idx == 0:
            trial.report(pr_auc, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
    
    # 平均PR-AUCを返す
    mean_pr_auc = np.mean(pr_aucs)
    return mean_pr_auc


def run_optimization(X, y, base_scale_pos_weight):
    """Optunaによる最適化実行"""
    print("\n" + "=" * 80)
    print(f"[OPTUNA] ハイパーパラメータ最適化開始")
    print(f"  試行回数: {N_TRIALS}")
    print(f"  交差検証: {N_FOLDS}-fold")
    print(f"  目的関数: PR-AUC最大化")
    print("=" * 80)
    
    # Studyの作成
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    
    # 最適化実行
    study.optimize(
        lambda trial: objective(trial, X, y, base_scale_pos_weight),
        n_trials=N_TRIALS,
        timeout=TIMEOUT,
        show_progress_bar=True,
    )
    
    print("\n[DONE] 最適化完了")
    print(f"  最良のPR-AUC: {study.best_value:.4f}")
    print(f"  完了した試行数: {len(study.trials)}")
    
    return study


def evaluate_best_model(study, X, y):
    """最良パラメータでモデルを評価"""
    print("\n" + "=" * 80)
    print("[EVAL] 最良パラメータでの詳細評価")
    print("=" * 80)
    
    best_params = study.best_params.copy()
    best_params.update({
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    })
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    fold_metrics = []
    y_true_all = []
    y_prob_all = []
    feature_importances = pd.DataFrame()
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # モデル学習
        model = lgb.LGBMClassifier(**best_params)
        model.fit(X_train, y_train)
        
        # 予測
        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        
        y_true_all.extend(y_val)
        y_prob_all.extend(y_prob)
        
        # メトリクス計算
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        pr_auc = average_precision_score(y_val, y_prob)
        
        print(f"  Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, PR-AUC: {pr_auc:.4f}")
        
        fold_metrics.append({
            "Fold": fold_idx + 1,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1,
            "PR_AUC": pr_auc,
        })
        
        # 特徴量重要度
        fi = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_,
            "fold": fold_idx + 1,
        })
        feature_importances = pd.concat([feature_importances, fi], axis=0)
    
    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all)
    
    # 全体のメトリクス
    roc_auc = roc_auc_score(y_true_all, y_prob_all)
    pr_auc_overall = average_precision_score(y_true_all, y_prob_all)
    
    print(f"\n[SCORE] 全体のスコア:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC: {pr_auc_overall:.4f}")
    
    metrics_df = pd.DataFrame(fold_metrics)
    
    return metrics_df, y_true_all, y_prob_all, feature_importances, best_params


def save_results(study, metrics_df, y_true, y_prob, feature_importances, best_params):
    """結果の保存"""
    print("\n[SAVE] 結果の保存中...")
    
    # 1. 最良パラメータの保存
    params_path = ANALYSIS_DIR / "optuna_best_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    print(f"[OK] 最良パラメータ: {params_path}")
    
    # 2. 最適化履歴の保存
    trials_df = study.trials_dataframe()
    history_path = ANALYSIS_DIR / "optuna_study_history.csv"
    trials_df.to_csv(history_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 最適化履歴: {history_path}")
    
    # 3. 交差検証メトリクスの保存
    cv_metrics_path = ANALYSIS_DIR / "optuna_cv_metrics.csv"
    metrics_df.to_csv(cv_metrics_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 交差検証メトリクス: {cv_metrics_path}")
    
    # 4. 特徴量重要度の保存
    feat_imp_mean = feature_importances.groupby("feature")["importance"].mean()
    feat_imp_mean = feat_imp_mean.sort_values(ascending=False)
    feat_imp_path = ANALYSIS_DIR / "optuna_feature_importance.csv"
    feat_imp_mean.to_csv(feat_imp_path, encoding="utf-8-sig")
    print(f"[OK] 特徴量重要度: {feat_imp_path}")
    
    # 5. 可視化の生成
    save_visualizations(study, y_true, y_prob, feat_imp_mean)
    
    print("[DONE] すべての結果を保存しました")


def save_visualizations(study, y_true, y_prob, feat_imp_mean):
    """可視化の生成と保存"""
    print("\n[VIZ] 可視化の生成中...")
    
    # 1. 最適化履歴
    try:
        fig = plot_optimization_history(study)
        fig.write_image(str(VIZ_DIR / "optuna_optimization_history.png"))
        print("[OK] 最適化履歴グラフ")
    except Exception as e:
        print(f"  Warning: 最適化履歴グラフの生成に失敗: {e}")
    
    # 2. パラメータ重要度
    try:
        fig = plot_param_importances(study)
        fig.write_image(str(VIZ_DIR / "optuna_param_importances.png"))
        print("[OK] パラメータ重要度グラフ")
    except Exception as e:
        print(f"  Warning: パラメータ重要度グラフの生成に失敗: {e}")
    
    # 3. パラレルコーディネート
    try:
        fig = plot_parallel_coordinate(study)
        fig.write_image(str(VIZ_DIR / "optuna_parallel_coordinate.png"))
        print("[OK] パラレルコーディネートグラフ")
    except Exception as e:
        print(f"  Warning: パラレルコーディネートグラフの生成に失敗: {e}")
    
    # 4. PR曲線
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, marker=".", label="Optuna最適化モデル")
    plt.xlabel("Recall (再現率)")
    plt.ylabel("Precision (適合率)")
    plt.title("Precision-Recall Curve (Optuna Optimized Model)")
    plt.legend()
    plt.grid(True)
    pr_curve_path = VIZ_DIR / "optuna_pr_curve.png"
    plt.savefig(pr_curve_path)
    plt.close()
    print("[OK] PR曲線")
    
    # 5. 混同行列
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["非死亡", "死亡"], yticklabels=["非死亡", "死亡"]
    )
    plt.title("Confusion Matrix (Optuna Optimized, Threshold=0.5)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    cm_path = VIZ_DIR / "optuna_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print("[OK] 混同行列")
    
    # 6. 特徴量重要度（Top 20）
    plt.figure(figsize=(10, 8))
    top20 = feat_imp_mean.head(20)
    sns.barplot(x=top20.values, y=top20.index, palette="viridis")
    plt.title("Feature Importance Top 20 (Optuna Optimized)")
    plt.xlabel("Importance")
    plt.tight_layout()
    fi_path = VIZ_DIR / "optuna_feature_importance_top20.png"
    plt.savefig(fi_path)
    plt.close()
    print("[OK] 特徴量重要度グラフ")


def main():
    """メイン処理"""
    print("=" * 80)
    print("LightGBM Optuna ハイパーパラメータチューニング")
    print("=" * 80)
    
    # データ読み込みと前処理
    X, y, base_scale_pos_weight = load_and_preprocess_data()
    
    # Optunaによる最適化
    study = run_optimization(X, y, base_scale_pos_weight)
    
    # 最良パラメータでの詳細評価
    metrics_df, y_true, y_prob, feature_importances, best_params = evaluate_best_model(study, X, y)
    
    # 結果の保存
    save_results(study, metrics_df, y_true, y_prob, feature_importances, best_params)
    
    print("\n" + "=" * 80)
    print("[COMPLETE] すべての処理が完了しました")
    print("=" * 80)
    print(f"\n最良パラメータ:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    print(f"\n交差検証の平均メトリクス:")
    print(metrics_df[["Accuracy", "Precision", "Recall", "F1_Score", "PR_AUC"]].mean())


if __name__ == "__main__":
    main()

