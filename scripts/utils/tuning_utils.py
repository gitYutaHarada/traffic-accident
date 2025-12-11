"""
LightGBMハイパーパラメータチューニング用ユーティリティモジュール

このモジュールには、チューニングスクリプトから分離された再利用可能な機能が含まれます：
- データローダーと検証
- 評価メトリクスの計算
- レポート生成
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_and_validate_data(
    data_path: Path, target_col: str, verbose: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    データを読み込み、基本的な検証を行う
    
    Args:
        data_path: データファイルのパス
        target_col: 目的変数のカラム名
        verbose: ログ出力するかどうか
    
    Returns:
        X: 特徴量DataFrame
        y: 目的変数Series
    """
    if verbose:
        print(f"\n[DATA] データ読み込み中: {data_path}")
    
    if not data_path.exists():
        raise FileNotFoundError(f"データファイルが見つかりません: {data_path}")
    
    df = pd.read_csv(data_path)
    
    if verbose:
        print(f"[OK] データ読み込み完了: {len(df):,} 件")
    
    if target_col not in df.columns:
        raise ValueError(f"目的変数 '{target_col}' が見つかりません")
    
    # 特徴量と目的変数の分離
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    if verbose:
        print(f"[INFO] 特徴量数: {X.shape[1]}")
        print(f"[INFO] クラス分布: 0={len(y) - y.sum():,}, 1={y.sum():,}")
    
    return X, y


def calculate_all_metrics(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    すべての評価指標を計算
    
    Args:
        y_true: 真のラベル
        y_prob: 予測確率
        threshold: 分類閾値
    
    Returns:
        メトリクスの辞書
    """
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
    }
    
    return metrics


def calculate_metrics_at_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float]
) -> pd.DataFrame:
    """
    複数の閾値で評価指標を計算
    
    Args:
        y_true: 真のラベル
        y_prob: 予測確率
        thresholds: 評価する閾値のリスト
    
    Returns:
        閾値ごとの評価指標DataFrame
    """
    results = []
    
    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        
        results.append({
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1": f1_score(y_true, y_pred, zero_division=0),
        })
    
    return pd.DataFrame(results)


def generate_tuning_report(
    best_params: Dict,
    cv_metrics: pd.DataFrame,
    threshold_metrics: pd.DataFrame,
    feature_importance: pd.Series,
    output_path: Path,
    study_summary: Optional[Dict] = None
) -> None:
    """
    Markdown形式のチューニングレポートを生成
    
    Args:
        best_params: 最良パラメータ
        cv_metrics: 交差検証メトリクス
        threshold_metrics: 閾値ごとのメトリクス
        feature_importance: 特徴量重要度
        output_path: 出力パス
        study_summary: Optunaスタディのサマリー（オプション）
    """
    report_lines = []
    
    # ヘッダー
    report_lines.append("# LightGBM ハイパーパラメータチューニング結果")
    report_lines.append(f"\n**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Optunaサマリー
    if study_summary:
        report_lines.append("## 最適化サマリー\n")
        report_lines.append(f"- **試行回数**: {study_summary.get('n_trials', 'N/A')}")
        report_lines.append(f"- **最良スコア (PR-AUC)**: {study_summary.get('best_value', 'N/A'):.4f}")
        report_lines.append("")
    
    # 最良パラメータ
    report_lines.append("## 最良パラメータ\n")
    report_lines.append("| パラメータ | 値 |")
    report_lines.append("|-----------|-----|")
    for key, value in best_params.items():
        if isinstance(value, float):
            report_lines.append(f"| {key} | {value:.6f} |")
        else:
            report_lines.append(f"| {key} | {value} |")
    report_lines.append("")
    
    # 交差検証結果
    report_lines.append("## 交差検証結果\n")
    report_lines.append("### Fold別メトリクス\n")
    report_lines.append(cv_metrics.to_markdown(index=False))
    report_lines.append("\n### 平均スコア\n")
    mean_metrics = cv_metrics.select_dtypes(include=[np.number]).mean()
    report_lines.append("| 指標 | スコア |")
    report_lines.append("|------|--------|")
    for metric, value in mean_metrics.items():
        report_lines.append(f"| {metric} | {value:.4f} |")
    report_lines.append("")
    
    # 閾値分析
    report_lines.append("## 閾値別性能\n")
    report_lines.append(threshold_metrics.to_markdown(index=False))
    report_lines.append("")
    
    # 特徴量重要度 Top 20
    report_lines.append("## 特徴量重要度 Top 20\n")
    top_features = feature_importance.head(20)
    report_lines.append("| 順位 | 特徴量 | 重要度 |")
    report_lines.append("|------|--------|--------|")
    for rank, (feature, importance) in enumerate(top_features.items(), 1):
        report_lines.append(f"| {rank} | {feature} | {importance:.2f} |")
    report_lines.append("")
    
    # ファイル出力
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"[OK] レポート生成: {output_path}")


def create_results_directory(base_dir: Path, prefix: str = "tuning") -> Path:
    """
    タイムスタンプ付きの結果ディレクトリを作成
    
    Args:
        base_dir: ベースディレクトリ
        prefix: ディレクトリ名のプレフィックス
    
    Returns:
        作成されたディレクトリのパス
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = base_dir / f"{prefix}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 可視化用サブディレクトリも作成
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    return results_dir
