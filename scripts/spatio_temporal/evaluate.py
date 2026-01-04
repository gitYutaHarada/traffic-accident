"""
評価モジュール
=============
PR-AUC, ROC-AUC, Precision@k, Recall@k, ECE, Brier Score
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    roc_curve, brier_score_loss, precision_score, recall_score, f1_score
)
import time


def compute_pr_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """PR-AUC (Average Precision) の計算"""
    return average_precision_score(y_true, y_pred)


def compute_roc_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """ROC-AUC の計算"""
    return roc_auc_score(y_true, y_pred)


def compute_precision_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Precision@k: Top-k予測のうち正例の割合
    """
    n = len(y_true)
    k = min(k, n)
    
    # 予測確率でソートしてTop-kを取得
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    
    return y_true[top_k_indices].sum() / k


def compute_recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Recall@k: 全正例のうちTop-kに含まれる割合
    """
    n = len(y_true)
    k = min(k, n)
    
    top_k_indices = np.argsort(y_pred)[::-1][:k]
    
    total_positives = y_true.sum()
    if total_positives == 0:
        return 0.0
    
    return y_true[top_k_indices].sum() / total_positives


def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE)
    
    予測確率と実際の頻度のずれを測定
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # このビンに入るサンプル
        in_bin = (y_pred >= bin_lower) & (y_pred < bin_upper)
        n_in_bin = in_bin.sum()
        
        if n_in_bin > 0:
            # 実際の正例率
            accuracy_in_bin = y_true[in_bin].mean()
            # 予測確率の平均
            confidence_in_bin = y_pred[in_bin].mean()
            
            ece += (n_in_bin / total_samples) * abs(accuracy_in_bin - confidence_in_bin)
    
    return ece


def compute_brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Brier Score の計算"""
    return brier_score_loss(y_true, y_pred)


def compute_metrics_at_threshold(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    threshold: float
) -> Dict[str, float]:
    """特定の閾値での評価指標"""
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    return {
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
    }


def find_threshold_for_recall(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    target_recall: float
) -> Tuple[float, float]:
    """
    目標Recallを達成する閾値とそのときのPrecisionを返す
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    
    # target_recall以上を達成する閾値を探す
    valid_idx = np.where(recalls >= target_recall)[0]
    
    if len(valid_idx) == 0:
        return 0.0, 0.0
    
    # Precisionが最大となる閾値
    best_idx = valid_idx[np.argmax(precisions[valid_idx])]
    
    if best_idx < len(thresholds):
        return thresholds[best_idx], precisions[best_idx]
    else:
        return 0.0, precisions[best_idx]


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k_values: List[int] = [100, 500, 1000],
    recall_targets: List[float] = [0.99, 0.95, 0.90],
) -> Dict:
    """
    包括的なモデル評価
    
    Returns:
        metrics: 評価指標の辞書
    """
    metrics = {}
    
    # 基本指標
    metrics['pr_auc'] = compute_pr_auc(y_true, y_pred)
    metrics['roc_auc'] = compute_roc_auc(y_true, y_pred)
    metrics['ece'] = compute_ece(y_true, y_pred)
    metrics['brier_score'] = compute_brier_score(y_true, y_pred)
    
    # Precision@k, Recall@k
    for k in k_values:
        metrics[f'precision_at_{k}'] = compute_precision_at_k(y_true, y_pred, k)
        metrics[f'recall_at_{k}'] = compute_recall_at_k(y_true, y_pred, k)
    
    # 動的閾値評価
    for target_recall in recall_targets:
        thresh, prec = find_threshold_for_recall(y_true, y_pred, target_recall)
        metrics[f'threshold_at_recall_{int(target_recall*100)}'] = thresh
        metrics[f'precision_at_recall_{int(target_recall*100)}'] = prec
    
    # 固定閾値評価
    for thresh in [0.3, 0.5, 0.7]:
        thresh_metrics = compute_metrics_at_threshold(y_true, y_pred, thresh)
        for k, v in thresh_metrics.items():
            metrics[f'{k}_at_{thresh}'] = v
    
    return metrics


def measure_inference_time(
    model,
    sample_input,
    n_samples: int = 100,
    warmup: int = 10,
) -> Dict:
    """
    推論時間の計測
    
    Returns:
        timing: 推論時間統計
    """
    import torch
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*sample_input) if isinstance(sample_input, tuple) else model(sample_input)
    
    # 計測
    times = []
    with torch.no_grad():
        for _ in range(n_samples):
            start = time.perf_counter()
            _ = model(*sample_input) if isinstance(sample_input, tuple) else model(sample_input)
            times.append(time.perf_counter() - start)
    
    times = np.array(times) * 1000  # ms
    
    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'median_ms': float(np.median(times)),
        'min_ms': float(np.min(times)),
        'max_ms': float(np.max(times)),
    }


def generate_evaluation_report(
    results: Dict[str, Dict],
    output_path: str,
) -> str:
    """
    評価結果のMarkdownレポート生成
    """
    report = []
    report.append("# Spatio-Temporal Model 評価レポート\n")
    report.append(f"生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # モデル比較テーブル
    report.append("\n## モデル比較\n")
    
    # 主要指標
    report.append("### 主要評価指標\n")
    report.append("| モデル | PR-AUC | ROC-AUC | ECE | Brier Score |\n")
    report.append("|--------|--------|---------|-----|-------------|\n")
    
    for model_name, metrics in results.items():
        report.append(
            f"| {model_name} | {metrics.get('pr_auc', 0):.4f} | "
            f"{metrics.get('roc_auc', 0):.4f} | {metrics.get('ece', 0):.4f} | "
            f"{metrics.get('brier_score', 0):.4f} |\n"
        )
    
    # Precision/Recall@k
    report.append("\n### Precision/Recall@k\n")
    report.append("| モデル | P@100 | R@100 | P@500 | R@500 | P@1000 | R@1000 |\n")
    report.append("|--------|-------|-------|-------|-------|--------|--------|\n")
    
    for model_name, metrics in results.items():
        report.append(
            f"| {model_name} | "
            f"{metrics.get('precision_at_100', 0):.4f} | {metrics.get('recall_at_100', 0):.4f} | "
            f"{metrics.get('precision_at_500', 0):.4f} | {metrics.get('recall_at_500', 0):.4f} | "
            f"{metrics.get('precision_at_1000', 0):.4f} | {metrics.get('recall_at_1000', 0):.4f} |\n"
        )
    
    # 動的閾値
    report.append("\n### 動的閾値評価\n")
    report.append("| モデル | Recall=99% Precision | Recall=95% Precision | Recall=90% Precision |\n")
    report.append("|--------|---------------------|---------------------|---------------------|\n")
    
    for model_name, metrics in results.items():
        report.append(
            f"| {model_name} | "
            f"{metrics.get('precision_at_recall_99', 0):.4f} | "
            f"{metrics.get('precision_at_recall_95', 0):.4f} | "
            f"{metrics.get('precision_at_recall_90', 0):.4f} |\n"
        )
    
    report_text = "".join(report)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    return report_text


class ModelEvaluator:
    """モデル評価クラス"""
    
    def __init__(self, output_dir: str = "results/spatio_temporal"):
        self.output_dir = output_dir
        self.results = {}
    
    def add_result(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray):
        """評価結果を追加"""
        metrics = evaluate_model(y_true, y_pred)
        self.results[model_name] = metrics
        
        print(f"\n📊 {model_name} 評価結果:")
        print(f"   PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"   ECE: {metrics['ece']:.4f}")
    
    def generate_report(self):
        """レポート生成"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        
        report_path = os.path.join(self.output_dir, "evaluation_report.md")
        report = generate_evaluation_report(self.results, report_path)
        
        print(f"\n📄 レポート生成: {report_path}")
        
        return report
    
    def save_results(self):
        """結果をJSON形式で保存"""
        import json
        import os
        
        os.makedirs(self.output_dir, exist_ok=True)
        results_path = os.path.join(self.output_dir, "results_summary.json")
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"📄 結果保存: {results_path}")
