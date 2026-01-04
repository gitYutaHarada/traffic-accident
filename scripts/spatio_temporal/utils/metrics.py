"""
評価指標ユーティリティ
====================
"""

import numpy as np
from typing import Dict


def compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    クラス重みの計算（不均衡対策）
    """
    n_samples = len(y)
    n_classes = len(np.unique(y))
    
    class_counts = np.bincount(y.astype(int))
    weights = n_samples / (n_classes * class_counts)
    
    return {i: w for i, w in enumerate(weights)}


def compute_pos_weight(y: np.ndarray) -> float:
    """
    正例の重み計算（BCEWithLogitsLoss用）
    """
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    
    return n_neg / n_pos if n_pos > 0 else 1.0
