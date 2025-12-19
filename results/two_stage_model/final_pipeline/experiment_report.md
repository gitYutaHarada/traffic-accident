# Focal Loss 実験レポート (Optuna最適化版)

**実行日時**: 2025-12-18 16:16:08
**実行時間**: 94.6秒

## パラメータ設定 (Optuna最適化)

| パラメータ | 値 |
|-----------|----| 
| Focal Alpha | 0.6321 |
| Focal Gamma | 1.1495 |
| num_leaves | 127 |
| max_depth | 6 |
| min_child_samples | 44 |
| reg_alpha | 2.3897 |
| reg_lambda | 2.2842 |
| colsample_bytree | 0.8646 |
| subsample | 0.6328 |
| learning_rate | 0.0477 |
| Stage 1 Recall Target | 99% |
| Under-sampling Ratio | 1:2 |
| Test Set Ratio | 20% |

## 結果サマリ

### Stage 1
- **閾値**: 0.0400
- **Recall**: 0.9913
- **フィルタリング率**: 24.52%

### Stage 2 (Focal Loss) - CV OOF評価

#### 固定閾値 (0.5) での評価
| 指標 | 値 |
|------|----| 
| Precision | 0.5929 |
| Recall | 0.0451 |
| F1 | 0.0838 |
| AUC | 0.8718 |

#### 動的閾値での評価 (CV OOF)
| Target Recall | 閾値 | Precision |
|---------------|------|----------|
| 99% | 0.2072 | 0.0122 |
| 98% | 0.2115 | 0.0126 |
| 95% | 0.2187 | 0.0147 |

## Baseline との比較 (CV OOF)

| 指標 | Baseline (Stage1) | Focal Loss (固定閾値) | 変化 |
|------|-------------------|----------------------|------|
| Precision | 0.0630 | 0.5929 | +841.45% |
| Recall | 0.7055 | 0.0451 | - |

## 予測スコア分布

```
mean=0.2402
std=0.0343
min=0.2041
max=0.6342
```

## 考察

- Focal Alpha=0.6321 は正例（死亡事故）の重みを調整
- Focal Gamma=1.1495 は難易度に応じた重み付け
- Optuna最適化により、Recall 99%時のPrecisionを最大化するパラメータを探索
- CV OOF と Test Set の結果が近いほど、汎化性能が高い
