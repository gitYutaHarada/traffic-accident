# Focal Loss 実験レポート (Optuna最適化版)

**実行日時**: 2025-12-22 15:24:08
**実行時間**: 106.5秒

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
- **閾値**: 0.0350
- **Recall**: 0.9921
- **フィルタリング率**: 24.89%

### Stage 2 (Focal Loss) - CV OOF評価

#### 固定閾値 (0.5) での評価
| 指標 | 値 |
|------|----| 
| Precision | 0.5822 |
| Recall | 0.0487 |
| F1 | 0.0899 |
| AUC | 0.8629 |

#### 動的閾値での評価 (CV OOF)
| Target Recall | 閾値 | Precision |
|---------------|------|----------|
| 99% | 0.1609 | 0.0124 |
| 98% | 0.1645 | 0.0132 |
| 95% | 0.1745 | 0.0149 |

## Baseline との比較 (CV OOF)

| 指標 | Baseline (Stage1) | Focal Loss (固定閾値) | 変化 |
|------|-------------------|----------------------|------|
| Precision | 0.0633 | 0.5822 | +820.33% |
| Recall | 0.7185 | 0.0487 | - |

## 予測スコア分布

```
mean=0.2116
std=0.0451
min=0.1568
max=0.6517
```

## 考察

- Focal Alpha=0.6321 は正例（死亡事故）の重みを調整
- Focal Gamma=1.1495 は難易度に応じた重み付け
- Optuna最適化により、Recall 99%時のPrecisionを最大化するパラメータを探索
- CV OOF と Test Set の結果が近いほど、汎化性能が高い
