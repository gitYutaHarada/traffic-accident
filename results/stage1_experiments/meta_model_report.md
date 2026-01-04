# Stage 1 ロジスティックメタモデル実験レポート

**実行日時**: 2025-12-24 16:48:00
**Target Recall**: 99.5%
**ベースライン**: Max Probability

## OOF結果

| 手法 | 閾値 | Recall | Pass Rate | Precision | AUC |
|------|------|--------|-----------|-----------|-----|
| Max Probability (Baseline) | 0.0371 | 0.9950 | 78.44% | 0.0109 | 0.9105 |
| Logistic Meta-Model | 0.0998 | 0.9950 | 76.82% | 0.0111 | 0.9124 |
| Simple Average | 0.0315 | 0.9950 | 78.66% | 0.0109 | 0.9112 |
| OR Ensemble (Reference) | LGBM:0.0289, Cat:0.0303 | 0.9975 | 83.67% | 0.0102 | - |

## Test結果

| 手法 | Recall | Pass Rate | Precision | AUC |
|------|--------|-----------|-----------|-----|
| Max Probability (Baseline) | 0.9957 | 78.64% | 0.0109 | 0.9100 |
| Logistic Meta-Model | 0.9942 | 77.26% | 0.0110 | 0.9112 |
| Simple Average | 0.9948 | 79.23% | 0.0108 | 0.9104 |
| OR Ensemble (Reference) | 0.9969 | 84.00% | 0.0102 | - |

## 最良手法

**Logistic Meta-Model** がTest Pass Rate **77.26%** で最良。
ベースライン(Max Probability)との比較で **1.38%** の改善。

## メタモデル係数

| 特徴量 | 係数 |
|--------|------|
| prob_catboost | 5.3863 |
| prob_lgbm | 0.8856 |

**Intercept**: -2.4010
