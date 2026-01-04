# Model Disagreement Analysis Report

Generated: 2025-12-30 17:01

## 1. Analysis Parameters

| Item | Value |
|:---|---:|
| **Threshold** | **0.1400** |
| Total records | 282,376 |

## 2. Pattern Distribution

| Pattern | Count | Ratio |
|:---|---:|---:|
| Both Negative | 161,148 | 57.1% |
| CatBoost Only | 118,519 | 42.0% |
| Both Positive | 2,605 | 0.9% |
| TabNet Only | 104 | 0.0% |

**Disagreement rate: 42.0%** (118,623 / 282,376)

## 3. Stacking Accuracy on Disagreement Cases

| Pattern | Count | Fatal | Stacking Accuracy | Recall |
|:---|---:|---:|---:|---:|
| Both Positive | 2,605 | 721 | 38.9% | 90.8% |
| Both Negative | 161,148 | 275 | 99.8% | 1.1% |
| TabNet Only | 104 | 19 | 52.9% | 68.4% |
| CatBoost Only | 118,519 | 1576 | 98.3% | 5.8% |

## 4. Aggressive vs Conservative Model

| Pattern | Aggressive Model | Fatal | Aggressive Correct | Stacking Decision |
|:---|:---:|---:|---:|:---|
| TabNet Only | TabNet | 19 | 19 | Agg:56, Cons:48 |
| CatBoost Only | CatBoost | 1576 | 1576 | Agg:644, Cons:117875 |

## 5. Findings

> Model disagreement rate is **42.0%**. Ensemble effect is expected.

> Both models agreed on positive: **2,605** cases. High confidence alerts.

## 6. Visualizations

- `disagreement_patterns.png`: Pattern distribution and accuracy comparison
