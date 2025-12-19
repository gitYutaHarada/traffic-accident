# TabNet Stage 2 実験レポート

**実行日時**: 2025-12-19 14:12:50
**実行時間**: 128.6秒

## モデル構成
- **Stage 1**: LightGBM Binary Classification (死亡 vs その他)
- **Stage 2**: TabNet Binary Classification (負傷 vs 死亡)
- **TabNet設定**: n_d=16, n_a=16, n_steps=5
- **バッチサイズ**: 4096

## 結果サマリ

### Stage 1 (Recall 95%)
- **閾値**: 0.1120
- **Recall**: 0.9506
- **フィルタリング率**: 56.04%
- **負傷事故(Class 1) 通過率**: 43.5%

### Stage 2 TabNet (CV OOF)

**Best F1 閾値 (0.8932) での評価**:
| 指標 | 値 |
|------|----| 
| F1 Score | 0.2464 |
| Precision | 0.2112 |
| Recall | 0.2957 |

**Overall Metrics (全体に対する評価)**:
| 指標 | 値 |
|------|----| 
| Final Precision | 0.2112 |
| Final Recall | 0.2811 |
| Final F1 | 0.2412 |
| AUC | 0.8976 |

### テストセット評価 (Hold-Out 20%)

**CV最適閾値 (0.8932) を適用**:
| 指標 | 値 |
|------|----| 
| Precision | 0.2266 |
| Recall | 0.2745 |
| F1 | 0.2483 |
| AUC | 0.8980 |

**参考: Test Ideal F1**: 0.2564

## 考察

- TabNetはAttention機構により、決定木では捉えにくい複雑な特徴表現を学習
- 負傷事故（Hard Negatives）と死亡事故の微細な違いをDeep Learningの表現力で識別
- LightGBMとの比較で、Precision/Recallの改善を確認する必要あり
