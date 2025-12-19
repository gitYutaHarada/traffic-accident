# 多クラス分類 Stage 2 実験レポート

**実行日時**: 2025-12-19 14:17:19
**実行時間**: 96.3秒

## モデル構成
- **Stage 1**: Binary Classification (死亡 vs その他)
- **Stage 2**: Multiclass Classification (0=無傷, 1=負傷, 2=死亡)
- **Objective**: multiclass (class_weight使用)

## 結果サマリ
        
### Stage 1 (Recall 95%)
- **閾値**: 0.1120
- **Recall**: 0.9506
- **フィルタリング率**: 56.04%
- **負傷事故(Class 1) 通過率**: 43.5%

### Stage 2 Binary Classification (CV OOF)

**Best F1 閾値 (0.3214) での評価**:
| 指標 | 値 |
|------|----| 
| F1 Score | 0.1857 |
| Precision | 0.1435 |
| Recall | 0.2632 |

**Overall Metrics (全体に対する評価)**:
| 指標 | 値 |
|------|----| 
| Final Precision | 0.1435 |
| Final Recall | 0.2502 |
| Final F1 | 0.1824 |
| AUC | 0.8723 |

### テストセット評価 (Hold-Out 20%)

**CV最適閾値 (0.3214) を適用**:
| 指標 | 値 |
|------|----| 
| Precision | 0.3498 |
| Recall | 0.0240 |
| F1 | 0.0449 |
| AUC | 0.8913 |

**参考: Test Ideal F1**: 0.2507

## 考察

- 多クラス分類により、モデルは「無傷」「負傷」「死亡」の3段階の重大性を学習
- P(Injury+) スコアで「明らかに無害な事故」を除外することで、Precision向上の余地あり
- Binary分類と比較して、死亡事故の特定精度が向上しているか要検証
