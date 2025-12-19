# Stage 2 Cleanlab Denoised 実験レポート

**実行日時**: 2025-12-19 14:16:21
**実行時間**: 104.2秒

## モデル構成
- **Stage 1**: Binary Classification (死亡 vs その他)
- **Stage 2**: Binary Classification (0=負傷, 1=死亡) - **Cleanlab Denoised**
- **ノイズ除外件数**: 79,491 件

## 結果サマリ
        
### Stage 1 (Recall 95%)
- **閾値**: 0.1120
- **Recall**: 0.9506
- **フィルタリング率**: 56.04%
- **負傷事故(Class 1) 通過率**: 43.5%

### Stage 2 Binary Classification (CV OOF) - Denoised

**Best F1 閾値 (0.2960) での評価**:
| 指標 | 値 |
|------|----| 
| F1 Score | 0.1912 |
| Precision | 0.1277 |
| Recall | 0.3806 |

**Overall Metrics (全体に対する評価)**:
| 指標 | 値 |
|------|----| 
| Final Precision | 0.1277 |
| Final Recall | 0.3618 |
| Final F1 | 0.1887 |
| AUC | 0.8799 |

### テストセット評価 (Hold-Out 20%)

**CV最適閾値 (0.2960) を適用**:
| 指標 | 値 |
|------|----| 
| Precision | 0.1559 |
| Recall | 0.3864 |
| F1 | 0.2221 |
| AUC | 0.8911 |

**参考: Test Ideal F1**: 0.2263

## 考察

- Cleanlabで検出された「Fatal Look-alikes」を学習データから除外
- これにより、モデルは「純粋な死亡パターン」のみを学習
- テストデータにはノイズが残っているため、現実世界での性能を正しく評価
