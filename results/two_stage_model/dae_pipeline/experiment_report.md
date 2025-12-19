# DAE特徴量統合実験レポート

**実行日時**: 2025-12-19 07:12:44
**実行時間**: 42814.7秒

## パラメータ設定

| パラメータ | 値 |
|-----------|----| 
| Focal Alpha | 0.6321 |
| Focal Gamma | 1.1495 |
| DAE Bottleneck | 128 |
| DAE Epochs | 15 |
| DAE Swap Noise | 0.15 |
| Stage 1 Recall Target | 99% |
| Test Set Ratio | 20% |

## 結果サマリ

### Stage 1
- **閾値**: 0.0010
- **Recall**: 1.0000
- **フィルタリング率**: 0.00%

### Stage 2 (Focal Loss + DAE) - CV OOF評価

#### 固定閾値 (0.5) での評価
| 指標 | 値 |
|------|----| 
| Precision | 0.6071 |
| Recall | 0.0405 |
| F1 | 0.0759 |
| AUC | 0.8568 |

#### 動的閾値での評価 (CV OOF)
| Target Recall | Precision |
|---------------|----------|
| 99% | 0.0092 |
| 98% | 0.0117 |

### テストセット評価 (Hold-Out 20%)

| 指標 | 値 |
|------|----| 
| Precision | 0.6012 |
| Recall | 0.0320 |
| F1 | 0.0607 |
| AUC | 0.8981 |

#### 動的閾値での評価 (Test Set)
| Target Recall | Precision |
|---------------|----------|
| 99% | 0.0086 |
| 98% | 0.0086 |
| 95% | 0.0178 |

## 考察

- DAE特徴量 (128次元) により、LightGBMが苦手な非線形関係を捕捉
- Swap Noise (15%) によるノイズ除去効果
- CV OOF と Test Set の結果が近いほど、汎化性能が高い
