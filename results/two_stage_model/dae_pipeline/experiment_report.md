# DAE特徴量統合実験レポート

**実行日時**: 2025-12-19 17:18:46
**実行時間**: 21833.9秒

## パラメータ設定

| パラメータ | 値 |
|-----------|----| 
| Focal Alpha | 0.6321 |
| Focal Gamma | 1.1495 |
| DAE Bottleneck | 128 |
| DAE Epochs | 15 |
| DAE Swap Noise | 0.15 |
| Stage 1 Recall Target | 95% |
| Test Set Ratio | 20% |

## 結果サマリ

### Stage 1
- **閾値**: 0.1120
- **Recall**: 0.9506
- **フィルタリング率**: 56.04%

### Stage 2 (Focal Loss + DAE) - CV OOF評価

#### 固定閾値 (0.5) での評価
| 指標 | 値 |
|------|----| 
| Precision | 0.6150 |
| Recall | 0.0384 |
| F1 | 0.0723 |
| AUC | 0.8407 |

#### 動的閾値での評価 (CV OOF)
| Target Recall | Precision |
|---------------|----------|
| 99% | 0.0195 |
| 98% | 0.0202 |

### テストセット評価 (Hold-Out 20%)

| 指標 | 値 |
|------|----| 
| Precision | 0.6645 |
| Recall | 0.0310 |
| F1 | 0.0593 |
| AUC | 0.8965 |

#### 動的閾値での評価 (Test Set)
| Target Recall | Precision |
|---------------|----------|
| 99% | 0.0186 |
| 98% | 0.0186 |
| 95% | 0.0234 |

## 考察

- DAE特徴量 (128次元) により、LightGBMが苦手な非線形関係を捕捉
- Swap Noise (15%) によるノイズ除去効果
- CV OOF と Test Set の結果が近いほど、汎化性能が高い
