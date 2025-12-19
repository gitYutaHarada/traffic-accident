# TabNet Stage 2 実験レポート

**実行日時**: 2025-12-19 13:00頃
**モデル**: TabNet (n_d=16, n_a=16, n_steps=5, Batch=4096)

## モデル構成
- **Stage 1**: LightGBM Binary Classification (死亡 vs その他)
- **Stage 2**: TabNet Binary Classification (負傷 vs 死亡)

## 結果サマリ

### Stage 1 (Recall 95%)
| 指標 | 値 |
|------|----| 
| 閾値 | 0.1120 |
| Recall | 95.06% |
| フィルタリング率 | 56.04% |
| 残存データ | 666,475件 |
| 死亡事例残存 | 12,371 / 13,014 (95.1%) |

### Stage 2 TabNet (5-Fold CV) 

#### Fold別結果
| Fold | Best Epoch | Val AUC |
|------|------------|---------|
| 1 | 3 | 0.8426 |
| 2 | 4 | 0.8395 |
| 3 | 23 | 0.8404 |
| 4 | 3 | 0.8385 |
| 5 | 1 | 0.8379 |
| **平均** | - | **0.8398** |

#### OOF評価 (閾値0.5)
| 指標 | 値 |
|------|----| 
| Accuracy | 0.8033 |
| Fatal AUC | 0.8393 |
| Precision | 0.0660 |
| Recall | 0.7297 |

#### Best F1 閾値 (0.8768) での評価
| 指標 | 値 |
|------|----| 
| F1 Score | 0.2493 |
| Precision | 0.2090 |
| Recall | 0.3090 |

**Confusion Matrix (Best F1閾値)**:
```
              Pred 0    Pred 1
Actual 0     639,632    14,472
Actual 1       8,548     3,823
```

#### 全体評価 (Best F1閾値適用)
| 指標 | 値 |
|------|----| 
| Final Precision | 0.2090 |
| Final Recall | 0.2938 |
| Final F1 | 0.2442 |
| AUC | 0.8981 |

#### Recall重視の評価
| Target Recall | 閾値 | Precision |
|---------------|------|-----------|
| 99% | 0.1195 | 0.0199 |
| 98% | 0.1368 | 0.0213 |
| 95% | 0.1833 | 0.0258 |

### テストセット評価 (Hold-Out 20%)
- Stage 1 フィルタリング後: 166,112 / 379,055
- 死亡事例残存: 3,083 / 3,253 (94.8%)
- **注意**: Stage 2予測でCUDAエラーが発生し、完全な評価は未完了

## 考察

### 良い点
- **AUC 0.84**: 決定木モデルと同等レベルの識別能力を達成
- **早期収束**: 多くのFoldでepoch 1-4でBest AUCを達成（過学習しやすい）
- **効率的なフィルタリング**: Stage 1で56%のデータを除外しながら95%のRecallを維持

### 課題
- **過学習の兆候**: 学習が進むとval_aucが低下する傾向（特にFold 4, 5）
- **低いPrecision**: Best F1閾値でもPrecision 20.9%（LightGBMと比較が必要）
- **CUDAエラー**: テストセット評価時にカテゴリインデックス問題が発生

### LightGBM Stage 2との比較が必要
- TabNet AUC: 0.8393 (Stage 2 OOF)
- 比較対象: `train_stage2_multiclass.py` の結果と比較推奨

## 技術的な問題点
テストセット評価時にCUDAエラー（`scatter gather kernel index out of bounds`）が発生。
原因: テストデータに訓練データにないカテゴリ値が存在し、Embedding層の次元を超えた可能性。

