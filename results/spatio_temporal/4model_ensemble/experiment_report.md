# 4モデル比較 + アンサンブル 実験レポート (v2)

**実行日時**: 2025-12-24 18:48:13
**実行時間**: 1568.6秒
**データ**: honhyo_for_analysis_with_traffic_hospital_no_leakage.csv

## 修正点 (v2)
- **MLP**: Target Encoding使用（Label Encodingの順序性問題を解消）
- **MLP**: Train/Val分割後にエンコーダをfit（リーク防止）
- **アンサンブル**: LogLossベースで重み最適化（閾値依存を排除）

## OOF評価 (5-Fold CV)

| モデル | ROC-AUC |
|--------|---------|
| LightGBM | 0.5152 |
| CatBoost | 0.5163 |
| TabNet | 0.5123 |
| MLP | 0.5157 |
| **Ensemble** | **0.5159** |

## テストセット評価

| モデル | ROC-AUC |
|--------|---------|
| LightGBM | 0.8774 |
| CatBoost | 0.9025 |
| TabNet | 0.8103 |
| MLP | 0.8855 |
| **Ensemble** | **0.8956** |

## アンサンブル重み (LogLoss最適化)

| モデル | 重み |
|--------|------|
| LightGBM | 0.010 |
| CatBoost | 0.493 |
| TabNet | 0.487 |
| MLP | 0.010 |

## 考察

- **最優秀単体モデル**: CatBoost (0.9025)
- **アンサンブル効果**: -0.69% (単体最高比)
