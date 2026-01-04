# Stage 2: 4モデルアンサンブル 実験レポート (v2)

**実行日時**: 2025-12-24 22:08:10
**実行時間**: 11395.7秒
**ハードウェア**: Intel Core Ultra 9 285K (n_jobs=8)

## パイプライン構成

- **Stage 1**: Weighted Ensemble (Recall 98%)
- **Stage 2**: LightGBM, CatBoost, TabNet (cat_idxs対応), MLP (Embedding層) → Weighted Ensemble

## Stage 1 フィルタリング結果

| 指標 | 値 |
|------|-----|
| 閾値 | 0.0645 |
| Train通過率 | 57.9% |
| Test通過率 | 58.3% |

## モデル比較

| Model | OOF ROC-AUC | Test ROC-AUC | OOF PR-AUC |
|-------|-------------|--------------|------------|
| lgbm | 0.8615 | 0.8657 | 0.1965 |
| catboost | 0.8609 | 0.8619 | 0.2056 |
| tabnet | 0.7986 | 0.8346 | 0.1173 |
| mlp | 0.8534 | 0.8634 | 0.1816 |
| ensemble | 0.8663 | 0.8684 | 0.2098 |

## アンサンブル重み

- **lgbm**: 0.4077
- **catboost**: 0.4317
- **tabnet**: 0.0000
- **mlp**: 0.1606

## 考察

- 最高単体モデル AUC: 0.8657
- アンサンブル AUC: 0.8684

## 修正点 (v2)

1. データ整合性: Stage 1 OOF予測と train_test_split の同期を確認
2. MLP: カテゴリ変数に Embedding 層を使用（誤った順序関係を排除）
3. TabNet: cat_idxs, cat_dims を正しく設定
4. Intel最適化: OMP/MKL スレッド設定を明示
5. PyTorch再現性: シード固定
