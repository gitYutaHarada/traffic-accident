# Optuna Focal Loss V2 実験レポート

## 実験概要
本実験では、Two-Stage ModelのStage 2 (LightGBM) におけるFocal Lossのハイパーパラメータおよびその他のモデルパラメータの最適化を行いました。
Focal Lossはクラス不均衡データにおける学習を改善し、特に重篤な事故（死亡事故など）の予測精度向上を目的としています。

- **実験日**: 2025/12/18
- **実験ID**: optuna_focal_loss_v2
- **試行回数**: 100回
- **評価指標**: Precision at Recall 0.99 (Recall 99%時のPrecision)

## 実験結果

### 最良スコア
- **Best Precision@Recall 0.99**: 0.1765 (Trial 93)

### 最適パラメータ (Best Params)
Optunaによって探索された最適パラメータは以下の通りです。

| パラメータ名 | 最適値 | 探索範囲・備考 |
| :--- | :--- | :--- |
| **focal_alpha** | 0.6321 | Focal Lossのalpha (正例の重み) |
| **focal_gamma** | 1.1495 | Focal Lossのgamma (難易度に応じた重み付け) |
| **num_leaves** | 127 | 決定木の葉の最大数 |
| **max_depth** | 6 | 木の深さの最大値 |
| **min_child_samples** | 44 | 葉に含まれる最小データ数 |
| **reg_alpha** | 2.3897 | L1正則化項 |
| **reg_lambda** | 2.2842 | L2正則化項 |
| **colsample_bytree** | 0.8646 | 列のサブサンプリング比率 |
| **subsample** | 0.6328 | 行のサブサンプリング比率 |
| **learning_rate** | 0.0477 | 学習率 |

## 考察
*   **Precisionの改善**: ベースラインや以前の実験と比較して、Recall 99%という高い再現率を維持しつつ、約17.65%のPrecisionを達成しました。
*   **Focal Lossパラメータ**: `focal_alpha` が約0.63、`focal_gamma` が約1.15という結果になりました。
    *   `alpha > 0.5` であることは、正例（死亡事故）に対して重み付けを強化していることを示唆します。
    *   `gamma` が1に近い値であることから、極端に難しいサンプルだけでなく、適度に難しいサンプルへの学習に重きを置いていると考えられます。
*   **モデルの複雑さ**: `max_depth` が6、`num_leaves` が127と、比較的深い木が選択されていますが、正則化パラメータ（`reg_alpha`, `reg_lambda`）もそれぞれ約2.3, 2.3と効いており、過学習を抑制しつつ複雑なパターンを学習しています。

## 今後のアクション
*   この最適化されたパラメータセットを用いて、最終的なTwo-Stage Model (Stage 2) の学習と評価を行います。
*   必要に応じて、検証データセット（Valid Set）だけでなくテストデータ（Test Set）での汎化性能を確認します。
