# Two-Stage Spatio-Temporal 実験レポート

**日付:** 2025-12-28
**実験:** Two-Stage Spatio-Temporal 4-Model Ensemble

---

## 1. 実験概要

本実験では、**Two-Stage 構成**（Stage 1 で易しい負例を除外 → Stage 2 で時空間特徴量を用いて詳細予測）を採用し、Spatio-Temporalモデルの性能を最大化することを目的としました。
前回の実験（Single-Stage Spatio-Temporal: 全データを一度に学習）と比較し、**「難易度の高い事故（Hard Samples）」に対する予測精度**がどのように変化したかを検証します。

*   **Stage 1**: 既存のLightGBM/CatBoostアンサンブルにより、再現率(Recall) 98% を維持しつつ、明らかに安全なデータ（約42%）を除外。
*   **Stage 2**: 残りの「Hard Samples（Train: 32万件, Test: 14万件, Fatal率: 0.85%）」に対し、Spatio-Temporal特徴量を用いた4モデル（LightGBM, CatBoost, MLP, TabNet）を学習。

---

## 2. 実験結果 (Stage 2 Hard Samples)

Stage 1通過データ（フィルタ済みデータ）に対するOOFおよびTest予測性能です。

| モデル | OOF AUC | OOF PR-AUC | Test AUC | Test PR-AUC |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | **0.8950** | 0.1530 | **0.8937** | 0.1581 |
| **CatBoost** | 0.8928 | **0.1657** | 0.8869 | **0.1754** |
| **Ensemble** | 0.8869 | 0.1642 | 0.8655 | 0.1611 |
| **MLP** | 0.8752 | 0.1314 | 0.8535 | 0.1379 |
| **TabNet** | 0.8696 | 0.1248 | 0.8867 | 0.1567 |

### 特記事項
1.  **GBDTの強さ**: LightGBMとCatBoostが高精度を記録しました。特にCatBoostはPR-AUC（適合率-再現率）でトップとなり、不均衡なHard Samplesの識別に成功しています。
2.  **アンサンブルの不振**: Ensembleスコアが単体モデル（CatBoost/LightGBM）を下回りました。これは、性能の低いMLP/TabNetに対し、最適化の結果として**等重み（各0.25）**が割り当てられてしまった（制約条件の影響の可能性）ことが原因と考えられます。
3.  **NNの苦戦**: MLPとTabNetはGBDTに劣る結果となりました。データ量がフィルタリングで減少（150万件→32万件）したため、Deep Learningモデルの学習が十分に進まなかった可能性があります。

---

## 3. Baseline（Single-Stage）との比較

前回の実験結果（Single-Stage Spatio-Temporal）と比較します。
※ 比較対象は、同じ評価指標（PR-AUC）を用いていると考えられる「Hard Samples」相当のスコアです。

| モデル | Single-Stage PR-AUC<br>(前回) | **Two-Stage** PR-AUC<br>(今回) | 変化率 | 考察 |
| :--- | :--- | :--- | :--- | :--- |
| **LightGBM** | 0.1349 | **0.1530** | **+13.4%** | Two-Stage化により大幅改善 |
| **CatBoost** | 0.1568 | **0.1657** | **+5.7%** | 安定して改善 |
| **TabNet** | **0.1575** | 0.1248 | ▼-20.7% | データ量減少により悪化 |
| **Ensemble** | **0.1729** | 0.1642 | ▼-5.0% | 弱いモデルの混入により悪化 |

### 考察: どちらのアプローチが良いか？

*   **GBDT (LightGBM/CatBoost) にとっては Two-Stage が有利**: 
    「易しい負例」が除外され、決定木が「難しい境界線」の学習に集中できたため、精度が向上しました。
*   **TabNet にとっては Single-Stage が有利**:
    Deep Learningモデルは大量のデータを必要とします。Stage 1でのデータ削減が、逆にモデルの汎化性能を落とす結果となりました（Single-Stage時はFull Dataで学習できたため強かった）。

---

## 4. 結論と推奨

1.  **採用手法の使い分け**:
    *   **GBDT系**は、今回の **Two-Stage Spatio-Temporal** モデルを採用すべきです。
    *   **TabNet**は、前回の **Single-Stage Spatio-Temporal** モデルを採用すべきです。

2.  **アンサンブル戦略の修正**:
    *   現在の「4モデル等重み」は最適ではありません。
    *   **Strong Learners (Two-Stage GBDTs)** と **Different View (Single-Stage TabNet)** を組み合わせる「Heterogeneous Ensemble」が最強の構成になる可能性があります。

3.  **提出データの復元**:
    *   今回作成した `final_submission_full.csv` は全テストデータをカバーしていますが、Stage 1除外データ（確率0）とStage 2予測データの結合部分での整合性検証が必要です（現在、全体AUCの算出で課題が残っています）。
    *   提出用としては、Stage 1のスコアをそのまま使用する（確率0にするのではなく、Stage 1の確率を使用する）方が、ランキング性能（AUC）を損なわず安全です。

### Next Step
*   Stage 3 (Stacking) では、**Two-Stage CatBoost** と **Single-Stage TabNet** のOOF予測値をメタ特徴量として入力することを強く推奨します。
