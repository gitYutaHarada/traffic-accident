# Stage 3: Stacking 最終実験レポート

**日付:** 2025-12-28
**実験:** Two-Stage CatBoost + Single-Stage TabNet Stacking (Logistic Regression)
**ステータス:** ✅ **成功 (ID Propagation 修正済み)**

---

## 1. 結果概要

ID Propagation（IDベースの整合性確保）の実装により、Stackingが正常に機能しました。

| 指標 | Score | 備考 |
| :--- | :--- | :--- |
| **OOF AUC** | 0.8809 | |
| **Test AUC** | **0.9030** | Single-Stage TabNet (0.9045) に迫る高精度 |
| **Test PR-AUC** | **0.1760** | |

### 実行時間
| Step | 処理内容 | 時間 |
| :--- | :--- | :--- |
| Step 1 | Single-Stage OOF再生成 | 0.3分 |
| Step 2 | Two-Stage OOF再生成 | 0.1分 |
| Step 3 | Stacking Meta-Model | 0.1分 |
| **合計** | | **0.6分** |

---

## 2. メタモデルの重み（係数）分析

前回の失敗（CatBoost係数がマイナス）から劇的に改善しました。

| 特徴量 | 係数 (Coefficient) | 解釈 |
| :--- | :--- | :--- |
| **tabnet_prob** | **+0.4549** | **支配的要因**。Single-Stage TabNetが最も信頼されている。 |
| **catboost_prob** | **+0.1558** | ✅ **正の寄与**。ID修正により、Two-Stageモデルが正しく評価されている。 |
| **stage1_prob** | +0.0190 | 補助的な寄与。 |
| **is_easy_sample** | +0.0116 | Easy Sample (Two-Stage未実施) かどうかのフラグも有効。 |
| **tabnet_x_catboost** | -0.1090 | 交互作用項は負（モデル間の相関補正と思われる）。 |

> **改善ポイント**: ID Propagation修正前は `catboost_prob` が `-0.0135` でしたが、修正後は `+0.1558` となり、明確に予測に貢献しています。

---

## 3. 成功要因の分析

1.  **ID Propagation (ID伝播)**:
    *   Single-Stage, Two-Stageの全工程で `original_index` を徹底して保持・伝播させました。
    *   Stacking時に行番号ではなく `original_index` をキーにマージすることで、87万行 vs 32万行のような行ズレ問題を完全に解消しました。

2.  **パフォーマンス・安定性最適化**:
    *   **TabNet高速化**: `dict.map()` によるベクトル化でエンコーディングを高速化。
    *   **E-core活用**: `DataLoader(num_workers=16)` により、データ読み込みをバックグラウンド化。
    *   **NaN対策**: 推論時のNaNフォールバックとバッチサイズ調整により、安定学習を実現。

## 4. 結論

*   **Stacking成功**: 異なる性質を持つモデル（GBDTベースのTwo-Stage, NNベースのSingle-Stage）を効果的に統合できました。
*   **最終モデル**: `results/stage3_stacking/final_submission_stacking.csv` を最終提出物として推奨します。
