# Implementation Plan: SHAP詳細分析 & Hard Negative Mining

## Goal
LightGBMモデルが獲得した「相互作用特徴量」の解釈性を深め（Phase 1）、依然として残る「手強い誤検知（Hard False Positives）」の原因を特定する（Phase 2）。これにより、モデルの信頼性確認と次なる改善施策（Phase 5）への橋渡しを行う。

## Phase 1: SHAP Interaction Analysis (リスクの向き解明)
モデルが `party_type_daytime` などを重要視していることは分かったが、「どのような組み合わせを危険と判断しているか」を確認する。

### 1. Script: `scripts/analysis/shap_interaction_detail.py`
*   **Input**:
    *   学習済みモデル（Pickle等で保存されていない場合は、再学習してSHAP値を生成するオプションを用意）
    *   `honhyo_with_interactions.csv`
*   **Process**:
    1.  学習済みモデルをロード（またはFeature Engineering含めパイプライン再実行）。
    2.  `party_type_daytime`, `road_shape_terrain` 等の重要カテゴリ変数を対象に、各カテゴリ値ごとの **平均SHAP値** を計算。
    3.  **Risk Matrix Heatmap**:
        *   元の構成要素（例: 縦軸=当事者種別, 横軸=昼夜）に戻してマトリックスを作成。
        *   セルの色でSHAP値（リスク寄与度）を可視化。
*   **Output**:
    *   `results/analysis/shap_detail/risk_matrix_{feature_name}.png`
    *   `results/analysis/shap_detail/high_risk_groups.csv`（SHAP値が高い組み合わせランキング）

## Phase 2: Hard Negative Mining (Hard FP 分析)
新モデルでも検知確率は高いが、実際には死亡事故ではなかったケース（Hard FP）を分析する。

### 1. Script: `scripts/analysis/analyze_hard_negatives.py`
*   **Input**:
    *   `results/experiments/interaction_features/fp_new_model.csv` (前回の実験出力)
*   **Process**:
    1.  **Selection**: 予測確率（Probability）が高い上位 500件 を抽出。
    2.  **Profiling**:
        *   単変量解析: Hard FPグループで頻出する特徴（この場所、この時間帯、この違反など）を集計。
        *   TP（True Positive）との比較: 「本当に危険だったケース」と「危険に見えて無事だったケース」の微細な違い（例えば、微小な速度差、当事者の年齢層のズレなど）を探る。
    3.  **Clustering**:
        *   Hard FPのみを対象にK-Meansクラスタリングを行い、「誤検知のパターン（例：雨天スリップ型誤検知、右折時見落とし型誤検知）」を分類する。
*   **Output**:
    *   `results/analysis/hard_negatives/hard_fp_profile.md`
    *   `results/analysis/hard_negatives/hard_fp_clusters.csv`

## Schedule
1. **Phase 1 Implementation**: `shap_interaction_detail.py` 作成 & 実行
2. **Review**: Risk Matrixを見て、直感に反する挙動がないか確認。
3. **Phase 2 Implementation**: `analyze_hard_negatives.py` 作成 & 実行
4. **Summary**: 分析結果を統合し、今後のロードマップ（モデルアンサンブルやルールの追加）を策定。
