# 特徴量エンジニアリング強化 実装計画

FP分析で明らかになった「当事者間の関係性」や「道路環境の複合リスク」を捉えるため、相互作用特徴量（Interaction Features）を追加実装し、モデル性能への影響を検証します。

## 目的
Stage 2 (LightGBM) の特徴量を強化し、特に誤検知（FP）を削減してPrecisionを向上させること。

## 実装内容

### 1. 新規特徴量の作成
`scripts/features/create_interaction_features.py` を新規作成し、以下の特徴量を生成します。

#### A. 当事者間の関係性 (Party Interactions)
当事者A（自分）とB（相手）の属性の「差」や「組み合わせ」に着目します。

| 特徴量名 | 内容 | 仮説 |
| :--- | :--- | :--- |
| **reg_stop_interaction** | `一時停止規制（A）` × `一時停止規制（B）` | 相手側のみ一時停止規制がある場合の事故は、相手の過失度が高くリスクが異なる可能性。 |
| **reg_speed_diff** | `速度規制（A）` - `速度規制（B）` | 速度差が大きい交差点や合流地点でのリスクを表現。 |
| **party_type_combo** | `当事者種別（A）` × `当事者種別（B）` | 「四輪×歩行者」「四輪×二輪」など、衝突パターンのリスク差を明示化。 |
| **age_diff** | `年齢（A）` - `年齢（B）` | 年齢差による反応速度や行動予測のズレ。（例: 高齢者 vs 若年者） |

#### B. 道路環境の複合リスク (Environment Interactions)
単独では危険でなくても、組み合わさると危険な場所を特定します。

| 特徴量名 | 内容 | 仮説 |
| :--- | :--- | :--- |
| **road_shape_terrain** | `道路形状` × `地形` | 「下り坂」の「カーブ」や「交差点」など、物理的挙動が不安定になりやすい場所。 |
| **signal_road_shape** | `信号機` × `道路形状` | 「信号なし」の「十字路」など、出会い頭事故のリスクが高い場所。 |
| **night_road_condition** | `昼夜` × `路面状態` | 「夜間」の「凍結/濡れ」など、視認性と制動距離の悪条件が重なる場合。 |

### 2. データセット生成
- 入力: `data/processed/honhyo_clean.csv` (または欠損補完後のベースデータ)
- 処理: 上記の特徴量を追加。カテゴリ変数は `category` 型、またはLabel Encoding。
- 出力: `data/processed/honhyo_with_interactions.csv`

### 3. 検証実験 (Experiment)
- スクリプト: `experiments/train_stage2_interactions.py` (既存の学習スクリプトをベースに作成)
- 比較対象: 現在の Stage 2 ベストモデル
- 評価指標:
    - **Precision @ Recall 99%** (重点指標)
    - AUC, LogLoss
    - Feature Importance (新特徴量が上位に来るか)

## 作業手順

1.  **スクリプト作成**: `scripts/features/create_interaction_features.py`
2.  **データ生成**: スクリプトを実行し、拡張データセットを作成。
3.  **学習・評価**: 新データで LightGBM を学習 (CV 5-fold)。
4.  **結果確認**: 前回同様のレポートを作成し、新特徴量の寄与度を確認。

## ユーザーへの確認事項
- 特徴量生成において、`honhyo_clean.csv` をベースにして良いか、あるいは `honhyo_clean_with_features.csv` (既存特徴量追加済み) をベースにするか？
    - *推奨*: **`honhyo_clean.csv`** から再生成する方が、重複や依存関係を整理しやすいです。既存の特徴量（エンコーディング済みでないもの）が必要です。
