# Hard Negative Mining Report

## 概要
- **分析対象**: 予測確率上位 500 件の False Positives
- **予測確率範囲**: 0.6279 - 0.6279

## 1. Hard FP vs True Positive プロファイル比較

Hard FPとTPの特徴量比較により、誤検知されやすいパターンを特定。
- **Categorical**: 最頻値（Mode）とその割合を比較
- **Numeric**: 平均値を比較
- **Note**: `**Diff**` は最頻値が異なることを示す（注目ポイント）

| Feature | Type | Hard FP | TP | Note |
| :--- | :--- | :--- | :--- | :--- |
| month | Numeric | 5.69 | 6.83 | Diff: -1.15 |
| 路面状態 | Categorical | 1 (77.8%) | 1 (82.1%) | Same |
| 天候 | Categorical | 1 (58.6%) | 1 (65.7%) | Same |
| 地形 | Categorical | 1 (72.0%) | 3 (41.5%) | **Diff** |
| signal_road_shape | Categorical | 7_14 (43.8%) | 7_14 (31.2%) | Same |
| area_id | Categorical | 4 (13.2%) | 13 (8.2%) | **Diff** |
| road_shape_terrain | Categorical | 14_1 (30.8%) | 1_1 (16.6%) | **Diff** |
| 昼夜 | Categorical | 22 (75.0%) | 12 (45.0%) | **Diff** |
| 当事者種別（当事者A） | Categorical | 3 (57.2%) | 3 (32.3%) | Same |
| 年齢（当事者A） | Numeric | 43.52 | 45.75 | Diff: -2.23 |
| party_type_daytime | Categorical | 3_22 (45.0%) | 3_22 (13.0%) | Same |
| stop_sign_interaction | Categorical | 0 (58.8%) | 0 (65.2%) | Same |
| 信号機 | Categorical | 7 (65.8%) | 7 (80.0%) | Same |
| 速度規制（指定のみ）（当事者B） | Categorical | 0 (100.0%) | 0 (74.8%) | Same |
| road_type | Categorical | 13 (83.6%) | 13 (73.0%) | Same |
| 道路形状 | Categorical | 14 (44.2%) | 1 (34.8%) | **Diff** |
| 速度規制（指定のみ）（当事者A） | Categorical | 3 (28.8%) | 10 (32.0%) | **Diff** |
| hour | Numeric | 2.61 | 12.33 | Diff: -9.72 |
| speed_shape_interaction | Categorical | 3_1 (14.0%) | 3_1 (10.7%) | Same |
| night_road_condition | Categorical | 22_1 (58.8%) | 12_1 (38.6%) | **Diff** |
| night_terrain | Categorical | 22_1 (55.4%) | 12_3 (22.1%) | **Diff** |

## 2. クラスタリング結果

Hard FPをK-Meansで複数のグループに分類し、誤検知のパターンを特定。
- One-Hot Encodingを使用（Label Encodingの距離問題を回避）

### クラスタ特性
|   Cluster |   Count |   Avg_Prob |   道路形状 |   昼夜 |   天候 |   地形 |   当事者種別（当事者A） |   年齢（当事者A） |   speed_reg_diff_abs |
|----------:|--------:|-----------:|-------:|-----:|-----:|-----:|--------------:|-----------:|---------------------:|
|         0 |     254 |     0.6279 |     14 |   22 |    1 |    1 |             3 |      45.48 |                 4.96 |
|         1 |      91 |     0.6279 |     14 |   22 |    2 |    1 |             3 |      41.9  |                 4.93 |
|         2 |       8 |     0.6279 |      0 |   22 |    1 |    3 |             3 |      37.75 |                 9.12 |
|         3 |     131 |     0.6279 |     14 |   22 |    1 |    2 |             3 |      40.91 |                 5.48 |
|         4 |      16 |     0.6279 |      1 |   22 |    1 |    1 |            17 |      46    |                 4.5  |

### クラスタ可視化 (PCA)
![Clusters](hard_fp_clusters_pca.png)

## 3. 次のアクション
1. 各クラスタの代表事例を詳細に確認し、誤検知の原因を特定する。
2. 特定のクラスタに対応する新たな特徴量や、ルールベースのフィルタを検討する。
3. Hard FP に対する重み付け学習（Hard Negative Mining in training）を検討する。
