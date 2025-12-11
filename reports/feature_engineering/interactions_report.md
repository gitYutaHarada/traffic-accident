# 実験レポート: クロス特徴量（Interaction Features）の追加検証

## 実験概要
- **目的**: ドメイン知識に基づく5つのクロス特徴量を追加し、精度向上を確認する。
- **ベースモデル**: LightGBM (Area ID n=50)
- **追加変数**:
    1. `interaction_type_speed` (車種 × 速度規制)
    2. `interaction_age_hour` (年齢 × 時間帯)
    3. `interaction_terrain_shape` (地形 × 道路形状)
    4. `interaction_area_daynight` (エリアID × 昼夜)
    5. `interaction_weather_road` (天候 × 路面状態)

## 📊 実験結果比較

| モデル設定 | AUC (平均) | Recall (平均) |
| :--- | :--- | :--- |
| ベースライン (n=50) | **0.8876** | **0.7567** |
| **+ クロス特徴量** | 0.8871 📉 | 0.7381 📉 |

### Feature Importance (上位抜粋)
追加した特徴量はモデルに強く利用されています。
- **2位**: `interaction_type_speed` (車種 × 速度) 
- **5位**: `interaction_age_hour` (年齢 × 時間)
- **6位**: `interaction_terrain_shape` (地形 × 形状)

### 考察
1. **重要だが精度には直結せず**: 追加した変数は重要度上位に入っており、モデルにとっては「使いやすい変数」であったことが分かります。特に「車種×速度（例：高速道路×二輪車）」は非常に強力な予測因子です。
2. **Recallの低下**: しかし、結果としてRecall（死亡事故の検知率）が約1.8ポイント低下しました。変数が複雑化したことで、かえって過学習気味になったか、単純なパターンの検出が阻害された可能性があります。
3. **結論**: 純粋な予測精度（スコア）を追求する場合、今回のクロス特徴量は（少なくとも全てを単純に追加するのは）逆効果の可能性があります。Feature Importanceが高いもの（車種×速度など）だけを厳選して残すなどの調整が必要です。

## 📂 生成ファイル
- `results/experiments/interactions/metrics_mean.csv`
- `results/experiments/interactions/feature_importance.png`
