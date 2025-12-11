# 交互作用特徴量分析

このディレクトリには、交互作用特徴量の生成・評価・レポート作成を行うスクリプトが含まれています。

## 📁 ファイル構成

| ファイル | 説明 |
|---------|------|
| `generate_interaction_features.py` | すべての2つの特徴量の組み合わせで交互作用特徴量を生成 |
| `evaluate_interaction_importance.py` | LightGBMで各交互作用特徴量の重要度を評価 |
| `generate_ranking_report.py` | 評価結果からランキングレポートと可視化を生成 |
| `run_interaction_analysis.py` | 上記3つのスクリプトを統合した実行パイプライン |

## 🚀 使い方

### 方法1: 統合パイプラインで実行（推奨）

すべてのステップをワンコマンドで実行:

```bash
python scripts/feature_engineering/run_interaction_analysis.py
```

### 方法2: 個別に実行

#### ステップ1: 交互作用特徴量の生成

```bash
python scripts/feature_engineering/generate_interaction_features.py
```

**出力**:
- `data/interaction_features_YYYYMMDD_HHMMSS/`: 各交互作用特徴量のpickleファイル
- `data/interaction_features_YYYYMMDD_HHMMSS/interaction_features_metadata.csv`: メタデータ

**所要時間**: 約5-10分

#### ステップ2: LightGBMで重要度評価

```bash
python scripts/feature_engineering/evaluate_interaction_importance.py
```

⚠️ **注意**: スクリプト内の`INTERACTION_DIR`を、ステップ1で生成されたディレクトリに変更してください。

**所要時間**: 約10-50時間（528通りの組み合わせ × 5-fold CV）

**出力**:
- `results/interaction_features/interaction_features_ranking_full_YYYYMMDD_HHMMSS.csv`: 全ランキング
- `results/interaction_features/interaction_features_ranking_top100_YYYYMMDD_HHMMSS.csv`: Top 100

#### ステップ3: ランキングレポート生成

```bash
python scripts/feature_engineering/generate_ranking_report.py
```

⚠️ **注意**: スクリプト内の`RANKING_CSV`を、ステップ2で生成されたCSVパスに変更してください。

**出力**:
- `results/interaction_features/interaction_features_ranking_report.md`: Markdownレポート
- `results/interaction_features/top20_bar_chart.png`: Top 20の棒グラフ
- `results/interaction_features/top30_heatmap.png`: Top 30のヒートマップ
- `results/interaction_features/distribution_plots.png`: 分布プロット

## 📊 評価手法

### LightGBMによる重要度評価

各交互作用特徴量を既存の33特徴量に1つずつ追加し、以下の方法で評価:

1. **ベースラインモデル**（交互作用特徴量なし）のPR-AUCを測定
2. 各交互作用特徴量を追加してPR-AUCを測定（5-fold Stratified CV）
3. **Delta PR-AUC**（向上度）でランキング

### 使用パラメータ

LightGBMチューニング最終レポートの最良パラメータを使用:
- learning_rate: 0.0766
- num_leaves: 125
- max_depth: 8
- min_child_samples: 278
- その他、チューニング済みの全パラメータ

## 📈 出力レポート

### ランキングCSV

| カラム | 説明 |
|--------|------|
| `rank` | ランク（1位から） |
| `feature_name` | 交互作用特徴量名 |
| `feature1`, `feature2` | 組み合わせた特徴量 |
| `interaction_type` | タイプ（cat_x_cat, num_x_num, num_x_cat） |
| `delta_pr_auc` | PR-AUCの向上度 |
| `pr_auc` | 交互作用特徴量追加後のPR-AUC |
| `delta_roc_auc`, `delta_f1` | ROC-AUCとF1の向上度 |

### Markdownレポート

- エグゼクティブサマリー
- Top 100 ランキング表
- 交互作用タイプ別の統計
- Top 10 の詳細分析
- 推奨事項

### 可視化

1. **棒グラフ**: Top 20の重要度を視覚化
2. **ヒートマップ**: Top 30の評価指標を比較
3. **分布プロット**: Delta PR-AUCの分布、タイプ別の平均、累積分布など

## ⚙️ カスタマイズ

### データパスの変更

各スクリプトの`main()`関数内で以下を変更:

```python
DATA_PATH = 'あなたのデータパス'
TARGET_COLUMN = '目的変数のカラム名'
```

### パラメータの調整

`evaluate_interaction_importance.py`で:

```python
self.n_folds = 5  # 交差検証のフォールド数
self.best_params = {...}  # LightGBMパラメータ
```

## 💡 Tips

### 計算時間の短縮

1. **サンプリング**: データの10-20%でスクリーニング
   ```python
   # generate_interaction_features.py で
   self.df = pd.read_csv(data_path).sample(frac=0.1, random_state=42)
   ```

2. **並列実行**: 複数のGPU/CPUで並列評価（実装が必要）

3. **フォールド数削減**: 5-fold → 3-fold
   ```python
   n_folds=3
   ```

### メモリ不足の場合

交互作用特徴量を個別のpickleファイルとして保存しているため、メモリ効率的です。もしメモリ不足が発生する場合は:

1. バッチサイズを小さくする
2. データ型を最適化（int64 → int32など）

## 📝 次のステップ

1. レポートを確認して、Top 10の交互作用特徴量を特定
2. 元のデータセットにTop 10を追加
3. LightGBMモデルを再訓練
4. PR-AUCの向上を確認

## ⚠️ 注意事項

- **実行時間**: 全ステップで10-50時間かかる可能性があります
- **ディスク容量**: 交互作用特徴量のpickleファイルで数GB必要
- **データリーク**: 交互作用特徴量生成時に未来の情報を使わないよう注意
- **パス設定**: 各スクリプトで出力ディレクトリのパスを正しく設定してください

## 🐛 トラブルシューティング

### エラー: "ModuleNotFoundError"

```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn tqdm
```

### エラー: "FileNotFoundError"

スクリプト内の`DATA_PATH`、`INTERACTION_DIR`、`METADATA_PATH`のパスが正しいか確認してください。

### 評価が遅すぎる

- データをサンプリングする
- フォールド数を減らす（5 → 3）
- early_stoppingのroundsを少なくする（50 → 20）

---

**作成日**: 2025年12月11日  
**作成者**: Antigravity AI Agent
