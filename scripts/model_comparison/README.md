# ロジスティック回帰 vs LightGBM 比較分析

このディレクトリには、ロジスティック回帰とLightGBMを公平に比較するためのスクリプトが含まれています。

## 📁 ファイル構成

| ファイル | 説明 |
|---------|------|
| `train_logistic_regression_updated.py` | ロジスティック回帰の訓練と評価（LightGBMと同じデータセット） |
| `compare_models.py` | 両モデルの統合比較（同じfold分割で公平に比較） |
| `visualize_comparison.py` | 比較結果の可視化 |
| `run_full_comparison.py` | 上記3つをワンコマンドで実行 |
| `train_logistic_regression.py` | 旧版（参考用） |

## 🚀 使い方

### 方法1: 統合実行（推奨）

すべてのステップをワンコマンドで実行:

```bash
python scripts/model_comparison/run_full_comparison.py
```

これで以下が自動実行されます:
1. ロジスティック回帰の訓練と評価
2. LightGBMとの統合比較
3. 可視化の生成

### 方法2: 個別実行

#### ステップ1: ロジスティック回帰のみ訓練

```bash
python scripts/model_comparison/train_logistic_regression_updated.py
```

**出力**:
- `results/model_comparison/logistic_regression_updated/metrics_*.csv`
- `results/model_comparison/logistic_regression_updated/pr_curve_*.png`
- `results/model_comparison/logistic_regression_updated/roc_curve_*.png`
- `results/model_comparison/logistic_regression_updated/summary_report_*.md`

#### ステップ2: 統合比較

```bash
python scripts/model_comparison/compare_models.py
```

**出力**:
- `results/model_comparison/logreg_cv_results_*.csv`
- `results/model_comparison/lightgbm_cv_results_*.csv`
- `results/model_comparison/statistical_test_*.csv`
- `results/model_comparison/comparison_report_*.md`（★重要）

#### ステップ3: 可視化

```bash
python scripts/model_comparison/visualize_comparison.py
```

**出力**:
- `results/model_comparison/visualizations/metrics_comparison.png`
- `results/model_comparison/visualizations/box_plots.png`
- `results/model_comparison/visualizations/time_comparison.png`

---

## 📊 評価内容

### 比較される指標

| カテゴリ | 指標 |
|---------|------|
| **主要指標** | PR-AUC, ROC-AUC, F1 Score |
| **その他** | Accuracy, Precision, Recall |
| **計算コスト** | 訓練時間、予測時間 |
| **統計的検定** | Paired t-test（p値） |

### データセット

- **ファイル**: `data/processed/honhyo_clean_predictable_only.csv`
- **レコード数**: 1,895,275件
- **特徴量数**: 33
- **クラス不均衡比**: 115.51:1

### 評価方法

- **5-fold Stratified Cross-Validation**
- 両モデルで同じfold分割を使用（公平な比較）
- random_state=42で再現性を確保

---

## 📈 期待される結果

既存の比較結果（`lightgbm_vs_logreg.md`）から予測:

| 指標 | ロジスティック回帰 | LightGBM | 改善率 |
|------|------------------|----------|--------|
| **PR-AUC** | 0.0566 | **0.2056** | **+263%** |
| **ROC-AUC** | 0.8139 | **0.8879** | +9% |
| **F1 Score** | 0.1202 | **0.2883** | +140% |

**結論**: LightGBMが大幅に優位

---

## 🔧 カスタマイズ

### データセットの変更

各スクリプトのデフォルト値を変更:

```python
# train_logistic_regression_updated.py
model = LogisticRegressionModel(
    data_path='あなたのデータパス',  # ここを変更
    target_column='死者数',
    n_folds=5
)
```

### フォールド数の変更

```python
n_folds=3  # 5から3に変更（高速化）
```

### ロジスティック回帰のパラメータ調整

```python
LogisticRegression(
    penalty='l2',
    C=0.1,  # 正則化強度を変更
    solver='saga',
    max_iter=2000,  # 最大イテレーション数を増やす
    class_weight='balanced'
)
```

---

## 📝 出力レポートの見方

### 比較レポート（comparison_report_*.md）

#### エグゼクティブサマリー
- 両モデルのPR-AUCを比較
- 推奨モデルと理由

#### 主要指標の比較
- すべての評価指標を表形式で比較
- 差分と改善率を表示

#### 統計的検定
- Paired t-testの結果
- p値が0.05未満なら統計的に有意

#### 考察
- LightGBMの優位性
- ロジスティック回帰の特徴
- 推奨モデルの選択理由

---

## 💡 Tips

### 実行時間の短縮

1. **フォールド数を減らす**: 5-fold → 3-fold
2. **データをサンプリング**: 全データの10-20%で先にテスト

### メモリ不足の場合

- LightGBMの`n_jobs`を調整: `-1` → `4`など
- バッチサイズを小さくする

### エラー対処

#### "FileNotFoundError: データセットが見つかりません"
→ `data/processed/honhyo_clean_predictable_only.csv`が存在するか確認

#### "ImportError: sklearn がインストールされていません"
```bash
pip install scikit-learn lightgbm scipy matplotlib seaborn pandas tqdm
```

---

## 🎯 次のステップ

実行後:

1. **比較レポートを確認**
   - `results/model_comparison/comparison_report_*.md`を開く
   
2. **可視化を確認**
   - `results/model_comparison/visualizations/`内の画像を確認
   
3. **モデル選択を決定**
   - PR-AUCが最も重要な場合 → LightGBM
   - 解釈性が重要な場合 → ロジスティック回帰（ただし性能は低い）
   
4. **さらなる改善**
   - 選択したモデルに交互作用特徴量を追加
   - ハイパーパラメータの再調整
   - アンサンブルモデルの検討

---

## ⚠️ 注意事項

- **実行時間**: 全ステップで30-60分程度かかります
- **データセット**: 必ずLightGBMと同じデータセットを使用してください
- **再現性**: random_state=42で固定されていますが、環境によって微小な差異が出る可能性があります

---

**作成日**: 2025年12月11日  
**作成者**: Antigravity AI Agent  
**バージョン**: 1.0
