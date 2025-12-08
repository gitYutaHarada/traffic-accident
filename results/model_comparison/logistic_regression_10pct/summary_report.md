# ロジスティック回帰 実験結果 (10.0% サンプル)

**実験日時:** 2025年12月08日 14:34:48
**目的:** LightGBMとの比較のためのベースラインモデル
**サンプリング率:** 10.0%

---

## 📊 実験概要

### データセット
- ファイル: `data/processed/honhyo_model_ready.csv`
- 元データ数: 1,895,275 件
- 使用データ数: 189,528 件 (10.0%)
- Positive(死亡事故): 1,627 件
- Negative(非死亡): 187,901 件
- 不均衡比: 115.49:1

### 特徴量
- カテゴリカル変数: 22 カラム
- 数値変数: 26 カラム
- 総特徴量数: 48 (One-Hot Encoding後は増加)
- 高カーディナリティ処理: 上位50カテゴリ以外を'その他'に統合

### モデル設定
```python
LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='saga',
    max_iter=500,
    class_weight='balanced',  # クラス不均衡対策
    verbose=1,
    random_state=42
)
```

---

## 📈 評価結果

### 5-fold CV 平均スコア (Threshold 0.5)
| 指標 | スコア |
|------|--------|
| **Accuracy** | 0.8998 |
| **Precision** | 0.0816 |
| **Recall** | 0.9220 |
| **F1 Score** | 0.1487 |
| **AUC** | **0.9577** |

### 最適閾値の探索結果
| 設定 | 閾値 | Recall | Precision | F1 Score |
|------|------|--------|-----------|----------|
| **Max F1** | 1.0000 | 0.4474 | 0.2879 | 0.3503 |
| **Recall≥0.8** | 1.0000 | 0.8002 | 0.1413 | - |

---

## ⏱️ 実行時間

| 項目 | 時間 |
|------|------|
| **合計実行時間** | 0:15:57 |
| **交差検証時間** | 0:15:51 |
| **平均学習時間(1 Fold)** | 0:03:09 |
| **平均予測時間(1 Fold)** | 0:00:00 |

### 全データでの推定時間
- 現在のサンプル率: **10.0%**
- 実測時間: **0:15:57**
- 推定時間(線形スケーリング): **2:39:33**

---

## 💡 考察

### 前処理の違い
- **カテゴリカル変数**: One-Hot Encodingを使用(LightGBMはcategory型を直接扱える)
- **数値変数**: StandardScalerで標準化(LightGBMは不要)
- **欠損値**: SimpleImputerで補完(LightGBMは欠損値をそのまま扱える)

### モデルの特徴
- **線形モデル**: 特徴量間の複雑な相互作用を捉えにくい
- **解釈性**: 係数(Coefficients)から各特徴量の影響を直接読み取れる
- **計算コスト**: LightGBMより学習時間が短い(ただしOne-Hot Encodingで特徴量数が増加)

---

## 📁 出力ファイル
- [PR曲線](results/model_comparison/logistic_regression_10pct/pr_curve.png)
- [混同行列](results/model_comparison/logistic_regression_10pct/confusion_matrix.png)
- [評価指標CSV](results/model_comparison/logistic_regression_10pct/metrics.csv)
- [AUCスコア](results/model_comparison/logistic_regression_10pct/auc_score.txt)