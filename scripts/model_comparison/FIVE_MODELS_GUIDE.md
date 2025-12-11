# 5モデル比較実行ガイド

## 📦 作成したスクリプト

**compare_five_models.py** - 5モデル統合比較スクリプト

### 比較対象モデル

1. **ロジスティック回帰**（線形モデル・ベースライン）
2. **Random Forest**（バギング手法）
3. **LightGBM**（ブースティング・Leaf-wise）
4. **XGBoost**（ブースティング・Depth-wise）✨ NEW
5. **CatBoost**（ブースティング・カテゴリ特化）✨ NEW

---

## 🚀 実行前の準備

### ステップ1: 必要なライブラリのインストール確認

```bash
# XGBoostとCatBoostをインストール（未インストールの場合）
pip install xgboost catboost
```

**確認コマンド**:
```bash
python -c "import xgboost; import catboost; print('OK')"
```

→ `OK`と表示されればインストール済み

### ステップ2: インストール済みライブラリの確認

```bash
pip list | findstr -i "xgboost catboost lightgbm"
```

---

## 🎯 実行方法

### 基本実行

```bash
cd c:\Users\socce\software-lab\traffic-accident
python scripts/model_comparison/compare_five_models.py
```

### 推定実行時間

| モデル | Fold 1回あたり | 合計（5 folds） |
|--------|---------------|----------------|
| ロジスティック回帰 | 10-15分 | 50-75分 |
| Random Forest | 15-20分 | 75-100分 |
| LightGBM | 3-5分 | 15-25分 |
| XGBoost | 5-8分 | 25-40分 |
| CatBoost | 8-12分 | 40-60分 |
| **合計** | **41-60分** | **205-300分** |

**総推定時間**: **約3.5-5時間**

---

## 📊 出力ファイル

すべて `results/model_comparison/` に保存されます：

- `logreg_cv_5models_YYYYMMDD_HHMMSS.csv`
- `rf_cv_5models_YYYYMMDD_HHMMSS.csv`
- `lightgbm_cv_5models_YYYYMMDD_HHMMSS.csv`
- `xgboost_cv_5models_YYYYMMDD_HHMMSS.csv` ✨
- `catboost_cv_5models_YYYYMMDD_HHMMSS.csv` ✨

---

## 💡 Tips

### 実行時間を短縮したい場合

1. **フォールド数を減らす**
   ```python
   comparator = FiveModelComparator(n_folds=3)  # 5 → 3
   ```
   → 実行時間が約60%に短縮

2. **LightGBMとXGBoostとCatBoostのみ比較**
   - スクリプト内でロジスティック回帰とRFをコメントアウト

### 途中で中断してしまった場合

- 各foldの結果は独立しているため、再実行が必要
- 長時間実行の場合は、PCをスリープさせないように設定

---

## 🔍 エラー対処

### "ModuleNotFoundError: No module named 'xgboost'"

```bash
pip install xgboost
```

### "ModuleNotFoundError: No module named 'catboost'"

```bash
pip install catboost
```

### メモリ不足エラー

- `n_folds` を 3 に減らす
- 各モデルの `n_jobs` を `-1` から `4` に変更

---

## 📈 期待される結果

| 指標 | ロジスティック | RF | LightGBM | XGBoost | CatBoost |
|------|--------------|-----|----------|---------|----------|
| PR-AUC | 0.06 | 0.15 | **0.21** | 0.19 | 0.18 |
| ROC-AUC | 0.81 | 0.87 | **0.89** | 0.88 | 0.87 |
| F1 | 0.12 | 0.22 | **0.29** | 0.26 | 0.24 |

**予測**:
- **LightGBMが最良** の性能
- **XGBoostとCatBoost**がLightGBMに近い性能
- **Random Forest**が中間
- **ロジスティック回帰**がベースライン

---

**作成日**: 2025年12月11日 15:18  
**次のステップ**: ライブラリをインストールして実行
