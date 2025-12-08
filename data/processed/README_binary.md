# 二値分類用データセット

**ファイル名**: `honhyo_clean_binary.csv`  
**作成日**: 2025年12月8日  
**元データ**: `honhyo_clean_no_leakage.csv`

---

## 📊 データ仕様

### 変換内容

**死者数列を二値分類用に変換**:
- `死者数 = 0` → `0` (非死亡事故)
- `死者数 >= 1` → `1` (死亡事故)

### データ内容

- **行数**: 1,895,275
- **列数**: 36
- **目的変数**: 死者数 (0 or 1)

### クラス分布

| クラス | 件数 | 割合 |
|--------|------|------|
| 0 (非死亡事故) | 1,879,008 | 99.14% |
| 1 (死亡事故) | 16,267 | 0.86% |

**クラス不均衡比**: 115.5:1

---

## 💡 使用方法

### 基本的な読み込み

```python
import pandas as pd

# 二値分類用データの読み込み
df = pd.read_csv('data/processed/honhyo_clean_binary.csv')

# 特徴量と目的変数の分離
X = df.drop(columns=['死者数'])
y = df['死者数']  # 0 or 1

print(f"クラス分布:\n{y.value_counts()}")
```

### クラス不均衡への対処

```python
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# 方法1: SMOTE (アップサンプリング)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 方法2: クラスウェイト
class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(y), 
                                     y=y)
# LightGBMの場合
scale_pos_weight = class_weights[1] / class_weights[0]  # 約115.5
```

---

## 🔗 関連ファイル

- [元データ](honhyo_clean_no_leakage.csv) - 多クラス分類用（死者数: 0, 1, 2, 3, 4, 5）
- [README](README_clean_dataset.md) - クリーンデータセットの詳細
- [検証レポート](VERIFICATION_REPORT.md) - データリーク検証
- [作成スクリプト](../../scripts/data_processing/create_binary_dataset.py)

---

## ⚠️ 注意事項

1. **クラス不均衡**: 死亡事故は全体の約0.86%のため、適切な対策が必要
2. **評価指標**: Accuracyではなく、Recall/Precision/F1/PR-AUCを重視
3. **データリーク**: このデータは事後情報を完全に除外済み
