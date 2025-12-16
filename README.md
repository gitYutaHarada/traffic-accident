# 交通事故分析プロジェクト

交通事故データを用いた機械学習による**死亡事故リスク予測・分析**プロジェクト

## � プロジェクト概要

警察庁の交通事故統計データ（約190万件）を用いて、**事故発生前に観測可能な情報のみ**で死亡事故リスクを予測するモデルを構築しました。

### 主なモデル改善

| 施策 | 効果 |
|------|------|
| **地理情報のエリアID化** | Feature Importance 1位を獲得 |
| **カテゴリカル変数の適切な扱い** | LightGBMのcategory型を活用 |
| **カウントエンコーディング** | F1スコア約1.2%向上 |
| **日時情報の分解** | 月・時・曜日・年を特徴量化 |

## �📁 プロジェクト構造

```
traffic-accident/
├── data/
│   ├── raw/                        # 元データ
│   │   └── honhyo_all_shishasuu_binary.csv
│   └── processed/                  # 加工済みデータ
│       └── honhyo_model_ready.csv  # エリアID・日時分解済み
│
├── scripts/
│   ├── preprocessing/              # 前処理
│   │   └── create_model_dataset.py # データ加工スクリプト
│   └── analysis/                   # 分析スクリプト
│       ├── lightgbm_weighted_optimization.py  # 🎯 LightGBMモデル (推奨)
│       └── day_of_month_eda.py     # 日別事故傾向分析
│
├── outputs/results/
│   ├── experiments/                # 実験結果レポート
│   │   ├── categorical_datetime_experiment.md
│   │   └── day_of_month_analysis.md
│   ├── visualizations/             # 可視化
│   │   ├── feature_importance.png
│   │   ├── pr_curve_weighted.png
│   │   └── day_fatality_rate.png
│   └── analysis/                   # 分析結果CSV
│       └── weighted_model_metrics.csv
│
└── honhyo_all/details/             # データ定義書
    └── codebook_extracted.txt
```

## 🚀 使い方

### 1. 環境構築

```powershell
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn
```

### 2. データ前処理

```powershell
python scripts/preprocessing/create_model_dataset.py
```

### 3. モデル学習・評価

```powershell
python scripts/analysis/lightgbm_weighted_optimization.py
```

## � モデル性能

### LightGBM + scale_pos_weight モデル

| 指標 | スコア |
|------|--------|
| **AUC** | 0.885 |
| **F1 Score** | 0.198 (閾値0.5) |
| **Recall (発見率)** | 42.2% |
| **Precision (適合率)** | 12.9% |

> **注**: Recall 80%が必要な場合は閾値を0.032に設定（Precision 3.6%）

### 主要特徴量 (Feature Importance Top 5)

1. **Area_Cluster_ID** (地理エリア) - 10226
2. **路線コード** - 10055
3. **市区町村コード** - 2565
4. **地点コード** - 2060
5. **発生時** (時間帯) - 884

## 🔬 実験結果

詳細は `outputs/results/experiments/` を参照:

- **カテゴリカル変数・日時分解**: [categorical_datetime_experiment.md](outputs/results/experiments/categorical_datetime_experiment.md)
- **日別事故傾向分析**: [day_of_month_analysis.md](outputs/results/experiments/day_of_month_analysis.md)

## 🛠️ 技術スタック

- Python 3.x
- **LightGBM** (勾配ブースティング)
- pandas, numpy (データ処理)
- scikit-learn (評価指標, クラスタリング)
- matplotlib, seaborn (可視化)

## 📝 データ出典

警察庁 交通事故統計データ (2019-2023年)

## 📄 ライセンス

このプロジェクトは教育・研究目的で作成されています。
