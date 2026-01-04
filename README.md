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
├── results/
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

## 📊 モデル実験レポート: Stage 2 vs Stage 3

**[詳細レポート: model_comparison_report.md](results/md/model_comparison_report.md)**

最新の実験により、**Stage 3 Stacking** が最も高い予測性能（Test AUC: 0.9030, PR-AUC: 0.1760）を達成しました。

### アプローチの比較

| 重視点 | モデル | 特徴 | 評価 |
| :--- | :--- | :--- | :--- |
| **全体最適** | **Stage 2 Single-Stage** (TabNet等) | 学習データ全量を使用。汎化性能が高い。 | **AUC最強** (0.8984) |
| **難問特化** | **Stage 2 Two-Stage** (CatBoost等) | 「安全」データを除去し、判断が難しいデータに集中。 | **PR-AUCが高い** (0.1611) |
| **統合** | **Stage 3 Stacking** | 上記「最強の盾」と「最強の矛」をロジスティック回帰で統合。 | **総合最強** (AUC 0.9030 / PR-AUC 0.1760) |

### データセットの拡張
本実験では、警察庁データに加え、以下の外部データを統合して環境要因を考慮しています。
*   **国土交通省 道路交通センサス**: 交通量、大型車混入率、混雑度など
*   **国土数値情報 医療機関データ**: 最寄り病院への距離、病床数、救急指定など（救命可能性の指標）

### 🧪 関連スクリプト

本実験（Stage 2 ~ Stage 3）を実施するための主要スクリプトです。

#### 1. Stage 3 Stacking (最終モデル)
異なる特性を持つモデルを統合し、最終的な予測を出力します。
```powershell
python scripts/modeling/train_stage3_stacking.py
```
*   **主な機能**: Single-StageとTwo-Stageの予測値を統合、Easy Sampleの補完、マルチコ（多重共線性）対策

#### 2. Stage 2 Two-Stage (Specialist)
Stage 1で判別しきれなかった「難易度の高い」データに特化して学習します。
```powershell
python scripts/modeling/train_stage2_4models_spatiotemporal_twostage.py
```
*   **主な機能**: Stage 1の予測結果に基づき、安全なデータをフィルタリングして学習（LightGBM, CatBoost, MLP, TabNet）

### その他の実験レポート
詳細は `results/experiments/` を参照:
- **カテゴリカル変数・日時分解**: [categorical_datetime_experiment.md](results/experiments/categorical_datetime_experiment.md)
- **日別事故傾向分析**: [day_of_month_analysis.md](results/experiments/day_of_month_analysis.md)

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
