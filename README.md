# 交通事故分析プロジェクト

交通事故データを用いた機械学習による死亡事故リスク予測・分析プロジェクト

## 📁 プロジェクト構造

```
traffic-accident/
├── data/                           # データファイル
│   ├── raw/                        # 元データ
│   │   ├── honhyo_all_shishasuu_binary.csv   # 交通事故データ (約190万件)
│   │   └── codebook_2024.pdf                  # データ項目定義書
│   ├── processed/                  # 処理済みデータ
│   │   └── honhyo_all_preaccident_only.csv   # 事前観測可能データのみ
│   └── codebook/                   # コードブック
│       └── codebook_2024_extracted.txt        # PDF抽出テキスト
│
├── scripts/                        # 分析スクリプト
│   ├── data_processing/            # データ処理
│   │   ├── create_preaccident_dataset.py     # 事前データセット作成
│   │   └── extract_pdf.py                     # PDF→テキスト抽出
│   ├── analysis/                   # 分析スクリプト
│   │   ├── random_forest_prevention.py       # 🎯 事故予防モデル (推奨)
│   │   ├── random_forest_analysis.py         # 事故報告分析モデル (参考)
│   │   └── analyze_tojisha_risk.py           # 当事者種別リスク分析
│   └── utils/                      # ユーティリティ
│       └── inspect_data.py                    # データ確認
│
├── results/                        # 分析結果
│   ├── models/                     # モデル出力
│   │   ├── output_prevention.txt             # 予防モデル結果
│   │   ├── output_refined.txt                # 改良モデル結果
│   │   └── output.txt                        # 初期モデル結果
│   ├── visualizations/             # 可視化
│   │   ├── feature_importance_prevention.png  # 予防モデル特徴量
│   │   └── feature_importance.png            # 通常モデル特徴量  
│   └── analysis/                   # 分析結果CSV
│       └── tojisha_shubetsu_risk_ranking.csv # 当事者種別リスク表
│
└── docs/                           # ドキュメント
    ├── memo.txt
    └── header.txt
```

## 🚀 使い方

### 1. 環境構築

```powershell
# 必要なライブラリのインストール
pip install pandas numpy scikit-learn matplotlib seaborn PyPDF2
```

### 2. データ準備

元データ `honhyo_all_shishasuu_binary.csv` を `data/raw/` に配置

### 3. 事故予防モデルの実行

```powershell
# 事前観測データセット作成
python scripts/data_processing/create_preaccident_dataset.py

# 予防モデル訓練・評価
python scripts/analysis/random_forest_prevention.py
```

### 4. リスク分析

```powershell
# 当事者種別ごとの死亡率ランキング作成
python scripts/analysis/analyze_tojisha_risk.py
```

## 📊 モデル概要

### 🎯 事故予防モデル (推奨)

**特徴**: 事故発生**前**に観測可能なデータのみ使用

- **精度**: 99.16%
- **特徴量数**: 33項目
- **用途**: リアルタイム危険予知、カーナビ搭載可能

**主要リスク要因**:
1. 発生時刻 (深夜・早朝が危険)
2. 場所 (緯度・経度)
3. 自車種別 (二輪車 vs 乗用車)
4. 市区町村 (地域別リスク)

### 📋 事故報告モデル (参考)

**特徴**: 事故発生**後**の情報を含む

- **精度**: 99.17%
- **特徴量数**: 55項目
- **用途**: 事故分析、統計レポート作成

## 📈 分析結果

### 死亡率ランキング (当事者B)

1. **歩行者**: 5.65% (最も危険)
2. **列車**: 4.24%
3. **大型バイク (251~400cc)**: 3.47%
4. **大型バイク (401~750cc)**: 3.42%
5. **超大型バイク (751cc~)**: 3.00%

### 実用的な予防策

1. **深夜・早朝の運転を避ける**
2. **事故多発地点の把握・回避**
3. **二輪車より乗用車を選択**
4. **エアバッグ装備車両の使用**

## 🛠️ 技術スタック

- Python 3.x
- pandas, numpy (データ処理)
- scikit-learn (機械学習)
- matplotlib, seaborn (可視化)
- PyPDF2 (PDF処理)

## 📝 データ出典

警察庁 交通事故統計データ (2024年版)

## 📄 ライセンス

このプロジェクトは教育・研究目的で作成されています。
