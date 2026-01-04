# Spatio-Temporal Stage2 モデル

交通事故データを用いた「空間・時系列（Spatio-Temporal）を取り入れた死亡事故予測モデル（Stage2）」の実装。

## 📁 ディレクトリ構造

```
scripts/spatio_temporal/
├── run.py                          # 統合実行スクリプト
├── preprocess_spatio_temporal.py   # 前処理
├── graph_builder.py                # グラフ構築
├── train_spatio_temporal.py        # 学習パイプライン
├── optuna_search.py                # ハイパーパラメータ探索
├── evaluate.py                     # 評価
├── visualize.py                    # 可視化
├── models/
│   ├── lstm_geohash.py             # LSTM時系列モデル
│   ├── temporal_gnn.py             # Temporal GNN
│   └── knn_gnn.py                  # kNN-graph GNN
├── utils/
│   ├── checkpoint.py               # チェックポイント管理
│   └── metrics.py                  # 評価指標
├── Makefile
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🚀 クイックスタート

### 1. 依存関係のインストール

```bash
cd scripts/spatio_temporal
pip install -r requirements.txt
```

### 2. 全工程を一括実行

```bash
python run.py --all
```

これにより以下が順番に実行されます：
1. データ前処理（ジオハッシュ生成、時系列特徴量）
2. グラフ構築（kNNグラフ）
3. モデル学習（MLP、kNN-GNN）
4. 評価
5. 可視化（ヒートマップ、PR曲線等）
6. レポート生成

### 3. 個別実行

```bash
# 前処理のみ
python preprocess_spatio_temporal.py

# 学習のみ
python train_spatio_temporal.py --model knn_gnn

# Optuna探索
python optuna_search.py --n-trials 50
```

## 📊 モデル

### 1. MLP (ベースライン)
シンプルな多層パーセプトロン。空間情報を直接使用せず、特徴量のみで分類。

### 2. kNN-GNN
事故サンプルをノードとし、空間的近傍（Haversine距離でkNN）をエッジとしたグラフニューラルネットワーク。

### 3. Temporal GCN (TGCN)
時系列グラフ畳み込みネットワーク。GCN + GRU の組み合わせ。

### 4. LSTM (ジオハッシュ単位)
各ジオハッシュセルの時系列をLSTMで学習し、事故サンプルに結合。

## ⚙️ 主要パラメータ

| パラメータ | デフォルト値 | 説明 |
|-----------|-------------|------|
| `--data-path` | `data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv` | 入力データ |
| `--output-dir` | `results/spatio_temporal` | 出力ディレクトリ |
| `--train-years` | `2018,2019` | 学習データの年 |
| `--val-years` | `2020,2020` | 検証データの年 |
| `--test-years` | `2021,2024` | テストデータの年 |
| `--epochs` | `100` | 学習エポック数 |
| `--batch-size` | `1024` | バッチサイズ |
| `--k` | `8` | kNNグラフのk値 |
| `--optuna` | `False` | Optuna探索を実行 |
| `--n-optuna-trials` | `50` | Optunaの試行回数 |

## 📈 評価指標

- **PR-AUC** (最重要): Precision-Recall曲線の下の面積
- **Recall@k**: Top-k予測での正例検出率
- **Precision@k**: Top-k予測での適合率
- **ROC-AUC**: ROC曲線の下の面積
- **ECE**: Expected Calibration Error（校正誤差）
- **Brier Score**: 確率予測の精度

## 🗺️ 出力ファイル

| ファイル | 説明 |
|----------|------|
| `heatmap.html` | 予測確率のヒートマップ（Folium） |
| `top_n_map.html` | Top-100高リスク地点マップ |
| `pr_curve.png` | PR曲線 |
| `roc_curve.png` | ROC曲線 |
| `experiment_report.md` | 実験レポート |
| `results_summary.json` | 数値結果サマリ |
| `test_predictions.parquet` | テストデータの予測結果 |

## 🛡️ リーク防止

時系列リークを防ぐため、以下の対策を実装：

1. **時間ベース分割**: 年次でtrain/val/testを分割
2. **Shift処理**: 過去ウィンドウの集計時に`shift(1)`を適用
3. **未来情報の除外**: 予測時点より後のデータを使用しない

## 💾 チェックポイント機能

学習中断時に途中から再開可能：

```bash
# 自動的に最新のチェックポイントから再開
python train_spatio_temporal.py --model knn_gnn

# チェックポイントをクリアして最初から
rm -rf results/spatio_temporal/checkpoints
python train_spatio_temporal.py --model knn_gnn
```

## 🐳 Docker

```bash
# ビルド
docker build -t spatio-temporal-stage2 .

# 実行（GPU使用）
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  spatio-temporal-stage2 python run.py --all
```

## 📋 依存関係

- Python >= 3.8
- PyTorch >= 2.0.0
- PyTorch Geometric >= 2.3.0
- pandas, numpy, scikit-learn
- Optuna
- folium
- geohash2
- TensorBoard

## 🔍 トラブルシューティング

### PyTorch Geometricのインストール

```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install torch-geometric
```

### GPUメモリ不足

バッチサイズを小さくするか、サブセットで学習：

```bash
python train_spatio_temporal.py --batch-size 512
```

### ジオハッシュのインストール

```bash
pip install geohash2
```

## 📝 ライセンス

このプロジェクトは教育・研究目的で作成されています。
