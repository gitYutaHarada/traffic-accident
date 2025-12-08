# データリークが発生したファイル保管場所

**作成日時:** 2025年12月8日  
**目的:** データリークが発生していた実験に使用されたファイルを隔離保管

---

## ⚠️ 警告

**このディレクトリ内のファイルは、事後情報（事故発生後にのみわかる情報）を含んでおり、データリークが発生します。**

**機械学習モデルの学習には使用しないでください。**

---

## 📁 保管ファイル一覧

### 1. honhyo_model_ready.csv

**元の場所**: `data/processed/`  
**使用された実験**: 
- Optuna ハイパーパラメータチューニング
- その他の一部実験

**問題点**:
- `事故内容`列が含まれる（死亡/負傷の区分 - 100%リーク）
- `人身損傷程度（当事者A・B）`が含まれる
- `車両の損壊程度（当事者A・B）`が含まれる
- その他事後情報列8個が含まれる

**影響**:
- このデータで学習したモデルは虚偽の高性能（Recall 95.9%, Precision 55.1%等）を示した
- 実用性はゼロ

---

### 2. honhyo_all_shishasuu_binary.csv

**元の場所**: `data/raw/`  
**使用された実験**:
- LightGBM + Weight
- LightGBM + SMOTE
- Random Forest (refined)
- Random Forest (initial)

**問題点**:
- 元データそのものに事後情報列が含まれている
- スクリプトで除外を試みたが、全角/半角括弧の不一致で失敗

**影響**:
- 報告された全ての高性能が無効
- すべての実験を再実行する必要がある

---

### 3. honhyo_all_upsampled.csv

**元の場所**: `data/processed/`  
**使用された実験**: Random Forest (upsampled)

**問題点**:
- アップサンプリング済みだが、事後情報列を除外していない
- ベースデータに事後情報が含まれる

---

### 4. honhyo_all_preaccident_only.csv

**元の場所**: `data/processed/`  
**使用された実験**: 一部の初期実験

**問題点**:
- 名前は「事前情報のみ」だが、実際には事後情報が含まれていた可能性
- 検証が不十分

---

## 📊 保管された実験結果ファイル

### results/analysis/ (12ファイル)

データリーク実験の分析結果:

| ファイル | 実験 | 内容 |
|---------|------|------|
| `optuna_best_params.json` | Optuna | 最適パラメータ（無効） |
| `optuna_cv_metrics.csv` | Optuna | 交差検証メトリクス（無効） |
| `optuna_feature_importance.csv` | Optuna | 特徴量重要度（無効） |
| `optuna_study_history.csv` | Optuna | 最適化履歴（無効） |
| `lgbm_weighted_metrics.csv` | LightGBM + Weight | メトリクス（無効） |
| `lgbm_weighted_feature_importance.csv` | LightGBM + Weight | 特徴量重要度（無効） |
| `weighted_model_metrics.csv` | LightGBM + Weight | モデルメトリクス（無効） |
| `advanced_model_metrics.csv` | LightGBM + SMOTE | メトリクス（無効） |
| `refined_model_cv_metrics.csv` | Random Forest Refined | CV結果（無効） |
| `upsampled_model_metrics.csv` | Random Forest Upsampled | メトリクス（無効） |
| `feature_importance.csv` | Random Forest (初期) | 特徴量重要度（無効） |
| `weighted_auc_score.txt` | LightGBM + Weight | AUCスコア（無効） |

### results/experiments/ (5ファイル)

データリーク実験のレポート:

| ファイル | 実験 | 状態 |
|---------|------|------|
| `optuna_tuning_experiment.md` | Optuna | **無効** - データリークあり |
| `weighted_model_experiment.md` | LightGBM + Weight | **無効** - データリークあり |
| `advanced_model_experiment.md` | LightGBM + SMOTE | **無効** - データリークあり |
| `refined_model_experiment.md` | Random Forest Refined | **無効** - データリークあり |
| `upsampling_experiment.md` | Random Forest Upsampled | **無効** - データリークあり |

### results/visualizations/ (14ファイル)

データリーク実験の可視化:

| ファイル | 実験 | 内容 |
|---------|------|------|
| `optuna_pr_curve.png` | Optuna | PR曲線（無効） |
| `optuna_confusion_matrix.png` | Optuna | 混同行列（無効） |
| `optuna_feature_importance_top20.png` | Optuna | 特徴量重要度（無効） |
| `lgbm_weighted_pr_curve.png` | LightGBM + Weight | PR曲線（無効） |
| `lgbm_weighted_confusion_matrix.png` | LightGBM + Weight | 混同行列（無効） |
| `lgbm_weighted_feature_importance.png` | LightGBM + Weight | 特徴量重要度（無効） |
| `pr_curve_advanced.png` | LightGBM + SMOTE | PR曲線（無効） |
| `confusion_matrix_advanced.png` | LightGBM + SMOTE | 混同行列（無効） |
| `feature_importance_refined.png` | Random Forest Refined | 特徴量重要度（無効） |
| `feature_importance_upsampled.png` | Random Forest Upsampled | 特徴量重要度（無効） |
| `confusion_matrix_upsampled.png` | Random Forest Upsampled | 混同行列（無効） |
| `feature_importance.png` | Random Forest (初期) | 特徴量重要度（無効） |
| `pr_curve_weighted.png` | LightGBM + Weight | PR曲線（無効） |
| `confusion_matrix_weighted.png` | LightGBM + Weight | 混同行列（無効） |

> [!WARNING]
> これらの実験結果は**すべて無効**です。データリークにより虚偽の高性能を示しているため、参照のみに使用してください。

---

## ✅ 代替ファイル

これらのファイルの代わりに、以下を使用してください：

**クリーンデータセット**:
- **ファイル**: `data/processed/honhyo_clean_no_leakage.csv`
- **特徴**: 事後情報を完全に除外済み
- **検証**: codebookと照合済み
- **ドキュメント**: `data/processed/README_clean_dataset.md`

---

## 🗂️ 保管理由

### なぜ削除せず保管するのか？

1. **参照用**: 過去の実験結果と比較する際の参照データとして
2. **監査用**: データリーク発生の証拠として保管
3. **学習用**: 同じミスを繰り返さないため

### 取り扱い注意事項

> [!CAUTION]
> - **絶対に機械学習モデルの学習に使用しない**
> - **分析結果を外部に報告しない**
> - **参照のみに使用する**

---

## 📊 データリーク内容の詳細

### 含まれていた事後情報列（14列）

| # | 列名 | リーク度 | 理由 |
|---|------|----------|------|
| 1 | **事故内容** | **🔴 極めて高** | 死亡/負傷の区分そのもの |
| 2 | 人身損傷程度（当事者A） | 🔴 極めて高 | 死亡・重傷・軽傷 |
| 3 | 人身損傷程度（当事者B） | 🔴 極めて高 | 同上 |
| 4 | 負傷者数 | 🔴 極めて高 | 事故の重大性 |
| 5 | 車両の損壊程度（当事者A） | 🟠 高 | 衝突の重大性 |
| 6 | 車両の損壊程度（当事者B） | 🟠 高 | 同上 |
| 7 | 車両の衝突部位（当事者A） | 🟡 中 | 車両の損傷箇所 |
| 8 | 車両の衝突部位（当事者B） | 🟡 中 | 同上 |
| 9-12 | エアバッグ・サイドエアバッグ（A・B） | 🟢 低 | 作動状況が含まれる可能性 |
| 13-14 | 資料区分、本票番号 | - | 管理情報 |

### データリークの検証方法

`事故内容`列と`死者数`の相関:
```
事故内容  死者数=0  死者数=1
1(死亡)        0     16,266  ← 100%相関
2(負傷)  1,879,008       1
```

**結論**: 完璧な予測変数として機能してしまう

---

## 🔗 関連ドキュメント

### 調査レポート
- [data_leakage_investigation.md](file:///c:/Users/socce/software-lab/traffic-accident/results/experiments/data_leakage_investigation.md) - データリーク調査
- [comprehensive_data_leakage_audit.md](file:///c:/Users/socce/software-lab/traffic-accident/results/experiments/comprehensive_data_leakage_audit.md) - 全実験監査

### クリーンデータ
- [honhyo_clean_no_leakage.csv](file:///c:/Users/socce/software-lab/traffic-accident/data/processed/honhyo_clean_no_leakage.csv) - 使用すべきデータ
- [README_clean_dataset.md](file:///c:/Users/socce/software-lab/traffic-accident/data/processed/README_clean_dataset.md) - クリーンデータ説明
- [VERIFICATION_REPORT.md](file:///c:/Users/socce/software-lab/traffic-accident/data/processed/VERIFICATION_REPORT.md) - 最終検証レポート

---

## 📝 履歴

| 日時 | アクション | 詳細 |
|------|-----------|------|
| 2025-12-08 | ディレクトリ作成 | data_leakageディレクトリを作成 |
| 2025-12-08 | ファイル移動 | processedフォルダから3ファイルを移動 |
| 2025-12-08 | ファイルコピー | rawフォルダから1ファイルをコピー |
| 2025-12-08 | README作成 | 本ドキュメントを作成 |
