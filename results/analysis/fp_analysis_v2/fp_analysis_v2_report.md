# 誤検知（False Positive）分析レポート v2.1

生成日時: 2025-12-15 14:54:19

## 概要
本レポートは、死亡事故予測モデルの誤検知（False Positive）を詳細に分析した結果をまとめる。

## 1. 全体結果（閾値 = 0.5）
- True Positive (TP): 12,335
- False Positive (FP): 279,760
- Precision: 4.22%

## 2. 厳しい閾値での分析（閾値 = 0.94）
Precision 20%以上を目標とした閾値。
- Hard False Positive: 126
- Precision: 45.69%

## 3. 重傷度分析
詳細: `fp_severity_distribution.csv`

## 4. SHAP個票分析
保存先: `shap_force_plots/`
