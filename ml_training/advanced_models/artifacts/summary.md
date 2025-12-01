# 高度モデル 実験レポート
- 実行日時: 2025-11-20T15:24:51
- 入力データ: C:\Users\socce\4nd year\software labratory\0.オープンデータを用いた交通事故原因分析\オープンデータ\honhyo_all\honhyo_all_with_datetime.csv
- 総行数: 1,895,275 (学習 1,516,220, テスト 379,055)

## モデル別評価
| モデル | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | --- | --- | --- | --- |
| Random Forest | 0.9608 | 0.0073 | 0.0264 | 0.0115 | 0.6604 |
| XGBoost | 0.0960 | 0.0089 | 0.9404 | 0.0175 | 0.6679 |
| Neural Network (MLP) | 0.9914 | 0.4324 | 0.0049 | 0.0097 | 0.7591 |