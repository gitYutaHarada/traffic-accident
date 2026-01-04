# SHAP Analysis Report
Generated: 2025-12-30 17:13

## Surrogate Model Fidelity
- Pearson: 0.6397
- Spearman: 0.9282
- MAE: 0.2379

## Top 20 Features
1. 速度規制（指定のみ）（当事者B）_scaled: 0.7052
2. 車道幅員_scaled: 0.3826
3. geohash_accidents_past_365d_scaled: 0.3382
4. 当事者種別（当事者A）_te: 0.3229
5. hour_cos_scaled: 0.2091
6. 都道府県コード_te: 0.1522
7. 速度規制（指定のみ）（当事者A）_scaled: 0.1441
8. 一時停止規制　標識（当事者B）_0: 0.1387
9. 信号機_1: 0.1341
10. 中央分離帯施設等_5: 0.1123
11. 中央分離帯施設等_1: 0.1010
12. 年齢（当事者A）_scaled: 0.0989
13. 一時停止規制　標識（当事者B）_9: 0.0857
14. 一時停止規制　標識（当事者A）_9: 0.0726
15. year: 0.0726
16. 一時停止規制　表示（当事者B）_22: 0.0726
17. 歩車道区分_1: 0.0673
18. 道路形状_14: 0.0655
19. 市区町村コード_te: 0.0597
20. hour_scaled: 0.0573

## Files
- shap_summary_plot.png
- shap_bar_plot.png
- shap_feature_importance.csv
- high_risk_cases.csv
