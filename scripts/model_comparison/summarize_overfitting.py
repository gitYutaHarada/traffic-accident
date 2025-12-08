import pandas as pd

df = pd.read_csv('results/model_comparison/logistic_regression_1pct/overfitting_metrics.csv')

print('\n' + '='*80)
print('過学習分析結果サマリー')
print('='*80)

print('\n【訓練データの平均性能】')
print(f'  AUC:       {df["Train_AUC"].mean():.4f}')
print(f'  F1 Score:  {df["Train_F1"].mean():.4f}')
print(f'  Recall:    {df["Train_Recall"].mean():.4f}')
print(f'  Precision: {df["Train_Precision"].mean():.4f}')
print(f'  Accuracy:  {df["Train_Accuracy"].mean():.4f}')

print('\n【検証データの平均性能】')
print(f'  AUC:       {df["Val_AUC"].mean():.4f}')
print(f'  F1 Score:  {df["Val_F1"].mean():.4f}')
print(f'  Recall:    {df["Val_Recall"].mean():.4f}')
print(f'  Precision: {df["Val_Precision"].mean():.4f}')
print(f'  Accuracy:  {df["Val_Accuracy"].mean():.4f}')

print('\n【差分(訓練 - 検証)】')
avg_diff_auc = df['Diff_AUC'].mean()
avg_diff_f1 = df['Diff_F1'].mean()
avg_train_auc = df['Train_AUC'].mean()
avg_train_f1 = df['Train_F1'].mean()

print(f'  AUC:       {avg_diff_auc:+.4f} ({abs(avg_diff_auc)/avg_train_auc*100:.2f}%)')
print(f'  F1 Score:  {avg_diff_f1:+.4f} ({abs(avg_diff_f1)/avg_train_f1*100:.2f}%)')
print(f'  Recall:    {df["Diff_Recall"].mean():+.4f}')
print(f'  Precision: {df["Diff_Precision"].mean():+.4f}')
print(f'  Accuracy:  {df["Diff_Accuracy"].mean():+.4f}')

print('\n' + '='*80)
print('過学習の判定')
print('='*80)

auc_threshold = 0.05
f1_threshold = 0.10

print(f'\n判定基準:')
print(f'  AUC差分の閾値:  {auc_threshold} (5%)')
print(f'  F1差分の閾値:   {f1_threshold} (10%)')

print(f'\nAUCベース:')
if abs(avg_diff_auc) > auc_threshold:
    print(f'  ⚠️  過学習の可能性あり (差分: {abs(avg_diff_auc):.4f} > {auc_threshold})')
else:
    print(f'  ✅ 健全 (差分: {abs(avg_diff_auc):.4f} <= {auc_threshold})')

print(f'\nF1スコアベース:')
if abs(avg_diff_f1) > f1_threshold:
    print(f'  ⚠️  過学習の可能性あり (差分: {abs(avg_diff_f1):.4f} > {f1_threshold})')
else:
    print(f'  ✅ 健全 (差分: {abs(avg_diff_f1):.4f} <= {f1_threshold})')

print(f'\n【総合判定】')
if abs(avg_diff_auc) > auc_threshold or abs(avg_diff_f1) > f1_threshold:
    print('  ⚠️  過学習の兆候が見られます')
else:
    print('  ✅ 過学習は見られません(健全なモデル)')

print('\n' + '='*80)
