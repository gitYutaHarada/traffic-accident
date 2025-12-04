import pandas as pd
import numpy as np
import os

# CSVファイルのパス
csv_path = os.path.join(os.path.dirname(__file__), 'honhyo_all_shishasuu_binary.csv')

# データ読み込み（メモリ節約のためにchunksizeは使用しないが、ファイルが大きいため低メモリモード）
df = pd.read_csv(csv_path, low_memory=False)

# 目的変数（死者数）
if '死者数' not in df.columns:
    raise KeyError('死者数列が見つかりません')

target = df['死者数']

# 数値型カラムのみを対象に相関を計算（目的変数は除外）
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if '死者数' in numeric_cols:
    numeric_cols.remove('死者数')

correlations = {}
for col in numeric_cols:
    # 欠損値は除外して相関計算
    corr = target.corr(df[col])
    correlations[col] = corr

# 絶対値の大きい順にソート
sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

# 結果を書き出すテキストファイルのパス
output_path = os.path.join(os.path.dirname(__file__), 'death_feature_ranking.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write('死亡率に関与する要素ランキング（相関係数）\n')
    f.write('※相関係数が正でも負でも絶対値が大きいほど影響が大きいとみなします。\n\n')
    for i, (col, corr) in enumerate(sorted_corr, 1):
        f.write(f"{i}. {col}: {corr:.4f}\n")

print(f'Ranking written to {output_path}')
