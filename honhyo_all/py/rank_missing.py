import pandas as pd, os, sys
path = r'c:\\Users\\socce\\4nd year\\software labratory\\0.オープンデータを用いた交通事故原因分析\\オープンデータ\\honhyo_all\\honhyo_all_shishasuu_binary.csv'
print('Loading CSV...')
df = pd.read_csv(path, low_memory=False)
missing = df.isna().sum()
missing = missing[missing>0].sort_values(ascending=False)
print('Missing values per column (top 20):')
print(missing.head(20))
