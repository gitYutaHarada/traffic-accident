"""データの年別件数と日付範囲を確認するスクリプト"""
import pandas as pd

# データ読み込み
train = pd.read_parquet('data/spatio_temporal/raw_train.parquet')
val = pd.read_parquet('data/spatio_temporal/raw_val.parquet')
test = pd.read_parquet('data/spatio_temporal/raw_test.parquet')

# 統合
all_data = pd.concat([train, val, test], ignore_index=True)

print("=== 年ごとのデータ件数 ===")
year_counts = all_data['year'].value_counts().sort_index()
for year, count in year_counts.items():
    print(f"{int(year)}年: {count:,}件")
print("---")
print(f"合計: {len(all_data):,}件")

print("\n=== データセット別 ===")
print(f"Train (2018-2022): {len(train):,}件")
print(f"Validation (2023): {len(val):,}件")
print(f"Test (2024): {len(test):,}件")

# 日付の範囲を確認
all_data['date'] = pd.to_datetime(
    all_data['year'].astype(int).astype(str) + '-' + 
    all_data['month'].astype(int).astype(str).str.zfill(2) + '-' + 
    all_data['day'].astype(int).astype(str).str.zfill(2),
    errors='coerce'
)

min_date = all_data['date'].min()
max_date = all_data['date'].max()

print("\n=== データ期間 ===")
print(f"最初の日付: {min_date.strftime('%Y年%m月%d日')}")
print(f"最後の日付: {max_date.strftime('%Y年%m月%d日')}")
