import pandas as pd
import numpy as np
import sys

# Windows console encoding usually cp932, but standard output might handle utf-8.
# We will inspect values directly.

file_path = 'data/raw/honhyo_all_shishasuu_binary.csv'

print(f"Reading {file_path}...")

# 読み込み試行
success = False
for enc in ['utf-8', 'cp932', 'shift_jis', 'utf-16']:
    try:
        print(f"Trying encoding: {enc}")
        df = pd.read_csv(file_path, nrows=10000, encoding=enc)
        print("Success!")
        success = True
        break
    except Exception as e:
        print(f"Failed with {enc}: {e}")

if not success:
    print("Could not read file.")
    sys.exit(1)

# カラム名探索
lat_candidates = [c for c in df.columns if '北緯' in str(c) or '緯度' in str(c)]
lon_candidates = [c for c in df.columns if '東経' in str(c) or '経度' in str(c)]

print(f"Lat candidates: {lat_candidates}")
print(f"Lon candidates: {lon_candidates}")

if not lat_candidates or not lon_candidates:
    print("Could not find lat/lon columns.")
    # Show all columns to help debug (escape non-ascii if needed)
    # print(df.columns.tolist())
    sys.exit(1)

lat_col = lat_candidates[0]
lon_col = lon_candidates[0]

# 値の確認
print(f"--- Sample Values ---")
print(f"{lat_col}: {df[lat_col].head(5).tolist()}")
print(f"{lon_col}: {df[lon_col].head(5).tolist()}")

print(f"--- Stats (First 10000 rows) ---")
print(f"{lat_col} Min: {df[lat_col].min()}, Max: {df[lat_col].max()}")
print(f"{lon_col} Min: {df[lon_col].min()}, Max: {df[lon_col].max()}")
