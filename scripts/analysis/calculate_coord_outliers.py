import pandas as pd
import numpy as np

def main():
    data_path = "data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv"
    lat_col = '地点　緯度（北緯）'
    lon_col = '地点　経度（東経）'
    
    print(f"Reading {data_path}...")
    # Only load coordinate columns to save memory/time
    df = pd.read_csv(data_path, usecols=[lat_col, lon_col])
    
    original_len = len(df)
    print(f"Total rows: {original_len:,}")

    # --- Convert Coordinates Logic ---
    print("Converting coordinates...")
    def convert_coord_vectorized(series):
        """ベクトル化された座標変換（混在データ対応）"""
        result = pd.Series(index=series.index, dtype=float)
        
        # 欠損値を除外
        valid_mask = series.notna()
        valid_vals = series[valid_mask].astype(float)
        
        # 整数形式（>1000000）と度数法（<1000）を判別
        is_integer_format = valid_vals > 1000000
        
        # 整数形式の変換 (dddmmssss)
        int_vals = valid_vals[is_integer_format].astype(int)
        deg = int_vals // 10000000
        remainder = int_vals % 10000000
        minutes = remainder // 100000
        seconds = (remainder % 100000) / 1000
        result.loc[valid_vals[is_integer_format].index] = deg + minutes / 60 + seconds / 3600
        
        # 既に度数法のもの
        result.loc[valid_vals[~is_integer_format].index] = valid_vals[~is_integer_format]
        
        return result

    df['lat'] = convert_coord_vectorized(df[lat_col])
    df['lon'] = convert_coord_vectorized(df[lon_col])
    
    # --- Filtering Logic ---
    print("Filtering invalid coordinates...")
    lat_min, lat_max = 24.0, 46.0
    lon_min, lon_max = 122.0, 146.0
    
    # Check for NaN first (conversion failure or original missing)
    nan_coords = df['lat'].isna() | df['lon'].isna()
    nan_count = nan_coords.sum()
    print(f"NaN coords (or conversion failed): {nan_count:,}")
    
    # Filter valid range
    valid_range_mask = (
        (df['lat'] >= lat_min) & (df['lat'] <= lat_max) &
        (df['lon'] >= lon_min) & (df['lon'] <= lon_max)
    )
    
    # Valid rows are those that are NOT NaN AND in range
    final_valid_mask = (~nan_coords) & valid_range_mask
    
    valid_count = final_valid_mask.sum()
    removed_count = original_len - valid_count
    
    print("-" * 30)
    print(f"Original Count: {original_len:,}")
    print(f"Valid Count:    {valid_count:,}")
    print(f"Removed Count:  {removed_count:,}")
    print(f"Removed %:      {removed_count / original_len * 100:.4f}%")
    print("-" * 30)
    
    # 詳細な内訳
    # 1. NaN
    print(f"Breakdown:")
    print(f"  - NaN / Conversion Error: {nan_count:,}")
    
    # 2. Out of range (calculating on non-NaNs)
    out_of_range = (~valid_range_mask) & (~nan_coords)
    out_of_range_count = out_of_range.sum()
    print(f"  - Out of Range (Lat/Lon): {out_of_range_count:,}")
    
    if out_of_range_count > 0:
        outliers = df[out_of_range]
        print("\nSample Outliers:")
        print(outliers[['lat', 'lon']].head())

if __name__ == "__main__":
    main()
