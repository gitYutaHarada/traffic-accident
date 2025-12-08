
import pandas as pd
import numpy as np
import os

def map_road_type(code):
    """
    路線コードを道路種別(0-14)に変換する
    """
    try:
        c = int(code)
    except:
        return 13 # その他 (変換不能)

    if 1 <= c <= 999: return 0  # 国道
    if 1000 <= c <= 1499: return 1 # 主要地方道(県道)
    if 1500 <= c <= 1999: return 2 # 主要地方道(市道)
    if 2000 <= c <= 2999: return 3 # 一般都道府県道
    if 3000 <= c <= 3999: return 4 # 一般市町村道
    if 4000 <= c <= 4999: return 5 # 高速自動車国道
    if 5000 <= c <= 5499: return 6 # 自専道(指定)
    if 5500 <= c <= 5999: return 7 # 自専道(その他)
    if 6000 <= c <= 6999: return 8 # 道路運送法上
    if 7000 <= c <= 7999: return 9 # 農道
    if 8000 <= c <= 8499: return 10 # 林道
    if 8500 <= c <= 8999: return 11 # 港湾道
    if 9000 <= c <= 9499: return 12 # 私道
    if 9500 == c: return 13 # その他
    if 9900 == c: return 14 # 一般の交通の用に供するその他の道路
    
    return 13 # その他

def main():
    input_file = r"c:\Users\socce\software-lab\traffic-accident\data\processed\honhyo_clean_no_leakage.csv"
    output_file = r"c:\Users\socce\software-lab\traffic-accident\data\processed\honhyo_clean_road_type.csv"
    
    print(f"Loading data: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")
    
    print("Converting '路線コード' to 'road_type'...")
    # 欠損値対応
    df['路線コード'] = df['路線コード'].fillna(9500)
    df['road_type'] = df['路線コード'].apply(map_road_type)
    
    print("Dropping original '路線コード' column...")
    df = df.drop(columns=['路線コード'])
    
    print(f"New shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    
    print(f"Saving to: {output_file}")
    df.to_csv(output_file, index=False)
    print("Success!")

if __name__ == "__main__":
    main()
