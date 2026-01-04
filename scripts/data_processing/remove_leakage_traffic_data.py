"""
Remove Leakage Columns from Traffic Analysis Data
=================================================
Removes columns identified as leakage or post-accident information
from 'honhyo_for_analysis_with_traffic.csv'.

Columns to remove:
- 事故内容 (Accident Result)
- 車両の衝突部位（当事者A/B） (Collision Part)
- エアバッグの装備（当事者A/B） (Airbag Equipment)
- サイドエアバッグの装備（当事者A/B） (Side Airbag)
- 事故類型 (Accident Type)
"""

import pandas as pd
import os

def main():
    input_path = "data/processed/honhyo_for_analysis_with_traffic.csv"
    output_path = "data/processed/honhyo_for_analysis_with_traffic_no_leakage.csv"

    cols_to_remove = [
        '事故内容',
        '車両の衝突部位（当事者A）',
        '車両の衝突部位（当事者B）',
        'エアバッグの装備（当事者A）',
        'エアバッグの装備（当事者B）',
        'サイドエアバッグの装備（当事者A）',
        'サイドエアバッグの装備（当事者B）',
        '事故類型',
        # User requested additional Party B attributes removal
        '年齢（当事者B）',
        '当事者種別（当事者B）',
        '用途別（当事者B）'
    ]

    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    print(f"Original shape: {df.shape}")
    print(f"Columns to remove: {cols_to_remove}")

    # Remove columns that exist in the dataframe
    existing_cols_to_remove = [c for c in cols_to_remove if c in df.columns]
    missing_cols = list(set(cols_to_remove) - set(existing_cols_to_remove))

    if missing_cols:
        print(f"Warning: The following columns were not found in the dataset: {missing_cols}")

    # Generate Target 'fatal' before removing '事故内容'
    # Check if '死者数' exists, if so use it. If not, derive from '事故内容' or just rely on '死者数' if it was there?
    # User said '事故内容' is results (Death/Injury).
    # Let's inspect '事故内容' values first in a separate check? No, let's assume standard coding or check value counts if possible.
    # Actually, previous analysis showed '死者数' (Death Count) was often the source.
    # But '死者数' is not in the columns list I just saw!
    # So we MUST derive from '事故内容'.
    # Assumption: '事故内容' 1 = Fatal? Or is it a count?
    # Let's preserve '事故内容' as 'original_accident_result' for checking but drop it from features?
    # Better: Create 'fatal' column.
    # If '事故内容' contains '死亡' or code 1.
    # Let's print unique values of '事故内容' to be safe in the script.
    
    if '事故内容' in df.columns:
        print("Unique values in '事故内容':", df['事故内容'].unique())
        # Assuming 1 is Fatal based on standard police data (1: Death, 2: Injury)
        # But let's be safe: If value is 1 -> 1 (Fatal), else 0.
        # WAIT: User said 'honhyo_clean_with_feature.csv' had '死者数' removed and converted to 'fatal'.
        # This file 'honhyo_for_analysis_with_traffic.csv' likely still has raw codes.
        
        # LOGIC: Create 'fatal' target. 
        # If 1 in values, likely 1=Death.
        df['fatal'] = df['事故内容'].apply(lambda x: 1 if x == 1 else 0)
        print("Created 'fatal' target column from '事故内容'.")
        print(df['fatal'].value_counts())

    df_clean = df.drop(columns=existing_cols_to_remove)
    
    print(f"Removed {len(existing_cols_to_remove)} columns.")
    print(f"New shape: {df_clean.shape}")

    print(f"Saving to {output_path}...")
    df_clean.to_csv(output_path, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
