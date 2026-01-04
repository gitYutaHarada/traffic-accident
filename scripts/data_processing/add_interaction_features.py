"""
Add Interaction Features to Clean Dataset
=========================================
Reads the 'honhyo_clean_with_features.csv' and adds interaction features
inferred from 'honhyo_stage1_unified_filtered.csv' mapping logic.

Excluded features (Post-accident / Leakage):
- 単独事故 (Single vehicle accident)
- Any derived features from 事故類型 (Accident Type)
"""

import pandas as pd
import numpy as np
import os

def main():
    base_dir = "data/processed"
    full_path = os.path.join(base_dir, "honhyo_clean_with_features.csv")
    subset_path = os.path.join(base_dir, "honhyo_stage1_unified_filtered.csv")
    output_path = os.path.join(base_dir, "honhyo_clean_with_interactions_v2.csv")

    print(f"Loading Full Data: {full_path}")
    df_full = pd.read_csv(full_path)
    
    print(f"Loading Subset Data (for inference): {subset_path}")
    # Load only necessary columns for inference to save memory
    subset_cols = ['地形', '道路線形', '非市街地', 'カーブ', '当事者種別（当事者B）', '歩行者事故']
    # Check if cols exist, otherwise minimal load
    df_subset = pd.read_csv(subset_path, usecols=lambda x: x in subset_cols)

    print("Inferring Mappings...")
    
    # 1. Infer '非市街地' (Non-urban) from '地形'
    # Find codes where '非市街地' == 1
    non_urban_codes = set(df_subset[df_subset['非市街地'] == 1]['地形'].dropna().unique())
    print(f"   Non-Urban '地形' Codes: {non_urban_codes}")
    
    # 2. Infer 'カーブ' (Curve) from '道路線形'
    curve_codes = set(df_subset[df_subset['カーブ'] == 1]['道路線形'].dropna().unique())
    print(f"   Curve '道路線形' Codes: {curve_codes}")
    
    # 3. Infer '歩行者' (Pedestrian) from '当事者種別（当事者B）'
    # Use '歩行者事故' flag to find pedestrian codes for Party B
    pedestrian_codes = set(df_subset[df_subset['歩行者事故'] == 1]['当事者種別（当事者B）'].dropna().astype(str).unique())
    print(f"   Pedestrian '当事者種別（当事者B）' Codes: {pedestrian_codes}")
    
    print("Generating Features on Full Dataset...")
    
    # --- Feature Generation ---
    
    # A. Basic Flags
    df_full['非市街地'] = df_full['地形'].apply(lambda x: 1 if x in non_urban_codes else 0)
    df_full['カーブ'] = df_full['道路線形'].apply(lambda x: 1 if x in curve_codes else 0)
    
    # '歩行者事故' - Check Party B type
    # Note: Column name might differ slightly, check for '当事者種別（当事者B）'
    if '当事者種別（当事者B）' in df_full.columns:
        df_full['歩行者事故'] = df_full['当事者種別（当事者B）'].astype(str).apply(lambda x: 1 if x in pedestrian_codes else 0)
    else:
        print("   WARNING: '当事者種別（当事者B）' not found. Skipping '歩行者事故' flag and related interactions.")
        df_full['歩行者事故'] = 0

    # B. Age Flags (高齢者 >= 65)
    # Check column names for Age
    age_col_a = '年齢（当事者A）'
    age_col_b = '年齢（当事者B）'
    
    if age_col_a in df_full.columns:
        df_full['高齢者A'] = (df_full[age_col_a] >= 65).astype(int)
    else:
        df_full['高齢者A'] = 0
        
    if age_col_b in df_full.columns:
        df_full['高齢者B'] = (df_full[age_col_b] >= 65).astype(int)
    else:
        df_full['高齢者B'] = 0

    # C. Time Flags (深夜 22:00 - 05:00)
    # Need '発生日時_時' or similar. Check columns.
    hour_col = [c for c in df_full.columns if '時' in c and ('hour' in c or '発生日時' in c)]
    # Usually '発生日時_時' or just 'hour' if processed
    target_hour_col = '発生日時_時' if '発生日時_時' in df_full.columns else None
    
    # Try looking for just 'hour' or 'Hour'
    if not target_hour_col and 'hour' in df_full.columns:
        target_hour_col = 'hour'
    
    if target_hour_col:
        print(f"   Using hour column: {target_hour_col}")
        # Midnight: 22, 23, 0, 1, 2, 3, 4, 5 (up to 5 in description? Usually 22-05 means <5 or <=5)
        # Description said "22時〜翌5時". Let's assume 22,23,0,1,2,3,4.
        midnight_hours = [22, 23, 0, 1, 2, 3, 4]
        df_full['深夜'] = df_full[target_hour_col].apply(lambda x: 1 if x in midnight_hours else 0)
    else:
        print("   WARNING: Hour column not found. Skipping '深夜' flag and related interactions.")
        df_full['深夜'] = 0

    # --- Interactions ---
    
    print("Creating Interaction Features...")
    
    # 1. 非市街地_カーブ
    df_full['非市街地_カーブ'] = ((df_full['非市街地'] == 1) & (df_full['カーブ'] == 1)).astype(int)
    
    # 2. 高齢者B_深夜
    df_full['高齢者B_深夜'] = ((df_full['高齢者B'] == 1) & (df_full['深夜'] == 1)).astype(int)
    
    # 3. 高齢者B_歩行者
    df_full['高齢者B_歩行者'] = ((df_full['高齢者B'] == 1) & (df_full['歩行者事故'] == 1)).astype(int)
    
    # Excluded: 単独事故 related interactions
    
    # --- Save ---
    
    # Select columns to keep or just append keys?
    # User asked to ADD features.
    
    print(f"Saving to {output_path}...")
    df_full.to_csv(output_path, index=False)
    
    print("Done.")
    print(f"Original shape: {pd.read_csv(full_path, nrows=1).shape[1]} cols") # Approximation
    print(f"New shape: {df_full.shape}")
    print("Added columns:", 
          ['非市街地', 'カーブ', '歩行者事故', '高齢者A', '高齢者B', '深夜', 
           '非市街地_カーブ', '高齢者B_深夜', '高齢者B_歩行者'])

if __name__ == "__main__":
    main()
