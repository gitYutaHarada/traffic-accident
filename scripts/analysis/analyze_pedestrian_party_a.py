import pandas as pd
from pathlib import Path
import sys

# Windows environments encoding
sys.stdout.reconfigure(encoding='utf-8')

# Path to the dataset
# Using the raw one to ensure we have the original codes
DATA_PATH = Path('c:/Users/socce/software-lab/traffic-accident/data/raw/honhyo_all_shishasuu_binary.csv')

def analyze_party_a_pedestrian():
    if not DATA_PATH.exists():
        print(f"Error: File not found at {DATA_PATH}")
        return

    print("Loading dataset...")
    try:
        # Load necessary columns
        usecols = ['当事者種別（当事者A）', '当事者種別（当事者B）', '事故類型', '事故内容', '死者数']
        df = pd.read_csv(DATA_PATH, usecols=lambda c: c in usecols)
        
        target_col = '当事者種別（当事者A）'
        
        # Pedestrian Codes
        # 61: 歩行者
        # 71: 歩行者以外の道路上の人
        # 72: 道路外の人
        target_codes = [61, 71, 72]
        
        print(f"\nTotal Records: {len(df)}")
        
        for code in target_codes:
            count = len(df[df[target_col] == code])
            print(f"Count for Code {code}: {count}")
            
        # Filter for mainly 61
        df_ped = df[df[target_col].isin(target_codes)]
        
        if len(df_ped) > 0:
            print("\n--- Details for Pedestrian in Party A ---")
            print(df_ped[target_col].value_counts())
            
            print("\nAccident Type (事故類型) for these cases:")
            print(df_ped['事故類型'].value_counts())
            
            print("\nParty B Type (What did the pedestrian hit/get hit by?):")
            print(df_ped['当事者種別（当事者B）'].value_counts().head(10))
            
            print("\nExample records:")
            print(df_ped.head())
        else:
            print("\nNo pedestrians found in Party A.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    analyze_party_a_pedestrian()
