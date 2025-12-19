import pandas as pd
from pathlib import Path
import sys

# Windows environments encoding
sys.stdout.reconfigure(encoding='utf-8')

# Path to the dataset
DATA_PATH = Path('c:/Users/socce/software-lab/traffic-accident/data/raw/honhyo_all_shishasuu_binary.csv')

def analyze_dataset():
    if not DATA_PATH.exists():
        print(f"Error: File not found at {DATA_PATH}")
        return

    print("Loading dataset...")
    try:
        # Load columns of interest only to save memory if needed, but let's load all to be sure about column names
        # Reading first few rows to check column names first
        preview = pd.read_csv(DATA_PATH, nrows=5)
        cols = preview.columns.tolist()
        
        target_col_a = '人身損傷程度（当事者A）'
        target_col_b = '人身損傷程度（当事者B）'
        
        if target_col_a not in cols or target_col_b not in cols:
            print(f"Columns not found. Available columns: {cols}")
            return

        # Load specific columns
        df = pd.read_csv(DATA_PATH, usecols=[target_col_a, target_col_b, '本票番号', '都道府県コード', '事故内容', '死者数'])
        
        print(f"\nTotal records: {len(df)}")
        
        # Distribution
        print(f"\n--- Distribution for {target_col_a} ---")
        print(df[target_col_a].value_counts(dropna=False).sort_index())
        
        print(f"\n--- Distribution for {target_col_b} ---")
        print(df[target_col_b].value_counts(dropna=False).sort_index())
        
        # Filtering for both == 4
        # Codebook: 4 = 損傷なし (No Injury)
        both_no_injury = df[(df[target_col_a] == 4) & (df[target_col_b] == 4)]
        
        print(f"\n--- Records where both are 4 (No Injury) ---")
        count = len(both_no_injury)
        print(f"Count: {count}")
        
        if count > 0:
            print("\nPreview of matching records (first 20):")
            print(both_no_injury.head(20))
            
            # Additional check: If both are no injury, what is the '事故内容' (Accident details)? 
            # 1: Death, 2: Injury. If both A and B are not injured, maybe C (passenger) or Pedestrian was involved/injured?
            # Or maybe it's a property damage only accident (物件事故)? 
            # But this dataset 'honhyo' usually contains injury/fatal accidents. Property damage only might be excluded or limited.
            print("\nAccident Content (事故内容) for these cases:")
            print(both_no_injury['事故内容'].value_counts(dropna=False).sort_index())
            
            print("\nDeath Count (死者数) for these cases:")
            print(both_no_injury['死者数'].value_counts(dropna=False).sort_index())

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    analyze_dataset()
