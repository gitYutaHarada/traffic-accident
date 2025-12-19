import pandas as pd
from pathlib import Path
import sys

# Windows environments encoding
sys.stdout.reconfigure(encoding='utf-8')

# Path to the dataset
DATA_PATH = Path('c:/Users/socce/software-lab/traffic-accident/data/raw/honhyo_all_shishasuu_binary.csv')

def analyze_zero_severity_detailed():
    if not DATA_PATH.exists():
        print(f"Error: File not found at {DATA_PATH}")
        return

    print("Loading dataset...")
    try:
        # Load necessary columns including Party B info
        usecols = [
            '人身損傷程度（当事者A）', '当事者種別（当事者A）', 
            '人身損傷程度（当事者B）', '当事者種別（当事者B）',
            '事故類型', '事故内容', '車両単独事故の対象物'
        ]
        
        # Note: '車両単独事故の対象物' might be only in codebook but check if in CSV
        # Reading a sample to check available columns first
        preview = pd.read_csv(DATA_PATH, nrows=5)
        available_cols = preview.columns.tolist()
        final_usecols = [c for c in usecols if c in available_cols]
        
        df = pd.read_csv(DATA_PATH, usecols=final_usecols)
        
        target_col_a = '人身損傷程度（当事者A）'
        
        # Filter for 0
        df_zero = df[df[target_col_a] == 0]
        
        print(f"\n--- Detailed Analysis for Records where {target_col_a} == 0 ---")
        print(f"Count: {len(df_zero)}")
        
        # Group by Party A Type to explain scenarios
        print("\n--- Scenario 1: Party A is Object (75) ---")
        df_obj = df_zero[df_zero['当事者種別（当事者A）'] == 75]
        if not df_obj.empty:
            print(f"Count: {len(df_obj)}")
            print("Party B Type (Who hit existing object A?):")
            print(df_obj['当事者種別（当事者B）'].value_counts(dropna=False).head())
            print("Party B Injury Severity:")
            print(df_obj['人身損傷程度（当事者B）'].value_counts(dropna=False).head())
            if '事故類型' in df_obj.columns:
                print("Accident Type:")
                print(df_obj['事故類型'].value_counts(dropna=False).head())
                
        print("\n--- Scenario 2: Other Cases (Not Object) ---")
        df_other = df_zero[df_zero['当事者種別（当事者A）'] != 75]
        if not df_other.empty:
            print(f"Count: {len(df_other)}")
            print("Party A Type:")
            print(df_other['当事者種別（当事者A）'].value_counts(dropna=False))
            print("Party B Type:")
            print(df_other['当事者種別（当事者B）'].value_counts(dropna=False))
            if '事故類型' in df_other.columns:
                 print("Accident Type:")
                 print(df_other['事故類型'].value_counts(dropna=False))

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    analyze_zero_severity_detailed()
