import pandas as pd
from pathlib import Path
import sys

# Windows environments encoding
sys.stdout.reconfigure(encoding='utf-8')

# Path to the dataset
# Using the raw one to ensure we have the original codes
DATA_PATH = Path('c:/Users/socce/software-lab/traffic-accident/data/raw/honhyo_all_shishasuu_binary.csv')

def analyze_party_a_pedestrian_fatality():
    if not DATA_PATH.exists():
        print(f"Error: File not found at {DATA_PATH}")
        return

    print("Loading dataset...")
    try:
        # Load necessary columns
        usecols = ['当事者種別（当事者A）', '人身損傷程度（当事者A）', '死者数']
        df = pd.read_csv(DATA_PATH, usecols=lambda c: c in usecols)
        
        target_col = '当事者種別（当事者A）'
        severity_col = '人身損傷程度（当事者A）'
        
        # Pedestrian Codes: 61 only (since we found others were 0)
        df_ped = df[df[target_col] == 61]
        
        print(f"\nTotal Records with Party A as Pedestrian (61): {len(df_ped)}")
        
        # Fatality Check
        # Code 1: Death
        fatal_ped = df_ped[df_ped[severity_col] == 1]
        
        print(f"\nFatality Count (Party A Severity = 1): {len(fatal_ped)}")
        
        if len(fatal_ped) > 0:
            print("\nVerification check (Death count column for these rows):")
            print(fatal_ped['死者数'].value_counts())

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    analyze_party_a_pedestrian_fatality()
