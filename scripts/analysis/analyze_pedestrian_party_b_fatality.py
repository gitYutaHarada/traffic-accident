import pandas as pd
from pathlib import Path
import sys

# Windows environments encoding
sys.stdout.reconfigure(encoding='utf-8')

# Path to the dataset
DATA_PATH = Path('c:/Users/socce/software-lab/traffic-accident/data/raw/honhyo_all_shishasuu_binary.csv')

def analyze_party_b_pedestrian_fatality():
    if not DATA_PATH.exists():
        print(f"Error: File not found at {DATA_PATH}")
        return

    print("Loading dataset...")
    try:
        # Load necessary columns
        usecols = ['当事者種別（当事者B）', '人身損傷程度（当事者B）', '死者数']
        df = pd.read_csv(DATA_PATH, usecols=lambda c: c in usecols)
        
        target_col = '当事者種別（当事者B）'
        severity_col = '人身損傷程度（当事者B）'
        
        # Pedestrian Codes: 61 only
        df_ped = df[df[target_col] == 61]
        
        print(f"\nTotal Records with Party B as Pedestrian (61): {len(df_ped)}")
        
        # Fatality Check
        # Code 1: Death
        fatal_ped = df_ped[df_ped[severity_col] == 1]
        
        print(f"\nFatality Count (Party B Severity = 1): {len(fatal_ped)}")
        
        if len(df_ped) > 0:
            rate = len(fatal_ped) / len(df_ped) * 100
            print(f"Fatality Rate: {rate:.2f}%")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    analyze_party_b_pedestrian_fatality()
