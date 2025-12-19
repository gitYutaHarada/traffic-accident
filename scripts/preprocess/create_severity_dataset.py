import pandas as pd

from pathlib import Path
import sys

# Windows environment encoding
sys.stdout.reconfigure(encoding='utf-8')

# Paths
PROCESS_DIR = Path('c:/Users/socce/software-lab/traffic-accident/data/processed')
RAW_DIR = Path('c:/Users/socce/software-lab/traffic-accident/data/raw')
INPUT_CLEAN_PATH = PROCESS_DIR / 'honhyo_clean_with_features.csv'
INPUT_RAW_PATH = RAW_DIR / 'honhyo_all_shishasuu_binary.csv'
OUTPUT_PATH = PROCESS_DIR / 'honhyo_severity_classification.csv'

def create_severity_dataset():
    print(f"Loading clean dataset from {INPUT_CLEAN_PATH}...")
    if not INPUT_CLEAN_PATH.exists():
        print(f"Error: Clean file not found at {INPUT_CLEAN_PATH}")
        return
    df_clean = pd.read_csv(INPUT_CLEAN_PATH)
    
    print(f"Loading raw dataset from {INPUT_RAW_PATH} (only specific columns)...")
    if not INPUT_RAW_PATH.exists():
        print(f"Error: Raw file not found at {INPUT_RAW_PATH}")
        return
        
    # Read raw columns needed for calculation + alignment check
    raw_cols = ['人身損傷程度（当事者A）', '人身損傷程度（当事者B）', '死者数']
    df_raw = pd.read_csv(INPUT_RAW_PATH, usecols=raw_cols)

    # 1. Sanity Check: Row Count
    if len(df_clean) != len(df_raw):
        print(f"Error: Row count mismatch! Clean: {len(df_clean)}, Raw: {len(df_raw)}")
        return
    
    # 2. Sanity Check: Alignment (Death Count)
    # Note: '死者数' should exist in clean. If clean has renamed it (e.g., 'target'), check that.
    # Based on previous check, '死者数' likely exists or 'death_count'. 
    # Let's check available columns in clean briefly if needed, but assuming '死者数' based on prior context.
    
    clean_death_col = '死者数'
    if clean_death_col not in df_clean.columns:
        # Try to find a 'death' related column or assume 'target'
        candidates = [c for c in df_clean.columns if '死者' in c or 'target' in c]
        if candidates:
            clean_death_col = candidates[0]
            print(f"Warning: '死者数' not found, using '{clean_death_col}' for alignment check.")
        else:
            print("Warning: Cannot perform alignment check (death count column missing in clean). proceeding with index match.")
            clean_death_col = None

    if clean_death_col:
        # Check alignment. Note: Raw might be int, clean might be int/float.
        # Clean might have binary 0/1 for death, Raw has actual count 0,1,2...
        # If clean is binary classifier target, it would be 0 or 1.
        # Raw '死者数' > 0 should equal Clean 'Death' == 1.
        
        # Binary conversion for check
        raw_binary = (df_raw['死者数'] > 0).astype(int)
        clean_binary = (df_clean[clean_death_col] > 0).astype(int)
        
        match_rate = (raw_binary == clean_binary).mean()
        print(f"Alignment Check (Death vs Target): {match_rate*100:.4f}% match")
        
        if match_rate < 0.99:
             print("Error: Datasets do not appear to be aligned! Aborting.")
             return

    # 3. Calculation
    print("Calculating 'accident_severity'...")
    
    col_a = '人身損傷程度（当事者A）'
    col_b = '人身損傷程度（当事者B）'
    
    # Logic for Ordinal Regression:
    # 2: Fatal (Most Severe) -> One of them is 1
    # 1: Injury (Medium) -> One of them is 2, and neither is 1
    # 0: No Injury (Least Severe) -> Neither are 1 or 2
    
    # Combine from raw
    severity_a = df_raw[col_a]
    severity_b = df_raw[col_b]
    
    # Create pandas Series
    # Initialize with 0 (No Injury)
    new_severity = pd.Series(0, index=df_clean.index, name='accident_severity', dtype=int)
    
    # Condition masks
    cond_fatal = (severity_a == 1) | (severity_b == 1)
    cond_injury = (severity_a == 2) | (severity_b == 2)
    
    # Apply Priority: Fatal overrides Injury
    new_severity[cond_injury] = 1
    new_severity[cond_fatal] = 2 # Overwrite injury if fatal
    
    # Assign to clean dataframe
    df_clean['accident_severity'] = new_severity
    
    # 4. Distribution Check
    print("\n--- New Target Distribution (accident_severity) ---")
    print(df_clean['accident_severity'].value_counts().sort_index())
    
    # 5. Drop old target
    if clean_death_col:
        print(f"\nDropping original target '{clean_death_col}'...")
        df_clean.drop(columns=[clean_death_col], inplace=True)
        
    # 6. Save
    print(f"\nSaving to {OUTPUT_PATH}...")
    df_clean.to_csv(OUTPUT_PATH, index=False)
    print("Done.")

if __name__ == "__main__":
    create_severity_dataset()
