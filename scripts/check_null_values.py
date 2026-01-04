import pandas as pd
import numpy as np

def check_nulls():
    print("Loading data...")
    train = pd.read_parquet('data/spatio_temporal/raw_train.parquet')
    val = pd.read_parquet('data/spatio_temporal/raw_val.parquet')
    test = pd.read_parquet('data/spatio_temporal/raw_test.parquet')

    # Combine for analysis, similar to how they are treated in the pipeline (or analysis of total dataset)
    # The user asked about "how much this processing was actually performed" so checking the raw input is key.
    # We can check per split.
    
    datasets = {
        'Train': train,
        'Val': val,
        'Test': test
    }
    
    # Logic from TwoStageSpatioTemporalEnsemble._identify_columns
    target_col = "fatal"
    exclude_cols = [
        target_col, '死者数', '負傷者数', '重傷者数', '軽傷者数',
        '当事者A_死傷状況', '当事者B_死傷状況', '本票番号', '発生日時',
        'lat', 'lon', 'geohash', 'geohash_fine', 'date', 'year',
        'accident_id', 'original_index'
    ]
    
    # Identify columns based on full training set (Train + Val) as in the script
    full_train_df = pd.concat([train, val], ignore_index=True)
    available_cols = [c for c in full_train_df.columns if c not in exclude_cols]
    
    cat_cols = []
    num_cols = []
    
    for col in available_cols:
        if full_train_df[col].dtype == 'object':
            cat_cols.append(col)
        elif full_train_df[col].nunique() < 50 and full_train_df[col].dtype in ['int64', 'int32']:
            cat_cols.append(col)
        else:
            num_cols.append(col)
            
    print(f"\nIdentified {len(num_cols)} Numerical columns and {len(cat_cols)} Categorical columns.")
    
    total_missing_num_count = 0
    total_num_cells = 0
    total_missing_cat_count = 0
    total_cat_cells = 0
    
    print("\n=== Missing Value Statistics ===")
    
    for name, df in datasets.items():
        print(f"\n--- {name} Set ({len(df)} rows) ---")
        
        # Numerical
        num_missing = df[num_cols].isnull().sum().sum()
        num_cells = df[num_cols].size
        total_missing_num_count += num_missing
        total_num_cells += num_cells
        
        print(f"Numerical Missing: {num_missing:,} / {num_cells:,} ({num_missing/num_cells*100:.4f}%)")
        if num_missing > 0:
            print("  Columns with missing values (Num):")
            missing_cols = df[num_cols].isnull().sum()
            print(missing_cols[missing_cols > 0])

        # Categorical
        # Note: In the training script, they do: X[col].astype(str).fillna('missing')
        # So we should count actual NaNs.
        cat_missing = df[cat_cols].isnull().sum().sum()
        cat_cells = df[cat_cols].size
        total_missing_cat_count += cat_missing
        total_cat_cells += cat_cells

        print(f"Categorical Missing: {cat_missing:,} / {cat_cells:,} ({cat_missing/cat_cells*100:.4f}%)")
        if cat_missing > 0:
            print("  Columns with missing values (Cat):")
            missing_cols = df[cat_cols].isnull().sum()
            print(missing_cols[missing_cols > 0])

    print("\n=== Overall Summary ===")
    print(f"Total Numerical Missing: {total_missing_num_count:,} / {total_num_cells:,} ({total_missing_num_count/total_num_cells*100:.6f}%)")
    print(f"Total Categorical Missing: {total_missing_cat_count:,} / {total_cat_cells:,} ({total_missing_cat_count/total_cat_cells*100:.6f}%)")

if __name__ == "__main__":
    check_nulls()
