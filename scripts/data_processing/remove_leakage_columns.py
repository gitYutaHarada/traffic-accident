
import pandas as pd
import os
import sys

def main():
    input_file = r"c:\Users\socce\software-lab\traffic-accident\data\processed\honhyo_clean_predictable_only.csv"
    output_file = r"c:\Users\socce\software-lab\traffic-accident\data\processed\honhyo_clean_no_leakage.csv"
    
    print(f"Loading data: {input_file}")
    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")
    
    # 削除対象カラム
    # TODO: 他にもあればここに追加
    columns_to_remove = ['事故類型']
    
    # 削除前の確認
    for col in columns_to_remove:
        if col in df.columns:
            print(f"Found column to remove: {col}")
        else:
            print(f"Warning: Column {col} not found in dataset!")
            
    # 削除実行
    print("Removing columns...")
    df_clean = df.drop(columns=columns_to_remove, errors='ignore')
    
    # 厳格な削除確認 (Assertion)
    print("Verifying removal...")
    for col in columns_to_remove:
        if col in df_clean.columns:
            print(f"CRITICAL ERROR: Column {col} was NOT removed!")
            sys.exit(1)
        else:
            print(f"Verified: {col} is gone.")
            
    # 結果確認
    print(f"New shape: {df_clean.shape}")
    print(f"Removed columns count: {len(df.columns) - len(df_clean.columns)}")
    
    # 保存
    print(f"Saving to: {output_file}")
    df_clean.to_csv(output_file, index=False)
    print("Success!")

if __name__ == "__main__":
    main()
