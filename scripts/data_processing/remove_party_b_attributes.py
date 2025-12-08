"""
当事者B（相手側）の予測不可能な属性カラムを削除するスクリプト

削除対象カラム:
- 年齢（当事者B）: 相手の年齢は予測時に不明
- 当事者種別（当事者B）: 相手の車種は予測時に不明
- 用途別（当事者B）: 相手の用途（事業用/自家用）は予測時に不明

保持するカラム（道路構造として解釈可能）:
- 一時停止規制　標識（当事者B）
- 一時停止規制　表示（当事者B）
- 速度規制（指定のみ）（当事者B）
"""

import pandas as pd
import os
from pathlib import Path

def main():
    # ファイルパス
    input_file = r"c:\Users\socce\software-lab\traffic-accident\data\processed\honhyo_clean_binary.csv"
    output_file = r"c:\Users\socce\software-lab\traffic-accident\data\processed\honhyo_clean_predictable_only.csv"
    
    # データ読み込み
    print("データを読み込んでいます...")
    df = pd.read_csv(input_file)
    print(f"元のデータ: {len(df)}行 × {len(df.columns)}列")
    
    # 削除対象カラム
    columns_to_remove = [
        '年齢（当事者B）',
        '当事者種別（当事者B）',
        '用途別（当事者B）'
    ]
    
    # 削除前の確認
    print("\n削除するカラム:")
    for col in columns_to_remove:
        if col in df.columns:
            print(f"  ✓ {col}")
        else:
            print(f"  ✗ {col} (見つかりません)")
    
    # カラム削除
    print("\nカラムを削除しています...")
    df_cleaned = df.drop(columns=columns_to_remove, errors='ignore')
    
    print(f"削除後のデータ: {len(df_cleaned)}行 × {len(df_cleaned.columns)}列")
    print(f"削除されたカラム数: {len(df.columns) - len(df_cleaned.columns)}")
    
    # 保存
    print(f"\n新しいデータセットを保存しています: {output_file}")
    df_cleaned.to_csv(output_file, index=False)
    
    # 確認
    print("\n✅ 完了！")
    print(f"\n残っている当事者B関連のカラム（道路構造として保持）:")
    party_b_cols = [col for col in df_cleaned.columns if '当事者B' in col or '当事者Ｂ' in col]
    for col in party_b_cols:
        print(f"  - {col}")
    
    # 統計情報の表示
    print(f"\n新しいデータセットの情報:")
    print(f"  ファイル名: honhyo_clean_predictable_only.csv")
    print(f"  行数: {len(df_cleaned):,}")
    print(f"  列数: {len(df_cleaned.columns)}")
    print(f"  削除された列数: {len(columns_to_remove)}")

if __name__ == "__main__":
    main()
