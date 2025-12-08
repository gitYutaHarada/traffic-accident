"""
データセットの変数型を確認し、カテゴリカル変数と数値変数のリストを生成するスクリプト
"""

import pandas as pd
import numpy as np

def analyze_variable_types():
    # データ読み込み
    print("データを読み込んでいます...")
    df = pd.read_csv(r"c:\Users\socce\software-lab\traffic-accident\data\processed\honhyo_clean_predictable_only.csv")
    
    print(f"\nデータ形状: {df.shape}")
    print(f"カラム数: {len(df.columns)}")
    
    # 目的変数
    target = '死者数'
    
    # コードブックに基づくカテゴリカル変数（順序通り）
    categorical_features = [
        'ゾーン規制',
        '一時停止規制　標識（当事者A）',
        '一時停止規制　標識（当事者B）',
        '一時停止規制　表示（当事者A）',
        '一時停止規制　表示（当事者B）',
        '中央分離帯施設等',
        '事故類型',
        '信号機',
        '地形',
        '天候',
        '昼夜',
        '曜日(発生年月日)',
        '祝日(発生年月日)',
        '歩車道区分',
        '路面状態',
        '車道幅員',
        '道路形状',
        '道路線形',
        '路線コード',
        '年齢（当事者A）',
        '当事者種別（当事者A）',
        '用途別（当事者A）',
        '速度規制（指定のみ）（当事者A）',
        '速度規制（指定のみ）（当事者B）',
        '市区町村コード',
        '警察署等コード',
        '都道府県コード',
    ]
    
    # 数値変数
    numerical_features = [
        '地点　経度（東経）',
        '地点　緯度（北緯）',
        '地点コード',
    ]
    
    # その他（要変換）
    other_features = [
        '発生日時',
        '衝突地点',  # これもカテゴリカルの可能性
    ]
    
    print("\n" + "="*80)
    print("カテゴリカル変数の確認")
    print("="*80)
    
    for i, col in enumerate(categorical_features, 1):
        if col in df.columns:
            unique_count = df[col].nunique()
            sample_values = df[col].value_counts().head(5)
            print(f"\n{i}. {col}")
            print(f"   ユニーク値数: {unique_count}")
            print(f"   上位5値:\n{sample_values.to_string()}")
        else:
            print(f"\n{i}. {col} - ⚠️ カラムが見つかりません")
    
    print("\n" + "="*80)
    print("数値変数の確認")
    print("="*80)
    
    for i, col in enumerate(numerical_features, 1):
        if col in df.columns:
            print(f"\n{i}. {col}")
            print(f"   型: {df[col].dtype}")
            print(f"   統計:\n{df[col].describe()}")
        else:
            print(f"\n{i}. {col} - ⚠️ カラムが見つかりません")
    
    print("\n" + "="*80)
    print("その他の変数")
    print("="*80)
    
    for col in other_features:
        if col in df.columns:
            print(f"\n{col}")
            print(f"   型: {df[col].dtype}")
            print(f"   サンプル: {df[col].head(3).tolist()}")
    
    print("\n" + "="*80)
    print("目的変数")
    print("="*80)
    print(f"\n{target}")
    print(f"分布:\n{df[target].value_counts()}")
    print(f"比率:\n{df[target].value_counts(normalize=True)}")
    
    # LightGBM用のカテゴリカル変数インデックスを生成
    print("\n" + "="*80)
    print("LightGBM用カテゴリカル変数インデックス")
    print("="*80)
    
    categorical_indices = []
    for col in categorical_features:
        if col in df.columns:
            idx = df.columns.get_loc(col)
            categorical_indices.append(idx)
            print(f"{col}: インデックス {idx}")
    
    print(f"\n合計: {len(categorical_indices)}個のカテゴリカル変数")
    print(f"インデックスリスト: {categorical_indices}")
    
    # サマリー
    print("\n" + "="*80)
    print("サマリー")
    print("="*80)
    print(f"全カラム数: {len(df.columns)}")
    print(f"カテゴリカル変数: {len(categorical_features)}")
    print(f"数値変数: {len(numerical_features)}")
    print(f"その他: {len(other_features)}")
    print(f"目的変数: 1 ({target})")

if __name__ == "__main__":
    analyze_variable_types()
