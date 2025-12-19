"""
特徴量エンジニアリング強化: Interaction Features の生成

honhyo_clean_with_features.csv に対して、当事者間・道路環境の相互作用特徴量を追加し、
honhyo_with_interactions.csv として出力する。

修正履歴:
- is_night の判定を str.contains('2') から isin([21,22,23]) に修正（12=昼昼の誤判定回避）
- is_safe_night_urban に「単路(14)」条件を追加してHard FP Cluster 0を狙い撃ち
- is_night_truck（SHAP Top1の危険フラグ）を追加
"""

import pandas as pd
import os
import argparse
import numpy as np

def main(input_path: str, output_path: str):
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # === A. 当事者間の関係性 (Party Interactions) ===
    print("\n--- Creating Party Interaction Features ---")
    
    # A1. 一時停止規制の相互作用
    if '一時停止規制　標識（当事者A）' in df.columns and '一時停止規制　標識（当事者B）' in df.columns:
        a_stop = (df['一時停止規制　標識（当事者A）'] != 0).astype(int)
        b_stop = (df['一時停止規制　標識（当事者B）'] != 0).astype(int)
        df['stop_sign_interaction'] = a_stop * 2 + b_stop
        print(f"  Created 'stop_sign_interaction'.")
    
    # A2. 速度規制の差分・比率
    if '速度規制（指定のみ）（当事者A）' in df.columns and '速度規制（指定のみ）（当事者B）' in df.columns:
        df['speed_reg_diff'] = df['速度規制（指定のみ）（当事者A）'] - df['速度規制（指定のみ）（当事者B）']
        df['speed_reg_diff_abs'] = df['speed_reg_diff'].abs()
        
        # Bが脆弱（歩行者等）かもしれないフラグ (0=対象外)
        df['maybe_vulnerable_victim'] = ((df['速度規制（指定のみ）（当事者B）'] == 0) & (df['速度規制（指定のみ）（当事者A）'] > 0)).astype(int)
        print(f"  Created 'speed_reg_diff', 'speed_reg_diff_abs', 'maybe_vulnerable_victim'.")

    # === B. 道路環境の複合リスク (Environment Interactions) ===
    print("\n--- Creating Environment Interaction Features ---")
    
    # 文字列結合系の特徴量作成関数
    def create_str_interaction(col1, col2, new_col_name):
        if col1 in df.columns and col2 in df.columns:
            df[new_col_name] = df[col1].astype(str) + '_' + df[col2].astype(str)
            print(f"  Created '{new_col_name}'.")

    create_str_interaction('昼夜', '地形', 'night_terrain')
    create_str_interaction('道路形状', '地形', 'road_shape_terrain') # SHAP重要度高: トンネルx非市街地
    create_str_interaction('信号機', '道路形状', 'signal_road_shape')
    create_str_interaction('昼夜', '路面状態', 'night_road_condition')
    create_str_interaction('速度規制（指定のみ）（当事者A）', '道路形状', 'speed_shape_interaction')

    # === C. 追加の有用な相互作用 ===
    print("\n--- Creating Additional Interaction Features ---")
    
    create_str_interaction('当事者種別（当事者A）', '昼夜', 'party_type_daytime') # SHAP 1位
    create_str_interaction('当事者種別（当事者A）', '道路形状', 'party_type_road_shape')

    # === D. Antidote & Risk Features (SHAP/Hard FP 対策) ===
    print("\n--- Creating Antidote & Risk Features ---")

    # D1. is_safe_night_urban (安全な夜の市街地 - 解毒剤)
    if '地形' in df.columns and '昼夜' in df.columns and '当事者種別（当事者A）' in df.columns and '道路形状' in df.columns:
        # 地形: 1=市街地(人口集中), 2=市街地(その他)
        is_urban = df['地形'].isin([1, 2])
        
        # ### FIX: 昼コード'12'が'2'を含むため、str.containsは危険。リストで指定。
        is_night = df['昼夜'].isin([21, 22, 23]) 
        
        # 当事者種別A: 3=乗用車
        is_car_a = df['当事者種別（当事者A）'] == 3
        
        # ### ADD: Hard FP分析で多かった「単路(14)」を追加して条件を絞る
        is_single_road = df['道路形状'] == 14
        
        df['is_safe_night_urban'] = (is_urban & is_night & is_car_a & is_single_road).astype(int)
        print(f"  Created 'is_safe_night_urban'. Count={df['is_safe_night_urban'].sum()}")
    
    # ### ADD: D4. is_night_truck (危険な夜のトラック - 強調剤)
    # SHAP Interaction分析でTop1だった「貨物車(11) x 夜(22)」を明示
    if '当事者種別（当事者A）' in df.columns and '昼夜' in df.columns:
        # 当事者種別A: 11=普通貨物車 (12,13,14も貨物だがTop1は11)
        # 昼夜: 22=夜-夜
        df['is_night_truck'] = ((df['当事者種別（当事者A）'] == 11) & (df['昼夜'] == 22)).astype(int)
        print(f"  Created 'is_night_truck'. Count={df['is_night_truck'].sum()}")

    # D2. midnight_activity_flag (深夜活動リスク)
    if 'hour' in df.columns:
        df['midnight_activity_flag'] = df['hour'].apply(lambda x: 1 if (x >= 22 or x <= 4) else 0)
        print(f"  Created 'midnight_activity_flag'.")

    # D3. intersection_safety (交差点の安全性)
    if '信号機' in df.columns:
        # 信号あり(7以外) かつ 交差点(道路形状が単路系以外) というニュアンス
        # ここではシンプルに「信号機がある場所」を安全側特徴量とする
        has_signal = df['信号機'] != 7
        df['intersection_with_signal'] = has_signal.astype(int)
        print(f"  Created 'intersection_with_signal'.")
        
    # === 保存処理 ===
    print(f"\n--- Saving to {output_path} ---")
    
    # カテゴリ変換 (容量削減)
    interaction_cols = [c for c in df.columns if '_' in c and c not in pd.read_csv(input_path, nrows=0).columns]
    for col in interaction_cols:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved: {df.shape[0]} rows, {df.shape[1]} columns")
    
    new_cols = [c for c in df.columns if c not in pd.read_csv(input_path, nrows=0).columns]
    print(f"\nTotal new features added ({len(new_cols)}):")
    for c in new_cols:
        print(f"  - {c}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create interaction features.")
    parser.add_argument("--input", type=str, default="data/processed/honhyo_clean_with_features.csv")
    parser.add_argument("--output", type=str, default="data/processed/honhyo_with_interactions.csv")
    args = parser.parse_args()
    
    main(args.input, args.output)
