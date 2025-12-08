
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import os
import gc
from itertools import combinations

def estimate_cost():
    input_file = r"c:\Users\socce\software-lab\traffic-accident\data\processed\honhyo_clean_predictable_only.csv"
    
    print("データを読み込んでいます（ヘッダー確認用）...")
    # 全体の行数確認
    total_rows = 1895275 # 既知の値
    
    # カテゴリカル変数リスト（既知）
    cat_cols = [
        'ゾーン規制', '一時停止規制　標識（当事者A）', '一時停止規制　標識（当事者B）',
        '一時停止規制　表示（当事者A）', '一時停止規制　表示（当事者B）', '中央分離帯施設等',
        '事故類型', '信号機', '地形', '天候', '昼夜', '曜日(発生年月日)', '祝日(発生年月日)',
        '歩車道区分', '路面状態', '車道幅員', '道路形状', '道路線形', '路線コード',
        '年齢（当事者A）', '当事者種別（当事者A）', '用途別（当事者A）',
        '速度規制（指定のみ）（当事者A）', '速度規制（指定のみ）（当事者B）',
        '市区町村コード', '警察署等コード', '都道府県コード'
    ]
    
    # 組み合わせ数の計算
    n_cats = len(cat_cols)
    n_combinations = n_cats * (n_cats - 1) // 2
    print(f"\nカテゴリカル変数: {n_cats}個")
    print(f"生成される組み合わせ特徴量: {n_combinations}個")
    print(f"合計特徴量数（予測）: {len(pd.read_csv(input_file, nrows=1).columns) - 1 + n_combinations}個")

    # サンプリングデータでのベンチマーク
    sample_size = 100000 # 10万件でテスト
    print(f"\nベンチマーク開始（サンプルサイズ: {sample_size:,}行）")
    
    start_load = time.time()
    df_sample = pd.read_csv(input_file, nrows=sample_size)
    load_time = time.time() - start_load
    
    # カテゴリカル化
    for col in df_sample.columns:
        if df_sample[col].dtype == 'object':
             df_sample[col] = df_sample[col].astype('category')
        elif col in cat_cols:
             df_sample[col] = df_sample[col].astype('category')
             
    target_col = '死者数'
    X_base = df_sample.drop(columns=[target_col])
    y_sample = df_sample[target_col]
    
    # 1. 特徴量生成時間の計測
    print("1. 特徴量生成時間の計測中...")
    start_gen = time.time()
    
    # 文字列として結合してカテゴリカル化するコストを計測
    # メモリ節約のため、実際には数個だけ作って平均を取る
    test_pairs = list(combinations(cat_cols, 2))[:50] # 最初の50ペアだけテスト
    
    temp_cols = []
    for col1, col2 in test_pairs:
        # 文字列結合コスト
        new_col = df_sample[col1].astype(str) + "_" + df_sample[col2].astype(str)
        temp_cols.append(new_col.astype('category'))
        
    gen_time_50 = time.time() - start_gen
    estimated_gen_time_full_sample = gen_time_50 * (n_combinations / 50)
    print(f"  - 50ペア生成時間 (10万行): {gen_time_50:.4f}秒")
    print(f"  -> 全{n_combinations}ペア生成予測 (10万行): {estimated_gen_time_full_sample:.4f}秒")
    
    # 2. 学習時間の計測（ベースライン）
    print("2. 学習時間の計測（特徴量追加前）...")
    params = {'objective': 'binary', 'verbose': -1, 'num_leaves': 31}
    train_data = lgb.Dataset(X_base, label=y_sample)
    
    start_train_base = time.time()
    lgb.train(params, train_data, num_boost_round=10) # 10ラウンドだけ
    train_time_base = time.time() - start_train_base
    print(f"  - ベース学習時間 (10万行, 10rounds): {train_time_base:.4f}秒")
    
    # 3. 学習時間の計測（特徴量追加後）
    print("3. 学習時間の計測（特徴量追加後予測）...")
    # 実際に350カラム追加するとメモリがきついかもしれないので、
    # 50カラム追加時の増加分から外挿する
    
    X_added = X_base.copy()
    for i, col_data in enumerate(temp_cols):
        X_added[f"no_meaning_{i}"] = col_data
        
    train_data_added = lgb.Dataset(X_added, label=y_sample)
    start_train_added = time.time()
    lgb.train(params, train_data_added, num_boost_round=10)
    train_time_added = time.time() - start_train_added
    
    diff_time = train_time_added - train_time_base
    estimated_train_time_full = train_time_base + diff_time * (n_combinations / 50)
    
    print(f"  - 50特徴量追加学習時間: {train_time_added:.4f}秒")
    print(f"  -> 全{n_combinations}特徴量追加学習予測: {estimated_train_time_full:.4f}秒")
    
    # 全体予測への拡張
    scale_factor = total_rows / sample_size
    
    print("\n" + "="*50)
    print("予測結果 (全190万件)")
    print("="*50)
    
    total_gen_time = estimated_gen_time_full_sample * scale_factor
    total_train_time_1000rounds = estimated_train_time_full * scale_factor * (1000 / 10) # 1000 rounds想定
    cv_5_time = total_train_time_1000rounds * 5
    
    print(f"データ生成時間: 約 {total_gen_time/60:.1f} 分")
    print(f"学習時間 (1 model, 1000 rounds): 約 {total_train_time_1000rounds/60:.1f} 分")
    print(f"5-Fold CV 合計時間: 約 {cv_5_time/60:.1f} 分")
    print(f"合計所要時間: 約 {(total_gen_time + cv_5_time)/60:.1f} 分")
    
    # メモリ予測
    estimated_memory = (total_rows * n_combinations * 4) / (1024**3) # int32 (4bytes) と仮定
    print(f"\n必要メモリ増加予測: 約 {estimated_memory:.1f} GB (特徴量のみ)")
    print("注意: Pandas Object型での文字列処理中はさらに数倍のメモリが必要です。")

if __name__ == "__main__":
    estimate_cost()
