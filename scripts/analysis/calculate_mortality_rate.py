import pandas as pd
import sys

# データを読み込む
csv_path = 'data/raw/honhyo_all_shishasuu_binary.csv'
print(f"データ読み込み中: {csv_path}\n")

try:
    df = pd.read_csv(csv_path)
    
    # 基本情報を表示
    print("=" * 60)
    print("交通事故データの死亡率分析")
    print("=" * 60)
    
    total_accidents = len(df)
    print(f"\n総事故件数: {total_accidents:,} 件")
    
    # 列名を確認
    print(f"\n利用可能な列名:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    # 死者数の列を探す
    death_column = None
    for col in df.columns:
        if '死者' in col or 'shishasuu' in col.lower() or 'sishasuu' in col.lower():
            death_column = col
            break
    
    if death_column:
        print(f"\n使用する列: '{death_column}'")
        print(f"\n値の分布:")
        print(df[death_column].value_counts().sort_index())
        
        # 死亡事故の件数を計算（死者数が1以上）
        death_accidents = (df[death_column] >= 1).sum()
        mortality_rate = (death_accidents / total_accidents) * 100
        
        # 総死者数を計算
        total_deaths = df[death_column].sum()
        
        print(f"\n" + "=" * 60)
        print("計算結果")
        print("=" * 60)
        print(f"死亡事故件数: {death_accidents:,} 件")
        print(f"総死者数: {int(total_deaths):,} 人")
        print(f"死亡率（死亡事故の割合）: {mortality_rate:.4f}%")
        print(f"  → 約 {mortality_rate:.2f}%")
        print("=" * 60)
        
    else:
        print("\n警告: 死者数に関連する列が見つかりませんでした")
        print("最初の5行を表示:")
        print(df.head())
        
except FileNotFoundError:
    print(f"エラー: ファイル '{csv_path}' が見つかりません")
    sys.exit(1)
except Exception as e:
    print(f"エラーが発生しました: {e}")
    sys.exit(1)
