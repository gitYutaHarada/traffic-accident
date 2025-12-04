import pandas as pd
import numpy as np

def main():
    """
    事故発生前に観測可能な特徴のみを抽出したデータセットを作成
    """
    
    # 事前観測可能な特徴リスト（33項目）
    preaccident_features = [
        # 時間・場所情報（15項目）
        '発生日時　　年',
        '発生日時　　月',
        '発生日時　　日',
        '発生日時　　時',
        '発生日時　　分',
        '曜日(発生年月日)',
        '祝日(発生年月日)',
        '昼夜',
        '都道府県コード',
        '市区町村コード',
        '警察署等コード',
        '地点コード',
        '路線コード',
        '地点　緯度（北緯）',
        '地点　経度（東経）',
        
        # 環境情報（5項目）
        '天候',
        '地形',
        '路面状態',
        '道路形状',
        '道路線形',
        
        # 道路設備情報（8項目）
        '信号機',
        '一時停止規制　標識（当事者A）',
        '一時停止規制　表示（当事者A）',
        'ゾーン規制',
        '中央分離帯施設等',
        '歩車道区分',
        '車道幅員',
        '速度規制（指定のみ）（当事者A）',
        
        # 自車（当事者A）情報（5項目）
        '当事者種別（当事者A）',
        '年齢（当事者A）',
        '用途別（当事者A）',
        'エアバッグの装備（当事者A）',
        'サイドエアバッグの装備（当事者A）',
    ]
    
    # 目的変数
    target = '死者数'
    
    print("="*60)
    print("事故予防モデル用データセット作成")
    print("="*60)
    print()
    
    # データ読み込み
    input_file = r'C:\Users\socce\software-lab\traffic-accident\data\raw\honhyo_all_shishasuu_binary.csv'
    print(f"データを読み込んでいます: {input_file}")
    
    try:
        df = pd.read_csv(input_file, encoding='cp932')
    except UnicodeDecodeError:
        print("cp932での読み込みに失敗しました。utf-8で試行します。")
        df = pd.read_csv(input_file, encoding='utf-8')
    
    print(f"✓ 読み込み完了: {len(df):,} 行")
    print(f"✓ 元のデータ列数: {len(df.columns)} 列")
    print()
    
    # 利用可能な列を確認
    available_features = [col for col in preaccident_features if col in df.columns]
    missing_features = [col for col in preaccident_features if col not in df.columns]
    
    if missing_features:
        print("⚠️ 以下の列が見つかりませんでした:")
        for col in missing_features:
            print(f"  - {col}")
        print()
    
    # 事前観測可能特徴 + 目的変数を抽出
    columns_to_keep = available_features + [target]
    df_preaccident = df[columns_to_keep].copy()
    
    print(f"✓ 抽出した列数: {len(df_preaccident.columns)} 列")
    print(f"  - 事前観測可能特徴: {len(available_features)} 列")
    print(f"  - 目的変数: 1 列 ({target})")
    print()
    
    # 統計情報
    print("【データ統計】")
    print(f"総データ数: {len(df_preaccident):,} 件")
    print(f"死亡事故数: {(df_preaccident[target] == 1).sum():,} 件")
    print(f"死亡事故率: {(df_preaccident[target] == 1).mean() * 100:.2f}%")
    print()
    
    # 保存
    output_file = r'C:\Users\socce\software-lab\traffic-accident\data\processed\honhyo_all_preaccident_only.csv'
    print(f"データを保存しています: {output_file}")
    df_preaccident.to_csv(output_file, index=False, encoding='utf-8-sig')
    print("✓ 保存完了")
    print()
    
    print("="*60)
    print("処理完了！")
    print("="*60)
    print()
    print("【抽出された特徴一覧】")
    for i, col in enumerate(available_features, 1):
        print(f"{i:2d}. {col}")

if __name__ == "__main__":
    main()

