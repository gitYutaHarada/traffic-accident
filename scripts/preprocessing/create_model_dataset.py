import pandas as pd
import os

def main():
    """
    モデル学習用のデータセットを作成するスクリプト
    主な処理:
    1. Rawデータの読み込み
    2. 日時情報の分解（月、時、曜日）
    3. 加工済みデータの保存
    """
    print("=" * 80)
    print("データ前処理: 日時情報の分解とデータセット作成")
    print("=" * 80)

    # 入力と出力のパス
    input_path = 'data/raw/honhyo_all_shishasuu_binary.csv'
    output_dir = 'data/processed'
    output_file = 'honhyo_model_ready.csv'
    output_path = os.path.join(output_dir, output_file)
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📂 Rawデータ読み込み中: {input_path}")
    try:
        df = pd.read_csv(input_path)
        print(f"✓ 読み込み完了: {len(df):,} 件")
    except Exception as e:
        print(f"❌ エラー: {e}")
        return

    # 日時情報の処理
    print("\n📅 日時情報の処理中...")
    
    # 既存のカラム名（全角スペースが含まれていることに注意）
    # 実際のカラム名: '発生日時　　月', '発生日時　　時', '曜日(発生年月日)'
    
    rename_map = {
        '発生日時　　月': '発生月',
        '発生日時　　時': '発生時',
        '曜日(発生年月日)': '曜日',
        '発生日時　　年': '発生年'
    }
    
    # マッピング対象が存在するか確認
    available_cols = set(df.columns)
    valid_rename = {k: v for k, v in rename_map.items() if k in available_cols}
    
    if valid_rename:
        df = df.rename(columns=valid_rename)
        print(f"  + カラム名を変更しました: {valid_rename}")
    else:
        print("⚠️ 警告: 日時関連のカラムが見つかりませんでした。スキップします。")
        print(f"  存在するカラム: {list(df.columns)}")

    # 不要な日時関連カラムがあれば削除
    # ... (前回の変更内容の続き)
    
    # ---------------------------------------------------------
    # 緯度経度の処理 (Geo-Clustering)
    # ---------------------------------------------------------
    print("\n🗺️ 緯度経度の処理（エリアID化）中...")
    
    # カラム名の定義（全角スペースに注意）
    lat_col = '地点　緯度（北緯）'
    lon_col = '地点　経度（東経）'
    
    if lat_col in df.columns and lon_col in df.columns:
        from sklearn.cluster import MiniBatchKMeans
        import numpy as np

        def convert_dms_to_deg(v):
            """
            DMS形式の整数（例: 431412959）を10進数の度（Degree）に変換する
            想定形式: DDDMMSSsss (度, 分, 秒, ミリ秒)
            """
            try:
                if pd.isna(v) or v == 0:
                    return np.nan
                v = int(v)
                deg = v // 10000000
                rest = v % 10000000
                minute = rest // 100000
                second = (rest % 100000) / 1000.0
                
                return deg + (minute / 60.0) + (second / 3600.0)
            except:
                return np.nan

        print("  + 座標変換 (DMS -> Decimal Degree)...")
        # ベクトル化せずにapplyで処理（データ量的に少し重いが、複雑な演算なので安全に）
        # 高速化のため、0や欠損を除く
        
        # 一旦数値を変換用の一時カラムにする
        df['temp_lat'] = df[lat_col].apply(convert_dms_to_deg)
        df['temp_lon'] = df[lon_col].apply(convert_dms_to_deg)
        
        # 欠損値の処理（変換失敗や元々0だったもの）
        # 欠損がある行はクラスタリングできないため、全体の重心（平均）で埋めるか、除外する
        # ここでは平均値で埋める方針とする
        lat_mean = df['temp_lat'].mean()
        lon_mean = df['temp_lon'].mean()
        df['temp_lat'] = df['temp_lat'].fillna(lat_mean)
        df['temp_lon'] = df['temp_lon'].fillna(lon_mean)
        
        print(f"  + クラスタリング作成 (MiniBatchKMeans, n=500)...")
        # 緯度経度のスケールは日本国内であれば大きく違わないため、そのまま使う
        kmeans = MiniBatchKMeans(n_clusters=500, random_state=42, batch_size=4096, n_init=3)
        df['Area_Cluster_ID'] = kmeans.fit_predict(df[['temp_lat', 'temp_lon']])
        
        print("  + 'Area_Cluster_ID' カラム作成完了")
        
        # 元の緯度経度カラムと一時カラムを削除
        df = df.drop(columns=[lat_col, lon_col, 'temp_lat', 'temp_lon'])
        print(f"  - 元の緯度経度カラムを削除")
        
    else:
        print("⚠️ 警告: 緯度経度カラムが見つかりません。スキップします。")

    # 保存
    print(f"\n💾 加工済みデータを保存中: {output_path}")
    df.to_csv(output_path, index=False)
    print("✓ 保存完了")
    print("\n✅ 前処理完了")

if __name__ == "__main__":
    main()
