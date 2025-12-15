"""
honhyo_clean_road_type.csv に派生特徴量（area_id, month, day, hour）を追加して保存するスクリプト
"""
import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path

# パス設定
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # scripts/data_processing -> scripts -> プロジェクトルート
INPUT_FILE = BASE_DIR / "data" / "processed" / "honhyo_clean_road_type.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "honhyo_clean_with_features.csv"

N_CLUSTERS = 50  # エリアクラスタ数

print("=" * 80)
print("派生特徴量付きデータセット作成スクリプト")
print("=" * 80)

# データ読み込み
print(f"\n[1/5] データ読み込み中: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"✅ 読み込み完了: {len(df):,}件, {len(df.columns)}列")

# 緯度経度の変換関数
def convert_dms_to_deg(v):
    """度分秒形式（DDDMMSSsss）から10進数の度に変換"""
    v = np.where(v == 0, np.nan, v)
    ms = v % 1000
    v = v // 1000
    ss = v % 100
    v = v // 100
    mm = v % 100
    dd = v // 100
    return dd + mm/60 + (ss + ms/1000)/3600

# 緯度経度の変換
print("\n[2/5] 緯度経度を10進数に変換中...")
lon_col = '地点　経度（東経）'
lat_col = '地点　緯度（北緯）'

lon_val = df[lon_col].fillna(0).astype(np.int64)
lat_val = df[lat_col].fillna(0).astype(np.int64)

df['lon_deg'] = convert_dms_to_deg(lon_val)
df['lat_deg'] = convert_dms_to_deg(lat_val)
print("✅ 変換完了")

# KMeansクラスタリングでエリアID作成
print(f"\n[3/5] KMeansクラスタリング (n_clusters={N_CLUSTERS})...")
valid_coords = df[['lon_deg', 'lat_deg']].dropna()

# 初期値を-1に設定
df['area_id'] = -1

if len(valid_coords) > 0:
    kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, batch_size=4096, n_init=10)
    cluster_labels = kmeans.fit_predict(valid_coords)
    df.loc[valid_coords.index, 'area_id'] = cluster_labels

print(f"✅ area_id 作成完了 (ユニーク値: {df['area_id'].nunique()})")

# 日時分解
print("\n[4/5] 発生日時から year, month, day, hour を抽出中...")
df['発生日時'] = pd.to_datetime(df['発生日時'], errors='coerce')
df['year'] = df['発生日時'].dt.year
df['month'] = df['発生日時'].dt.month
df['day'] = df['発生日時'].dt.day
df['hour'] = df['発生日時'].dt.hour
print("✅ 日時分解完了")

# 不要列の削除（発生日時、緯度経度の中間変数）
print("\n[5/5] 不要列を削除中...")
drop_cols = ['lon_deg', 'lat_deg', '発生日時']
df = df.drop(columns=drop_cols)
print(f"✅ 削除完了: {drop_cols}")

# 保存
print(f"\n[保存] {OUTPUT_FILE}")
df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"✅ 保存完了: {len(df):,}件, {len(df.columns)}列")

# 新しい列の確認
print("\n" + "=" * 80)
print("追加された列:")
print("=" * 80)
for col in ['area_id', 'year', 'month', 'day', 'hour']:
    if col in df.columns:
        print(f"  - {col}: ユニーク値 {df[col].nunique()}, 欠損値 {df[col].isna().sum()}")

print("\n✅ 完了!")
