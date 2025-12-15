"""
地理情報クラスタリング（area_id）の可視化スクリプト
日本地図上にクラスタを色分けしてプロット
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.family'] = ['MS Gothic', 'Hiragino Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# パス設定
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "honhyo_clean_with_features.csv"
OUTPUT_DIR = BASE_DIR / "results" / "visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("地理情報クラスタリング可視化")
print("=" * 80)

# データ読み込み
print(f"\n[1/4] データ読み込み中: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"✅ 読み込み完了: {len(df):,}件")

# 緯度経度の再計算（元データから）
print("\n[2/4] 緯度経度を再計算中...")
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

lon_col = '地点　経度（東経）'
lat_col = '地点　緯度（北緯）'

if lon_col in df.columns and lat_col in df.columns:
    lon_val = df[lon_col].fillna(0).astype(np.int64)
    lat_val = df[lat_col].fillna(0).astype(np.int64)
    df['lon_deg'] = convert_dms_to_deg(lon_val)
    df['lat_deg'] = convert_dms_to_deg(lat_val)
    print("✅ 緯度経度変換完了")
else:
    print("❌ 緯度経度列が見つかりません")
    exit(1)

# 有効なデータのみ抽出
print("\n[3/4] 有効な座標データを抽出中...")
df_valid = df[['lon_deg', 'lat_deg', 'area_id']].dropna()
df_valid = df_valid[df_valid['area_id'] >= 0]  # -1（欠損）を除外
print(f"✅ 有効データ: {len(df_valid):,}件")
print(f"   クラスタ数: {df_valid['area_id'].nunique()}")

# クラスタごとの統計
print("\n[4/4] 可視化中...")
n_clusters = df_valid['area_id'].nunique()

# カラーマップ設定
cmap = plt.cm.get_cmap('tab20', n_clusters)

# ============================================
# 図1: 全体マップ（クラスタ別色分け）
# ============================================
fig1, ax1 = plt.subplots(figsize=(14, 12))

# サンプリング（描画高速化）
sample_size = min(50000, len(df_valid))
df_sample = df_valid.sample(n=sample_size, random_state=42)

scatter = ax1.scatter(
    df_sample['lon_deg'], 
    df_sample['lat_deg'],
    c=df_sample['area_id'],
    cmap='tab20',
    alpha=0.5,
    s=3,
    edgecolors='none'
)

ax1.set_xlabel('経度 (東経)', fontsize=12)
ax1.set_ylabel('緯度 (北緯)', fontsize=12)
ax1.set_title(f'交通事故発生地点のクラスタリング結果\n(area_id: {n_clusters}クラスタ, サンプル数: {sample_size:,}件)', fontsize=14)
ax1.set_xlim(128, 146)  # 日本の経度範囲
ax1.set_ylim(30, 46)    # 日本の緯度範囲
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)

# カラーバー
cbar = plt.colorbar(scatter, ax=ax1, shrink=0.6)
cbar.set_label('area_id', fontsize=11)

output_file1 = OUTPUT_DIR / "geo_clusters_japan_map.png"
fig1.savefig(output_file1, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✅ 保存: {output_file1}")
plt.close(fig1)

# ============================================
# 図2: クラスタ中心点のプロット
# ============================================
fig2, ax2 = plt.subplots(figsize=(14, 12))

# クラスタ中心を計算
cluster_centers = df_valid.groupby('area_id').agg({
    'lon_deg': 'mean',
    'lat_deg': 'mean'
}).reset_index()

cluster_counts = df_valid.groupby('area_id').size().reset_index(name='count')
cluster_centers = cluster_centers.merge(cluster_counts, on='area_id')

# 中心点をプロット（サイズは事故件数に比例）
sizes = (cluster_centers['count'] / cluster_centers['count'].max()) * 500 + 50

scatter2 = ax2.scatter(
    cluster_centers['lon_deg'],
    cluster_centers['lat_deg'],
    c=cluster_centers['area_id'],
    cmap='tab20',
    s=sizes,
    alpha=0.7,
    edgecolors='black',
    linewidths=1
)

# クラスタIDラベル
for _, row in cluster_centers.iterrows():
    ax2.annotate(
        f"{int(row['area_id'])}",
        (row['lon_deg'], row['lat_deg']),
        fontsize=8,
        ha='center',
        va='center',
        color='white',
        fontweight='bold'
    )

ax2.set_xlabel('経度 (東経)', fontsize=12)
ax2.set_ylabel('緯度 (北緯)', fontsize=12)
ax2.set_title(f'クラスタ中心点と事故件数\n（円の大きさ = 事故件数, {n_clusters}クラスタ）', fontsize=14)
ax2.set_xlim(128, 146)
ax2.set_ylim(30, 46)
ax2.set_aspect('equal')
ax2.grid(True, alpha=0.3)

output_file2 = OUTPUT_DIR / "geo_clusters_centers.png"
fig2.savefig(output_file2, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✅ 保存: {output_file2}")
plt.close(fig2)

# ============================================
# 図3: 主要都市圏の拡大図
# ============================================
regions = {
    '関東': {'lon': (138.5, 141.0), 'lat': (34.5, 37.0)},
    '関西': {'lon': (134.5, 136.5), 'lat': (34.0, 35.5)},
    '東海': {'lon': (136.0, 138.5), 'lat': (34.5, 36.0)},
}

fig3, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, (region_name, bounds) in zip(axes, regions.items()):
    # 地域内のデータ抽出
    mask = (
        (df_valid['lon_deg'] >= bounds['lon'][0]) & 
        (df_valid['lon_deg'] <= bounds['lon'][1]) &
        (df_valid['lat_deg'] >= bounds['lat'][0]) & 
        (df_valid['lat_deg'] <= bounds['lat'][1])
    )
    df_region = df_valid[mask]
    
    # サンプリング
    sample_n = min(20000, len(df_region))
    if sample_n > 0:
        df_region_sample = df_region.sample(n=sample_n, random_state=42)
        
        ax.scatter(
            df_region_sample['lon_deg'],
            df_region_sample['lat_deg'],
            c=df_region_sample['area_id'],
            cmap='tab20',
            alpha=0.6,
            s=5,
            edgecolors='none'
        )
    
    ax.set_xlim(bounds['lon'])
    ax.set_ylim(bounds['lat'])
    ax.set_title(f'{region_name}圏\n({len(df_region):,}件)', fontsize=12)
    ax.set_xlabel('経度', fontsize=10)
    ax.set_ylabel('緯度', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

fig3.suptitle('主要都市圏のクラスタリング詳細', fontsize=14, y=1.02)
plt.tight_layout()

output_file3 = OUTPUT_DIR / "geo_clusters_regions.png"
fig3.savefig(output_file3, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✅ 保存: {output_file3}")
plt.close(fig3)

# ============================================
# クラスタ統計情報の出力
# ============================================
print("\n" + "=" * 80)
print("クラスタ統計情報")
print("=" * 80)

stats = df_valid.groupby('area_id').agg({
    'lon_deg': ['mean', 'min', 'max'],
    'lat_deg': ['mean', 'min', 'max'],
}).reset_index()
stats.columns = ['area_id', 'lon_mean', 'lon_min', 'lon_max', 'lat_mean', 'lat_min', 'lat_max']
stats = stats.merge(cluster_counts, on='area_id')
stats = stats.sort_values('count', ascending=False)

print(f"\n上位10クラスタ（事故件数順）:")
print("-" * 70)
for _, row in stats.head(10).iterrows():
    print(f"  area_id={int(row['area_id']):2d}: {row['count']:,}件 "
          f"(経度:{row['lon_mean']:.2f}, 緯度:{row['lat_mean']:.2f})")

# 統計CSVの保存
stats_file = OUTPUT_DIR / "geo_clusters_stats.csv"
stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
print(f"\n✅ 統計情報保存: {stats_file}")

print("\n" + "=" * 80)
print("✅ 可視化完了!")
print("=" * 80)
print(f"\n生成画像:")
print(f"  1. {output_file1}")
print(f"  2. {output_file2}")
print(f"  3. {output_file3}")
