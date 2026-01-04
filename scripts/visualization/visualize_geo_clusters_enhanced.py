"""
地理情報クラスタリング（area_id）の改善版可視化スクリプト
- 地図背景（海岸線・県境）の追加
- ボロノイ図による領域可視化
- 離散的カラーパレット
- 事故多発クラスタのハイライト
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================
# ライブラリのチェックとインポート
# ============================================
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("⚠️ Cartopyがインストールされていません。基本的な描画モードで実行します。")
    print("   インストール方法: pip install cartopy")

try:
    from scipy.spatial import Voronoi
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("⚠️ scipy.spatialがインストールされていません。ボロノイ図はスキップされます。")

# 日本語フォント設定（クロスプラットフォーム対応）
plt.rcParams['font.family'] = ['Meiryo', 'Yu Gothic', 'MS Gothic', 'Hiragino Sans', 'TakaoGothic', 'IPAGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# パス設定
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "honhyo_clean_with_features.csv"
OUTPUT_DIR = BASE_DIR / "results" / "visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 日本の地理的範囲（沖縄を含むよう拡大）
JAPAN_LON_MIN, JAPAN_LON_MAX = 122, 148
JAPAN_LAT_MIN, JAPAN_LAT_MAX = 24, 46


def convert_dms_to_deg(v):
    """度分秒形式（DDDMMSSsss）から10進数の度に変換
    
    安全策: 既に10進数形式（180未満の値）の場合は変換をスキップ
    """
    v = np.where(v == 0, np.nan, v)
    
    # 安全策: 既に10進数形式なら変換しない
    max_val = np.nanmax(v)
    if max_val < 180:
        print("ℹ️ データは既に10進数形式のようです。変換をスキップします。")
        return v.astype(float)
    
    ms = v % 1000
    v = v // 1000
    ss = v % 100
    v = v // 100
    mm = v % 100
    dd = v // 100
    return dd + mm/60 + (ss + ms/1000)/3600


def create_qualitative_colors(n_clusters, top_n=10, cluster_counts=None):
    """
    離散的カラーパレットを作成
    - トップN個の事故多発クラスタ: 暖色系（赤〜オレンジ）
    - その他: グレースケール
    """
    colors = {}
    
    if cluster_counts is not None:
        # 事故件数でソートしてトップNを取得
        sorted_clusters = cluster_counts.sort_values('count', ascending=False)
        top_clusters = sorted_clusters.head(top_n)['area_id'].tolist()
        
        # 暖色系グラデーション（赤→オレンジ→黄）
        warm_colors = plt.cm.Reds(np.linspace(0.9, 0.4, top_n))
        
        for i, cluster_id in enumerate(top_clusters):
            colors[cluster_id] = warm_colors[i]
        
        # その他のクラスタはグレー
        other_clusters = sorted_clusters.iloc[top_n:]['area_id'].tolist()
        for cluster_id in other_clusters:
            colors[cluster_id] = (0.7, 0.7, 0.7, 0.5)  # ライトグレー、半透明
    else:
        # クラスタ件数情報がない場合はtab20を使用
        cmap = plt.cm.get_cmap('tab20', n_clusters)
        for i in range(n_clusters):
            colors[i] = cmap(i)
    
    return colors


def plot_voronoi_on_ax(ax, cluster_centers, xlim, ylim, transform=None):
    """ボロノイ図を描画（境界のみ）
    
    Args:
        ax: matplotlib axes
        cluster_centers: クラスタ中心点のDataFrame
        xlim, ylim: 表示範囲
        transform: Cartopy使用時の座標変換オブジェクト（ccrs.PlateCarree()等）
    """
    if not HAS_SCIPY or len(cluster_centers) < 4:
        return
    
    points = cluster_centers[['lon_deg', 'lat_deg']].values
    
    # 境界外にダミー点を追加（ボロノイ図の端を閉じるため）
    far = 100
    dummy_points = np.array([
        [xlim[0] - far, ylim[0] - far],
        [xlim[0] - far, ylim[1] + far],
        [xlim[1] + far, ylim[0] - far],
        [xlim[1] + far, ylim[1] + far],
    ])
    all_points = np.vstack([points, dummy_points])
    
    try:
        vor = Voronoi(all_points)
        
        # ボロノイ辺を描画
        for ridge_vertices in vor.ridge_vertices:
            if -1 not in ridge_vertices:
                v0, v1 = ridge_vertices
                p0 = vor.vertices[v0]
                p1 = vor.vertices[v1]
                
                # 表示範囲内の線のみ描画
                if (xlim[0] <= p0[0] <= xlim[1] and xlim[0] <= p1[0] <= xlim[1] and
                    ylim[0] <= p0[1] <= ylim[1] and ylim[0] <= p1[1] <= ylim[1]):
                    
                    # Cartopy使用時はtransform引数を適用
                    plot_kwargs = {'linewidth': 0.8, 'alpha': 0.6, 'color': 'black'}
                    if transform is not None:
                        plot_kwargs['transform'] = transform
                    
                    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], **plot_kwargs)
                    
    except Exception as e:
        print(f"  ボロノイ図描画エラー: {e}")


def create_basemap_figure(title, figsize=(14, 12)):
    """地図背景付きの図を作成"""
    if HAS_CARTOPY:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        
        # 地図要素の追加
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='darkblue')
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle='--', edgecolor='gray')
        ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', alpha=0.3)
        ax.add_feature(cfeature.OCEAN, facecolor='#e0f0ff', alpha=0.3)
        
        # 県境（Statesは行政区分）
        try:
            ax.add_feature(cfeature.STATES, linewidth=0.3, edgecolor='gray', alpha=0.5)
        except:
            pass
        
        ax.set_extent([JAPAN_LON_MIN, JAPAN_LON_MAX, JAPAN_LAT_MIN, JAPAN_LAT_MAX], 
                      crs=ccrs.PlateCarree())
        ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlim(JAPAN_LON_MIN, JAPAN_LON_MAX)
        ax.set_ylim(JAPAN_LAT_MIN, JAPAN_LAT_MAX)
        # 日本の中心（北緯35度付近）に合わせてアスペクト比を調整
        # equal だと横長に見えるため、緯度に応じた補正を適用
        ax.set_aspect(1.0 / np.cos(np.radians(38)))
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('経度 (東経)', fontsize=12)
        ax.set_ylabel('緯度 (北緯)', fontsize=12)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    return fig, ax


def main():
    print("=" * 80)
    print("地理情報クラスタリング 改善版可視化")
    print("=" * 80)
    
    # データ読み込み
    print(f"\n[1/6] データ読み込み中: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"✅ 読み込み完了: {len(df):,}件")
    
    # 緯度経度の再計算
    print("\n[2/6] 緯度経度を再計算中...")
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
        return
    
    # 有効なデータのみ抽出
    print("\n[3/6] 有効な座標データを抽出中...")
    df_valid = df[['lon_deg', 'lat_deg', 'area_id']].dropna()
    df_valid = df_valid[df_valid['area_id'] >= 0]
    print(f"✅ 有効データ: {len(df_valid):,}件")
    
    n_clusters = df_valid['area_id'].nunique()
    print(f"   クラスタ数: {n_clusters}")
    
    # クラスタ統計計算
    cluster_centers = df_valid.groupby('area_id').agg({
        'lon_deg': 'mean',
        'lat_deg': 'mean'
    }).reset_index()
    
    cluster_counts = df_valid.groupby('area_id').size().reset_index(name='count')
    cluster_centers = cluster_centers.merge(cluster_counts, on='area_id')
    
    # カラーパレット作成
    colors = create_qualitative_colors(n_clusters, top_n=10, cluster_counts=cluster_counts)
    
    # ============================================
    # 図1: 地図背景付き全体図（トップ10ハイライト）
    # ============================================
    print("\n[4/6] 地図背景付き全体図を作成中...")
    
    fig1, ax1 = create_basemap_figure(
        f'交通事故発生地点のクラスタリング結果\n(事故多発トップ10クラスタをハイライト)'
    )
    
    # サンプリング
    sample_size = min(50000, len(df_valid))
    df_sample = df_valid.sample(n=sample_size, random_state=42)
    
    # クラスタごとに色付けしてプロット
    for area_id in df_sample['area_id'].unique():
        mask = df_sample['area_id'] == area_id
        color = colors.get(area_id, (0.7, 0.7, 0.7, 0.5))
        
        if HAS_CARTOPY:
            ax1.scatter(
                df_sample.loc[mask, 'lon_deg'],
                df_sample.loc[mask, 'lat_deg'],
                c=[color],
                s=3,
                alpha=0.6,
                edgecolors='none',
                transform=ccrs.PlateCarree()
            )
        else:
            ax1.scatter(
                df_sample.loc[mask, 'lon_deg'],
                df_sample.loc[mask, 'lat_deg'],
                c=[color],
                s=3,
                alpha=0.6,
                edgecolors='none'
            )
    
    # 凡例
    top10_clusters = cluster_counts.sort_values('count', ascending=False).head(10)
    legend_patches = []
    for i, (_, row) in enumerate(top10_clusters.iterrows()):
        area_id = row['area_id']
        count = row['count']
        color = colors[area_id]
        patch = mpatches.Patch(color=color, label=f"#{i+1}: Area {int(area_id)} ({count:,}件)")
        legend_patches.append(patch)
    
    legend_patches.append(mpatches.Patch(color=(0.7, 0.7, 0.7, 0.5), label="その他のクラスタ"))
    ax1.legend(handles=legend_patches, loc='upper left', fontsize=9)
    
    output_file1 = OUTPUT_DIR / "geo_clusters_highlight_top10.png"
    fig1.savefig(output_file1, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ 保存: {output_file1}")
    plt.close(fig1)
    
    # ============================================
    # 図2: ボロノイ図
    # ============================================
    if HAS_SCIPY:
        print("\n[5/6] ボロノイ図を作成中...")
        
        fig2, ax2 = create_basemap_figure(
            f'クラスタ領域（ボロノイ図）\n({n_clusters}クラスタの支配領域)'
        )
        
        # クラスタ中心点をプロット
        sizes = (cluster_centers['count'] / cluster_centers['count'].max()) * 300 + 30
        
        for _, row in cluster_centers.iterrows():
            area_id = row['area_id']
            color = colors.get(area_id, (0.7, 0.7, 0.7, 0.5))
            size = (row['count'] / cluster_centers['count'].max()) * 300 + 30
            
            if HAS_CARTOPY:
                ax2.scatter(row['lon_deg'], row['lat_deg'], 
                           c=[color], s=size, alpha=0.8,
                           edgecolors='black', linewidths=0.5,
                           transform=ccrs.PlateCarree(), zorder=5)
            else:
                ax2.scatter(row['lon_deg'], row['lat_deg'], 
                           c=[color], s=size, alpha=0.8,
                           edgecolors='black', linewidths=0.5, zorder=5)
        
        # ボロノイ境界線（Cartopy使用時はtransformを渡す）
        transform_arg = ccrs.PlateCarree() if HAS_CARTOPY else None
        plot_voronoi_on_ax(ax2, cluster_centers, 
                          (JAPAN_LON_MIN, JAPAN_LON_MAX), 
                          (JAPAN_LAT_MIN, JAPAN_LAT_MAX),
                          transform=transform_arg)
        
        # クラスタIDラベル
        for _, row in cluster_centers.iterrows():
            ax2.annotate(
                f"{int(row['area_id'])}",
                (row['lon_deg'], row['lat_deg']),
                fontsize=7,
                ha='center',
                va='center',
                color='white',
                fontweight='bold',
                zorder=10
            )
        
        output_file2 = OUTPUT_DIR / "geo_clusters_voronoi.png"
        fig2.savefig(output_file2, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✅ 保存: {output_file2}")
        plt.close(fig2)
    else:
        print("\n[5/6] ボロノイ図: scipyがないためスキップ")
    
    # ============================================
    # 図3: 主要都市圏の拡大図
    # ============================================
    print("\n[6/6] 主要都市圏の拡大図を作成中...")
    
    regions = {
        '東京23区': {'lon': (139.5, 140.0), 'lat': (35.5, 35.85)},
        '大阪市': {'lon': (135.3, 135.7), 'lat': (34.5, 34.8)},
        '名古屋市': {'lon': (136.7, 137.1), 'lat': (35.0, 35.3)},
        '首都圏全体': {'lon': (138.8, 140.5), 'lat': (35.0, 36.2)},
        '関西圏全体': {'lon': (134.8, 136.2), 'lat': (34.2, 35.2)},
    }
    
    fig3, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (region_name, bounds) in enumerate(regions.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # 地域内のデータ抽出
        mask = (
            (df_valid['lon_deg'] >= bounds['lon'][0]) & 
            (df_valid['lon_deg'] <= bounds['lon'][1]) &
            (df_valid['lat_deg'] >= bounds['lat'][0]) & 
            (df_valid['lat_deg'] <= bounds['lat'][1])
        )
        df_region = df_valid[mask]
        
        # サンプリング
        sample_n = min(15000, len(df_region))
        if sample_n > 0:
            df_region_sample = df_region.sample(n=sample_n, random_state=42)
            
            # クラスタ別に色付け
            for area_id in df_region_sample['area_id'].unique():
                area_mask = df_region_sample['area_id'] == area_id
                color = colors.get(area_id, (0.7, 0.7, 0.7, 0.5))
                ax.scatter(
                    df_region_sample.loc[area_mask, 'lon_deg'],
                    df_region_sample.loc[area_mask, 'lat_deg'],
                    c=[color],
                    s=8,
                    alpha=0.7,
                    edgecolors='none'
                )
        
        ax.set_xlim(bounds['lon'])
        ax.set_ylim(bounds['lat'])
        ax.set_title(f'{region_name}\n({len(df_region):,}件)', fontsize=11, fontweight='bold')
        ax.set_xlabel('経度', fontsize=9)
        ax.set_ylabel('緯度', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # 余ったサブプロットを非表示に
    for idx in range(len(regions), len(axes)):
        axes[idx].axis('off')
    
    fig3.suptitle('主要都市圏のクラスタリング詳細（事故多発エリアをハイライト）', 
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file3 = OUTPUT_DIR / "geo_clusters_city_zoom.png"
    fig3.savefig(output_file3, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ 保存: {output_file3}")
    plt.close(fig3)
    
    # ============================================
    # 図4: 地図背景付き全体図（全クラスタ表示）
    # ============================================
    print("\n[追加] 地図背景付き全体図（全クラスタ）を作成中...")
    
    fig4, ax4 = create_basemap_figure(
        f'交通事故発生地点のクラスタリング結果\n({n_clusters}クラスタ, n={sample_size:,}サンプル)'
    )
    
    # 全クラスタを離散色で表示
    discrete_cmap = plt.cm.get_cmap('tab20', n_clusters)
    
    for i, area_id in enumerate(sorted(df_sample['area_id'].unique())):
        mask = df_sample['area_id'] == area_id
        color = discrete_cmap(i % 20)
        
        if HAS_CARTOPY:
            ax4.scatter(
                df_sample.loc[mask, 'lon_deg'],
                df_sample.loc[mask, 'lat_deg'],
                c=[color],
                s=3,
                alpha=0.6,
                edgecolors='none',
                transform=ccrs.PlateCarree()
            )
        else:
            ax4.scatter(
                df_sample.loc[mask, 'lon_deg'],
                df_sample.loc[mask, 'lat_deg'],
                c=[color],
                s=3,
                alpha=0.6,
                edgecolors='none'
            )
    
    output_file4 = OUTPUT_DIR / "geo_clusters_with_basemap.png"
    fig4.savefig(output_file4, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ 保存: {output_file4}")
    plt.close(fig4)
    
    # 統計CSV保存
    stats = cluster_centers.sort_values('count', ascending=False)
    stats_file = OUTPUT_DIR / "geo_clusters_stats_enhanced.csv"
    stats.to_csv(stats_file, index=False, encoding='utf-8-sig')
    print(f"✅ 統計保存: {stats_file}")
    
    print("\n" + "=" * 80)
    print("✅ 改善版可視化完了!")
    print("=" * 80)
    print(f"\n生成画像:")
    print(f"  1. {output_file1} - トップ10ハイライト")
    if HAS_SCIPY:
        print(f"  2. {output_file2} - ボロノイ図")
    print(f"  3. {output_file3} - 都市圏拡大図")
    print(f"  4. {output_file4} - 地図背景付き全体図")


if __name__ == "__main__":
    main()
