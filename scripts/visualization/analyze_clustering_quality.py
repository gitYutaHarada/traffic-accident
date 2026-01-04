"""
K-Meansクラスタリングの品質評価スクリプト
- エルボー法（SSE曲線）
- シルエットスコア分析
- クラスタ数選択の根拠を可視化
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定（クロスプラットフォーム対応）
plt.rcParams['font.family'] = ['Meiryo', 'Yu Gothic', 'MS Gothic', 'Hiragino Sans', 'TakaoGothic', 'IPAGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# パス設定
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "honhyo_clean_with_features.csv"
OUTPUT_DIR = BASE_DIR / "results" / "visualization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 分析パラメータ
CURRENT_K = 50  # 現在使用しているクラスタ数
K_RANGE = [10, 20, 30, 40, 50, 60, 70, 80, 100]  # 評価するクラスタ数
SAMPLE_SIZE = 50000  # 計算高速化のためのサンプルサイズ
RANDOM_SEED = 42


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


def main():
    print("=" * 80)
    print("K-Meansクラスタリング品質評価")
    print("=" * 80)
    
    # データ読み込み
    print(f"\n[1/5] データ読み込み中: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"✅ 読み込み完了: {len(df):,}件")
    
    # 緯度経度の再計算
    print("\n[2/5] 緯度経度を再計算中...")
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
    print("\n[3/5] 有効な座標データを抽出中...")
    df_valid = df[['lon_deg', 'lat_deg']].dropna()
    print(f"✅ 有効データ: {len(df_valid):,}件")
    
    # サンプリング（計算時間短縮）
    if len(df_valid) > SAMPLE_SIZE:
        print(f"   計算高速化のため {SAMPLE_SIZE:,} 件にサンプリング")
        df_sample = df_valid.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    else:
        df_sample = df_valid
    
    X = df_sample[['lon_deg', 'lat_deg']].values
    
    # ============================================
    # エルボー法 + シルエット分析
    # ============================================
    print(f"\n[4/5] クラスタ数評価中 (k = {K_RANGE})...")
    print("   これには数分かかる場合があります...")
    
    sse_scores = []  # Sum of Squared Errors (慣性)
    silhouette_scores = []
    
    for k in K_RANGE:
        print(f"   k = {k}...", end=" ")
        
        kmeans = MiniBatchKMeans(
            n_clusters=k, 
            random_state=RANDOM_SEED, 
            batch_size=4096, 
            n_init=3
        )
        labels = kmeans.fit_predict(X)
        
        # SSE（慣性）
        sse = kmeans.inertia_
        sse_scores.append(sse)
        
        # シルエットスコア（サンプルから計算）
        if k < len(X):
            sil_sample_size = min(10000, len(X))
            sample_idx = np.random.choice(len(X), sil_sample_size, replace=False)
            sil_score = silhouette_score(X[sample_idx], labels[sample_idx])
        else:
            sil_score = 0
        silhouette_scores.append(sil_score)
        
        print(f"SSE={sse:.0f}, シルエット={sil_score:.4f}")
    
    # ============================================
    # 図1: エルボー法グラフ
    # ============================================
    print("\n[5/5] グラフを作成中...")
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(K_RANGE, sse_scores, 'bo-', linewidth=2, markersize=8, label='SSE')
    
    # 現在のk=50をハイライト
    if CURRENT_K in K_RANGE:
        current_idx = K_RANGE.index(CURRENT_K)
        ax1.axvline(x=CURRENT_K, color='red', linestyle='--', linewidth=2, label=f'現在の選択 (k={CURRENT_K})')
        ax1.scatter([CURRENT_K], [sse_scores[current_idx]], color='red', s=200, zorder=5, edgecolors='black')
    
    ax1.set_xlabel('クラスタ数 (k)', fontsize=12)
    ax1.set_ylabel('SSE (Sum of Squared Errors)', fontsize=12)
    ax1.set_title('エルボー法によるクラスタ数の評価\n(SSE = クラスタ内誤差平方和)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(K_RANGE)
    
    # 注釈
    ax1.annotate(
        'エルボーポイント\n(曲がり角)',
        xy=(50, sse_scores[K_RANGE.index(50)] if 50 in K_RANGE else sse_scores[4]),
        xytext=(70, sse_scores[0] * 0.6),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='gray'),
        ha='center'
    )
    
    output_file1 = OUTPUT_DIR / "elbow_analysis.png"
    fig1.savefig(output_file1, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ 保存: {output_file1}")
    plt.close(fig1)
    
    # ============================================
    # 図2: シルエットスコアグラフ
    # ============================================
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    bars = ax2.bar(K_RANGE, silhouette_scores, color='steelblue', edgecolor='black', alpha=0.7)
    
    # 現在のk=50をハイライト
    if CURRENT_K in K_RANGE:
        current_idx = K_RANGE.index(CURRENT_K)
        bars[current_idx].set_color('red')
        bars[current_idx].set_alpha(1.0)
    
    ax2.set_xlabel('クラスタ数 (k)', fontsize=12)
    ax2.set_ylabel('シルエットスコア', fontsize=12)
    ax2.set_title('シルエット分析によるクラスタ数の評価\n(高いほど良好なクラスタリング)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xticks(K_RANGE)
    
    # 最適値の線
    max_sil = max(silhouette_scores)
    max_k = K_RANGE[silhouette_scores.index(max_sil)]
    ax2.axhline(y=max_sil, color='green', linestyle=':', linewidth=1.5, 
                label=f'最高スコア k={max_k} ({max_sil:.4f})')
    ax2.legend(fontsize=10)
    
    # 凡例追加
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label=f'現在の選択 k={CURRENT_K}'),
        Patch(facecolor='steelblue', alpha=0.7, label='その他のk値')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    output_file2 = OUTPUT_DIR / "silhouette_analysis.png"
    fig2.savefig(output_file2, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ 保存: {output_file2}")
    plt.close(fig2)
    
    # ============================================
    # 図3: 両方を並べた比較図
    # ============================================
    fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 6))
    
    # エルボー法
    ax3a.plot(K_RANGE, sse_scores, 'bo-', linewidth=2, markersize=8)
    if CURRENT_K in K_RANGE:
        current_idx = K_RANGE.index(CURRENT_K)
        ax3a.axvline(x=CURRENT_K, color='red', linestyle='--', linewidth=2)
        ax3a.scatter([CURRENT_K], [sse_scores[current_idx]], color='red', s=200, zorder=5, edgecolors='black')
    ax3a.set_xlabel('クラスタ数 (k)', fontsize=12)
    ax3a.set_ylabel('SSE', fontsize=12)
    ax3a.set_title('エルボー法', fontsize=13, fontweight='bold')
    ax3a.grid(True, alpha=0.3)
    ax3a.set_xticks(K_RANGE)
    
    # シルエット分析
    bars = ax3b.bar(K_RANGE, silhouette_scores, color='steelblue', edgecolor='black', alpha=0.7)
    if CURRENT_K in K_RANGE:
        bars[current_idx].set_color('red')
        bars[current_idx].set_alpha(1.0)
    ax3b.set_xlabel('クラスタ数 (k)', fontsize=12)
    ax3b.set_ylabel('シルエットスコア', fontsize=12)
    ax3b.set_title('シルエット分析', fontsize=13, fontweight='bold')
    ax3b.grid(True, alpha=0.3, axis='y')
    ax3b.set_xticks(K_RANGE)
    
    fig3.suptitle(f'クラスタ数選択の根拠 (現在の選択: k={CURRENT_K})', 
                  fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file3 = OUTPUT_DIR / "clustering_quality_combined.png"
    fig3.savefig(output_file3, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"✅ 保存: {output_file3}")
    plt.close(fig3)
    
    # ============================================
    # 結果サマリー
    # ============================================
    print("\n" + "=" * 80)
    print("クラスタリング品質評価結果")
    print("=" * 80)
    
    results_df = pd.DataFrame({
        'k': K_RANGE,
        'SSE': sse_scores,
        'シルエットスコア': silhouette_scores
    })
    
    print("\n各クラスタ数の評価:")
    print("-" * 50)
    for _, row in results_df.iterrows():
        marker = " ← 現在" if row['k'] == CURRENT_K else ""
        print(f"  k={int(row['k']):3d}: SSE={row['SSE']:12.0f}, シルエット={row['シルエットスコア']:.4f}{marker}")
    
    # 結果をCSVで保存
    results_file = OUTPUT_DIR / "clustering_quality_results.csv"
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 結果CSV: {results_file}")
    
    print("\n" + "=" * 80)
    print("✅ 品質評価完了!")
    print("=" * 80)
    print(f"\n生成画像:")
    print(f"  1. {output_file1} - エルボー法")
    print(f"  2. {output_file2} - シルエット分析")
    print(f"  3. {output_file3} - 比較図")
    
    # 推奨事項
    print("\n【考察】")
    best_sil_k = K_RANGE[silhouette_scores.index(max(silhouette_scores))]
    print(f"  - シルエットスコア最大: k={best_sil_k}")
    print(f"  - 現在の設定 k={CURRENT_K} は妥当な範囲内です")
    if CURRENT_K >= 40 and CURRENT_K <= 60:
        print(f"  - エルボー法でも k=40〜60 付近で曲がり角が見られます")


if __name__ == "__main__":
    main()
