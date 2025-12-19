"""
Phase 2: Hard Negative Mining (Hard FP 分析) - 修正版

新モデルでも高確率で誤検知してしまう「Hard False Positives」を特定し、
共通するパターンを探る。クラスタリングにより誤検知のタイプを分類する。

修正点:
1. 数値コードのカテゴリ変数（信号機, 天候等）はMode（最頻値）で比較
2. クラスタリングはOne-Hot Encodingを使用（Label Encodingの距離問題を回避）

Output:
- hard_fp_profile.md (Hard FPの特徴プロファイリング)
- hard_fp_clusters.csv (クラスタリング結果)
- 可視化プロット
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# フォント設定（安全策）
fonts = [f.name for f in fm.fontManager.ttflist]
if 'MS Gothic' in fonts:
    mpl.rcParams['font.family'] = 'MS Gothic'
elif 'IPAexGothic' in fonts:
    mpl.rcParams['font.family'] = 'IPAexGothic'
else:
    print("Warning: Japanese font not found. Using default.")
mpl.rcParams['axes.unicode_minus'] = False

# パス設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FP_DATA_PATH = os.path.join(BASE_DIR, "results", "experiments", "interaction_features", "fp_new_model.csv")
TP_DATA_PATH = os.path.join(BASE_DIR, "results", "experiments", "interaction_features", "tp_new_model.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "analysis", "hard_negatives")
os.makedirs(RESULTS_DIR, exist_ok=True)

# === 修正1: 数値だがカテゴリとして扱うべきカラムを定義 ===
# これらは数値コードだが、平均値に意味がない（名義尺度）
CATEGORICAL_COLS = [
    '道路形状', '信号機', '昼夜', '天候', '路面状態', '地形',
    '当事者種別（当事者A）', '当事者種別（当事者B）',
    '速度規制（指定のみ）（当事者A）', '速度規制（指定のみ）（当事者B）',
    'area_id', 'road_type', 'stop_sign_interaction',
    # 相互作用特徴量（文字列結合型）
    'party_type_daytime', 'road_shape_terrain', 'night_terrain',
    'signal_road_shape', 'night_road_condition', 'speed_shape_interaction'
]


def load_data():
    """FPデータとTPデータを読み込む"""
    print("Loading FP and TP data...")
    
    if not os.path.exists(FP_DATA_PATH):
        print(f"Error: File not found {FP_DATA_PATH}")
        return pd.DataFrame(), pd.DataFrame()
    
    df_fp = pd.read_csv(FP_DATA_PATH)
    df_tp = pd.read_csv(TP_DATA_PATH)
    
    print(f"  FP samples: {len(df_fp)}")
    print(f"  TP samples: {len(df_tp)}")
    
    return df_fp, df_tp


def select_hard_fp(df_fp: pd.DataFrame, top_n: int = 500):
    """予測確率が高い上位 N 件を Hard FP として選択"""
    print(f"\nSelecting top {top_n} Hard FPs by prediction probability...")
    
    if 'oof_proba' not in df_fp.columns:
        print("  Warning: 'oof_proba' column not found. Using all FPs.")
        return df_fp.head(top_n)
    
    df_hard = df_fp.nlargest(top_n, 'oof_proba').copy()
    print(f"  Hard FP probability range: {df_hard['oof_proba'].min():.4f} - {df_hard['oof_proba'].max():.4f}")
    
    return df_hard


def profile_hard_fp(df_hard: pd.DataFrame, df_tp: pd.DataFrame):
    """Hard FP と TP の特徴を比較（修正版: カテゴリ変数はMode比較）"""
    print("\n--- Profiling Hard FP vs TP ---")
    
    # 分析対象の特徴量
    target_features = list(set(CATEGORICAL_COLS + ['年齢（当事者A）', 'hour', 'month']))
    
    profile_results = []
    
    for feat in target_features:
        if feat not in df_hard.columns:
            continue
        
        # カテゴリ変数かどうかを判定（定義リスト or object型）
        is_categorical = feat in CATEGORICAL_COLS or df_hard[feat].dtype == 'object'
        
        if not is_categorical:
            # 純粋な数値データ (年齢, hour等) -> 平均値比較
            hard_mean = df_hard[feat].mean()
            tp_mean = df_tp[feat].mean() if feat in df_tp.columns else np.nan
            diff = hard_mean - tp_mean if not np.isnan(tp_mean) else np.nan
            
            profile_results.append({
                'Feature': feat,
                'Type': 'Numeric',
                'Hard_FP_Val': f"{hard_mean:.2f}",
                'TP_Val': f"{tp_mean:.2f}",
                'Note': f"Diff: {diff:.2f}" if not np.isnan(diff) else "N/A"
            })
            print(f"  {feat} (Numeric): Hard FP={hard_mean:.2f}, TP={tp_mean:.2f}")
        
        else:
            # カテゴリデータ -> 最頻値(Mode)とその割合比較
            hard_mode = df_hard[feat].mode()[0] if not df_hard[feat].mode().empty else "N/A"
            hard_pct = (df_hard[feat] == hard_mode).mean() * 100
            
            tp_mode = "N/A"
            tp_pct = 0
            if feat in df_tp.columns and not df_tp[feat].mode().empty:
                tp_mode = df_tp[feat].mode()[0]
                tp_pct = (df_tp[feat] == tp_mode).mean() * 100
            
            # モードが一致しているか
            match_mark = "Same" if str(hard_mode) == str(tp_mode) else "**Diff**"
            
            profile_results.append({
                'Feature': feat,
                'Type': 'Categorical',
                'Hard_FP_Val': f"{hard_mode} ({hard_pct:.1f}%)",
                'TP_Val': f"{tp_mode} ({tp_pct:.1f}%)",
                'Note': match_mark
            })
            print(f"  {feat} (Cat): Hard FP mode={hard_mode} ({hard_pct:.1f}%), TP mode={tp_mode} ({tp_pct:.1f}%)")
    
    return pd.DataFrame(profile_results)


def cluster_hard_fp(df_hard: pd.DataFrame, n_clusters: int = 5):
    """Hard FP クラスタリング（One-Hot Encoding修正版）"""
    print(f"\n--- Clustering Hard FPs into {n_clusters} groups ---")
    
    # クラスタリングに使う特徴量（基本的属性に絞る）
    cluster_cols = [
        '道路形状', '昼夜', '天候', '地形',
        '当事者種別（当事者A）', '年齢（当事者A）', 'speed_reg_diff_abs'
    ]
    
    # 利用可能なものだけフィルタ
    use_cols = [c for c in cluster_cols if c in df_hard.columns]
    print(f"  Using {len(use_cols)} features for clustering: {use_cols}")
    
    X = df_hard[use_cols].copy()
    
    # === 修正2: 数値変数とカテゴリ変数を分けて処理 ===
    num_cols = [c for c in use_cols if c not in CATEGORICAL_COLS]
    cat_cols_for_cluster = [c for c in use_cols if c in CATEGORICAL_COLS]
    
    # 数値変数の欠損埋め
    if num_cols:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    
    # カテゴリ変数をOne-Hot Encoding（距離の概念が適切になる）
    X_encoded = pd.get_dummies(X, columns=cat_cols_for_cluster, dummy_na=False, drop_first=False)
    print(f"  Features after One-Hot Encoding: {X_encoded.shape[1]}")
    
    # スケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    # K-Meansクラスタリング
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_hard = df_hard.copy()
    df_hard['cluster'] = clusters
    
    # クラスタごとの統計
    print("\n  Cluster Summary:")
    for c in range(n_clusters):
        cluster_size = (clusters == c).sum()
        cluster_data = df_hard[df_hard['cluster'] == c]
        avg_prob = cluster_data['oof_proba'].mean() if 'oof_proba' in cluster_data.columns else 0
        print(f"    Cluster {c}: {cluster_size} samples, Avg Prob={avg_prob:.4f}")
    
    # PCAで2Dに射影して可視化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Hard FP Clusters (PCA on One-Hot Features)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'hard_fp_clusters_pca.png'), dpi=150)
    plt.close()
    print(f"  Saved PCA plot to {RESULTS_DIR}")
    
    return df_hard, use_cols


def analyze_cluster_characteristics(df_hard: pd.DataFrame, cluster_cols: list):
    """クラスタごとの特徴分析（Mode対応版）"""
    print("\n--- Cluster Characteristics ---")
    
    results = []
    unique_clusters = sorted(df_hard['cluster'].unique())
    
    for c in unique_clusters:
        subset = df_hard[df_hard['cluster'] == c]
        row = {'Cluster': c, 'Count': len(subset)}
        
        # OOF確率の平均
        if 'oof_proba' in subset.columns:
            row['Avg_Prob'] = f"{subset['oof_proba'].mean():.4f}"
        
        # 各特徴量の代表値
        for col in cluster_cols:
            if col not in subset.columns:
                continue
            
            if col in CATEGORICAL_COLS:
                # カテゴリなら最頻値
                mode_val = subset[col].mode()[0] if not subset[col].mode().empty else np.nan
                row[col] = mode_val
            else:
                # 数値なら平均値
                row[col] = f"{subset[col].mean():.2f}"
        
        results.append(row)
    
    return pd.DataFrame(results)


def generate_report(profile_df: pd.DataFrame, cluster_df: pd.DataFrame, df_hard: pd.DataFrame):
    """分析レポートを生成"""
    report_path = os.path.join(RESULTS_DIR, 'hard_fp_profile.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Hard Negative Mining Report\n\n")
        f.write("## 概要\n")
        f.write(f"- **分析対象**: 予測確率上位 {len(df_hard)} 件の False Positives\n")
        if 'oof_proba' in df_hard.columns:
            f.write(f"- **予測確率範囲**: {df_hard['oof_proba'].min():.4f} - {df_hard['oof_proba'].max():.4f}\n\n")
        
        f.write("## 1. Hard FP vs True Positive プロファイル比較\n\n")
        f.write("Hard FPとTPの特徴量比較により、誤検知されやすいパターンを特定。\n")
        f.write("- **Categorical**: 最頻値（Mode）とその割合を比較\n")
        f.write("- **Numeric**: 平均値を比較\n")
        f.write("- **Note**: `**Diff**` は最頻値が異なることを示す（注目ポイント）\n\n")
        
        f.write("| Feature | Type | Hard FP | TP | Note |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for _, row in profile_df.iterrows():
            f.write(f"| {row['Feature']} | {row['Type']} | {row['Hard_FP_Val']} | {row['TP_Val']} | {row['Note']} |\n")
        f.write("\n")
        
        f.write("## 2. クラスタリング結果\n\n")
        f.write("Hard FPをK-Meansで複数のグループに分類し、誤検知のパターンを特定。\n")
        f.write("- One-Hot Encodingを使用（Label Encodingの距離問題を回避）\n\n")
        
        f.write("### クラスタ特性\n")
        f.write(cluster_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("### クラスタ可視化 (PCA)\n")
        f.write("![Clusters](hard_fp_clusters_pca.png)\n\n")
        
        f.write("## 3. 次のアクション\n")
        f.write("1. 各クラスタの代表事例を詳細に確認し、誤検知の原因を特定する。\n")
        f.write("2. 特定のクラスタに対応する新たな特徴量や、ルールベースのフィルタを検討する。\n")
        f.write("3. Hard FP に対する重み付け学習（Hard Negative Mining in training）を検討する。\n")
    
    print(f"\nReport saved to: {report_path}")


def main():
    print("=" * 60)
    print("Phase 2: Hard Negative Mining (Improved)")
    print("=" * 60)
    
    # データ読み込み
    df_fp, df_tp = load_data()
    
    if df_fp.empty:
        print("No data to analyze. Exiting.")
        return
    
    # Hard FP 選択
    df_hard = select_hard_fp(df_fp, top_n=500)
    
    # プロファイリング
    profile_df = profile_hard_fp(df_hard, df_tp)
    
    # クラスタリング
    df_hard, cluster_features = cluster_hard_fp(df_hard, n_clusters=5)
    
    # クラスタ特性分析
    cluster_df = analyze_cluster_characteristics(df_hard, cluster_features)
    
    # 結果保存
    df_hard.to_csv(os.path.join(RESULTS_DIR, 'hard_fp_clusters.csv'), index=False)
    cluster_df.to_csv(os.path.join(RESULTS_DIR, 'cluster_characteristics.csv'), index=False)
    profile_df.to_csv(os.path.join(RESULTS_DIR, 'profile_comparison.csv'), index=False)
    
    # レポート生成
    generate_report(profile_df, cluster_df, df_hard)
    
    print("\nPhase 2 Complete!")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
