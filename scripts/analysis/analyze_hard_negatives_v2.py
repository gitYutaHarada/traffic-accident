"""
Phase 3: Re-Profiling Hard FPs (残存FP再分析)

モデル改善後に残った「より難しいFP」を再分析し、
以前の主犯格パターンが消えたか、新たな強敵が出現しているかを確認する。

Output:
- reprofiling_report.md (比較レポート)
- cluster_comparison.csv (前回 vs 今回のクラスタ比較)
- hard_fp_clusters_v2.csv (新しいクラスタリング結果)
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

# フォント設定
fonts = [f.name for f in fm.fontManager.ttflist]
if 'MS Gothic' in fonts:
    mpl.rcParams['font.family'] = 'MS Gothic'
elif 'IPAexGothic' in fonts:
    mpl.rcParams['font.family'] = 'IPAexGothic'
mpl.rcParams['axes.unicode_minus'] = False

# パス設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FP_DATA_PATH = os.path.join(BASE_DIR, "results", "experiments", "interaction_features", "fp_new_model.csv")
TP_DATA_PATH = os.path.join(BASE_DIR, "results", "experiments", "interaction_features", "tp_new_model.csv")
PREV_CLUSTER_PATH = os.path.join(BASE_DIR, "results", "analysis", "hard_negatives", "cluster_characteristics.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "analysis", "hard_negatives_v2")
os.makedirs(RESULTS_DIR, exist_ok=True)

# カテゴリ変数定義
CATEGORICAL_COLS = [
    '道路形状', '信号機', '昼夜', '天候', '路面状態', '地形',
    '当事者種別（当事者A）', '速度規制（指定のみ）（当事者A）', '速度規制（指定のみ）（当事者B）',
    'party_type_daytime', 'road_shape_terrain', 'night_terrain'
]

# 前回のCluster 0の特性（主犯格）- 値と割合のペア
PREV_CLUSTER_0_PROFILE = {
    '昼夜': (22, 75.0),           # 75% が夜間
    '地形': (1, 72.0),            # 72% が市街地
    '道路形状': (14, 44.0),       # 44% が単路
    '当事者種別（当事者A）': (3, 57.0)  # 57% が乗用車
}


def load_data():
    """データ読み込み"""
    print("Loading data...")
    
    # データパス検証
    if not os.path.exists(FP_DATA_PATH):
        print(f"  ⚠️ Warning: FP data not found at {FP_DATA_PATH}")
        print(f"  Make sure you ran the experiment first!")
        return pd.DataFrame(), pd.DataFrame(), None
    
    df_fp = pd.read_csv(FP_DATA_PATH)
    df_tp = pd.read_csv(TP_DATA_PATH)
    
    print(f"  Current FP samples: {len(df_fp)}")
    print(f"  Current TP samples: {len(df_tp)}")
    print(f"  Data loaded from: {FP_DATA_PATH}")
    
    # 前回のクラスタ特性を読み込み（存在する場合）
    prev_cluster = None
    if os.path.exists(PREV_CLUSTER_PATH):
        prev_cluster = pd.read_csv(PREV_CLUSTER_PATH)
        print(f"  Previous cluster characteristics loaded")
    
    return df_fp, df_tp, prev_cluster


def select_hard_fp(df_fp: pd.DataFrame, top_n: int = 500):
    """Hard FP 選択"""
    print(f"\nSelecting top {top_n} Hard FPs...")
    
    if 'oof_proba' not in df_fp.columns:
        return df_fp.head(top_n)
    
    df_hard = df_fp.nlargest(top_n, 'oof_proba').copy()
    print(f"  Probability range: {df_hard['oof_proba'].min():.4f} - {df_hard['oof_proba'].max():.4f}")
    
    return df_hard


def profile_current_fp(df_hard: pd.DataFrame):
    """現在のHard FPのプロファイル（新特徴量含む）"""
    print("\n--- Current Hard FP Profile ---")
    
    profile = {}
    
    # 新特徴量を追加
    new_features = ['night_terrain', 'party_type_daytime', 'road_shape_terrain']
    key_features = ['昼夜', '地形', '道路形状', '当事者種別（当事者A）', '天候', '信号機', 'hour'] + new_features
    
    for feat in key_features:
        if feat not in df_hard.columns:
            continue
        
        if feat in CATEGORICAL_COLS or df_hard[feat].dtype == 'object':
            mode = df_hard[feat].mode()[0] if not df_hard[feat].mode().empty else None
            pct = (df_hard[feat] == mode).mean() * 100
            profile[feat] = {'mode': mode, 'pct': pct}
            print(f"  {feat}: mode={mode} ({pct:.1f}%)")
        else:
            mean = df_hard[feat].mean()
            profile[feat] = {'mean': mean}
            print(f"  {feat}: mean={mean:.2f}")
    
    return profile


def check_cluster0_elimination(current_profile: dict):
    """Cluster 0 (前回の主犯格) が消えたかチェック - 割合変化も評価"""
    print("\n--- Cluster 0 Elimination Check ---")
    
    score = 0
    details = []
    
    for feat, (prev_val, prev_pct) in PREV_CLUSTER_0_PROFILE.items():
        if feat not in current_profile:
            continue
        
        curr_mode = current_profile[feat].get('mode')
        curr_pct = current_profile[feat].get('pct', 0)
        
        # 判定ロジック
        if curr_mode != prev_val:
            status = "✅ Disappeared (Type Change)"
            score += 1
        elif curr_pct < (prev_pct * 0.7):  # 3割以上減っていれば改善
            status = f"✅ Reduced ({prev_pct:.0f}% → {curr_pct:.1f}%)"
            score += 0.5
        else:
            status = f"⚠️ Still Dominant ({prev_pct:.0f}% → {curr_pct:.1f}%)"
        
        details.append({
            'feature': feat,
            'prev_val': prev_val,
            'prev_pct': prev_pct,
            'curr_mode': curr_mode,
            'curr_pct': curr_pct,
            'status': status
        })
        print(f"  {feat}: {prev_val}({prev_pct:.0f}%) → {curr_mode}({curr_pct:.1f}%) | {status}")
    
    # 総合判定
    total = len(PREV_CLUSTER_0_PROFILE)
    print(f"\n  Score: {score}/{total}")
    
    if score >= 3:
        print("  ✅ 成功: 主犯格パターンは解消または大幅に縮小しました。")
        return True, details
    elif score >= 2:
        print("  ⚠️ 部分的改善: 一部のパターンが改善しましたが、まだ傾向が残っています。")
        return False, details
    else:
        print("  ⚠️ 警告: まだ以前の傾向が強く残っています。")
        return False, details


def cluster_hard_fp(df_hard: pd.DataFrame, n_clusters: int = 5):
    """クラスタリング"""
    print(f"\n--- Clustering into {n_clusters} groups ---")
    
    cluster_cols = [
        '道路形状', '昼夜', '天候', '地形',
        '当事者種別（当事者A）', '年齢（当事者A）', 'speed_reg_diff_abs'
    ]
    use_cols = [c for c in cluster_cols if c in df_hard.columns]
    
    X = df_hard[use_cols].copy()
    
    num_cols = [c for c in use_cols if c not in CATEGORICAL_COLS]
    cat_cols = [c for c in use_cols if c in CATEGORICAL_COLS]
    
    if num_cols:
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    
    X_encoded = pd.get_dummies(X, columns=cat_cols, dummy_na=False, drop_first=False)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_hard = df_hard.copy()
    df_hard['cluster'] = clusters
    
    # PCA可視化
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Hard FP Clusters v2 (After Model Improvement)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'hard_fp_clusters_v2_pca.png'), dpi=150)
    plt.close()
    
    return df_hard, use_cols


def analyze_new_clusters(df_hard: pd.DataFrame, cluster_cols: list):
    """新しいクラスタの特性分析"""
    print("\n--- New Cluster Characteristics ---")
    
    results = []
    
    for c in sorted(df_hard['cluster'].unique()):
        subset = df_hard[df_hard['cluster'] == c]
        row = {'Cluster': c, 'Count': len(subset)}
        
        if 'oof_proba' in subset.columns:
            row['Avg_Prob'] = subset['oof_proba'].mean()
        
        # 各特徴量の代表値
        for col in cluster_cols:
            if col not in subset.columns:
                continue
            if col in CATEGORICAL_COLS:
                mode = subset[col].mode()[0] if not subset[col].mode().empty else None
                pct = (subset[col] == mode).mean() * 100
                row[f'{col}_mode'] = mode
                row[f'{col}_pct'] = f"{pct:.1f}%"
            else:
                row[f'{col}_mean'] = subset[col].mean()
        
        results.append(row)
        print(f"  Cluster {c}: {len(subset)} samples")
    
    return pd.DataFrame(results)


def detect_new_enemies(df_hard: pd.DataFrame, prev_cluster: pd.DataFrame):
    """新たな強敵パターンの検出"""
    print("\n--- New Enemy Pattern Detection ---")
    
    new_patterns = []
    
    # 頻出パターンを検出
    key_features = ['昼夜', '地形', '道路形状', '天候', '当事者種別（当事者A）']
    
    for feat in key_features:
        if feat not in df_hard.columns:
            continue
        
        value_counts = df_hard[feat].value_counts(normalize=True)
        top_value = value_counts.index[0]
        top_pct = value_counts.iloc[0] * 100
        
        # 50%以上を占めるパターンを「支配的」と判定
        if top_pct >= 50:
            new_patterns.append({
                'Feature': feat,
                'Dominant_Value': top_value,
                'Percentage': f"{top_pct:.1f}%",
                'Warning': "⚠️ NEW ENEMY" if top_pct >= 70 else "👀 Watch"
            })
            print(f"  {feat}: {top_value} ({top_pct:.1f}%) → {new_patterns[-1]['Warning']}")
    
    return pd.DataFrame(new_patterns)


def generate_report(current_profile: dict, cluster0_eliminated: bool, elimination_details: list,
                    cluster_df: pd.DataFrame, new_enemies: pd.DataFrame, df_hard: pd.DataFrame):
    """レポート生成"""
    report_path = os.path.join(RESULTS_DIR, 'reprofiling_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Hard FP Re-Profiling Report (v2)\n\n")
        f.write("## 概要\n")
        f.write(f"モデル改善後の残存FPを再分析した結果を報告する。\n\n")
        f.write(f"- **分析対象**: 予測確率上位 {len(df_hard)} 件のFP\n")
        f.write(f"- **予測確率範囲**: {df_hard['oof_proba'].min():.4f} - {df_hard['oof_proba'].max():.4f}\n\n")
        
        f.write("## 1. Cluster 0 消滅チェック\n\n")
        if cluster0_eliminated:
            f.write("> [!TIP]\n")
            f.write("> **✅ 成功**: 前回の主犯格パターン（夜間×市街地×単路×乗用車）は変化しました。\n\n")
        else:
            f.write("> [!WARNING]\n")
            f.write("> **⚠️ 警告**: 前回の主犯格パターンは依然として残っています。\n\n")
        
        f.write("### 前回 vs 今回の比較（割合変化評価）\n")
        f.write("| Feature | 前回値 | 前回割合 | 今回値 | 今回割合 | Status |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for d in elimination_details:
            f.write(f"| {d['feature']} | {d['prev_val']} | {d['prev_pct']:.0f}% | {d['curr_mode']} | {d['curr_pct']:.1f}% | {d['status']} |\n")
        f.write("\n")
        
        f.write("## 2. 新たな強敵パターン\n\n")
        if len(new_enemies) > 0:
            f.write(new_enemies.to_markdown(index=False))
        else:
            f.write("新たな支配的パターンは検出されませんでした。\n")
        f.write("\n\n")
        
        f.write("## 3. クラスタ分析\n\n")
        f.write(cluster_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 4. クラスタ可視化\n")
        f.write("![Clusters v2](hard_fp_clusters_v2_pca.png)\n\n")
        
        f.write("## 5. 次のアクション\n")
        if cluster0_eliminated:
            f.write("1. 新たに検出されたパターンに対する特徴量エンジニアリングを検討\n")
            f.write("2. モデルアンサンブルによる更なる精度向上\n")
        else:
            f.write("1. ルールベースのフィルタリングを検討（特徴量では区別困難）\n")
            f.write("2. Cluster 0 に対する重み付け学習\n")
    
    print(f"\nReport saved: {report_path}")


def main():
    print("=" * 60)
    print("Phase 3: Re-Profiling Hard FPs (After Model Improvement)")
    print("=" * 60)
    
    # データ読み込み
    df_fp, df_tp, prev_cluster = load_data()
    
    # Hard FP 選択
    df_hard = select_hard_fp(df_fp, top_n=500)
    
    # 現在のプロファイル
    current_profile = profile_current_fp(df_hard)
    
    # Cluster 0 消滅チェック
    cluster0_eliminated, elimination_details = check_cluster0_elimination(current_profile)
    
    # クラスタリング
    df_hard, cluster_cols = cluster_hard_fp(df_hard, n_clusters=5)
    
    # クラスタ特性分析
    cluster_df = analyze_new_clusters(df_hard, cluster_cols)
    
    # 新たな強敵検出
    new_enemies = detect_new_enemies(df_hard, prev_cluster)
    
    # 結果保存
    df_hard.to_csv(os.path.join(RESULTS_DIR, 'hard_fp_clusters_v2.csv'), index=False)
    cluster_df.to_csv(os.path.join(RESULTS_DIR, 'cluster_characteristics_v2.csv'), index=False)
    
    # レポート生成
    generate_report(current_profile, cluster0_eliminated, elimination_details, cluster_df, new_enemies, df_hard)
    
    print("\nPhase 3 Complete!")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
