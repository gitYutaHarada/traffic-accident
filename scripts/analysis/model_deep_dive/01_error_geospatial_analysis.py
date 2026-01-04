"""
地理的エラー分析 (Geospatial Error Analysis)
==============================================
モデルが「どこで」間違えているかを分析する。

【機能】
- False Positive / False Negative の地理的分布を可視化
- エリア（都道府県/クラスター）ごとの精度差を分析
- 病院距離との関連を検証（データがある場合）

【出力】
- エリア別エラー率のヒートマップ
- FP/FN分析レポート

使用方法:
    python scripts/analysis/model_deep_dive/01_error_geospatial_analysis.py
    python scripts/analysis/model_deep_dive/01_error_geospatial_analysis.py --threshold 0.1
"""

import sys
import io

# Windows環境での文字化け対策: 標準出力をUTF-8に設定
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import json
import argparse
import pickle

# ========================================
# フォント設定（クロスプラットフォーム対応）
# ========================================
def setup_japanese_font():
    """日本語フォントを設定（環境に応じてフォールバック）"""
    try:
        import japanize_matplotlib
        print("   Font: japanize_matplotlib")
        return True
    except ImportError:
        pass
    
    import platform
    if platform.system() == 'Windows':
        try:
            plt.rcParams['font.family'] = 'MS Gothic'
            print("   Font: MS Gothic")
            return True
        except:
            pass
    
    for font in ['IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP', 'DejaVu Sans']:
        try:
            plt.rcParams['font.family'] = font
            print(f"   Font: {font}")
            return True
        except:
            continue
    
    print("   [WARN] Japanese font not found. Using English labels.")
    return False


# ========================================
# 定数
# ========================================
RANDOM_SEED = 42

# パス設定
DATA_DIR = Path("data")
SPATIO_TEMPORAL_DIR = DATA_DIR / "spatio_temporal"
RESULTS_DIR = Path("results")
STACKING_DIR = RESULTS_DIR / "stage3_stacking"

OUTPUT_DIR = RESULTS_DIR / "analysis" / "model_deep_dive" / "geospatial"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# クラスタリングモデル保存パス
KMEANS_MODEL_PATH = OUTPUT_DIR / "area_kmeans_model.pkl"


def load_data():
    """テストデータとStacking予測値をロード"""
    print("[DATA] Loading data...")
    
    # Stage 3 Stacking 予測値
    stacking_preds = pd.read_csv(STACKING_DIR / "test_predictions.csv")
    print(f"   Stacking predictions: {len(stacking_preds):,} records")
    
    # Raw テストデータ（緯度経度含む）
    raw_test = pd.read_parquet(SPATIO_TEMPORAL_DIR / "raw_test.parquet")
    raw_test = raw_test.reset_index(drop=True)
    raw_test['original_index'] = raw_test.index
    print(f"   Raw test: {len(raw_test):,} records")
    
    # マージ前の整合性チェック
    if len(stacking_preds) != len(raw_test):
        print(f"   [WARN] Count mismatch: preds={len(stacking_preds)}, test={len(raw_test)}")
    
    # 必要なカラムを選択
    merge_cols = ['original_index', 'lat', 'lon']
    optional_cols = ['fatal', 'area_id', 'road_type', 'hour', 'road_width', 'weather',
                     'hospital_distance', 'nearest_hospital_dist']
    for col in optional_cols:
        if col in raw_test.columns:
            merge_cols.append(col)
    
    # マージ（1:1の検証付き）
    try:
        df = stacking_preds.merge(
            raw_test[merge_cols], on='original_index', how='left', validate='1:1'
        )
    except pd.errors.MergeError as e:
        print(f"   [WARN] Merge error: {e}. Disabling validate...")
        df = stacking_preds.merge(raw_test[merge_cols], on='original_index', how='left')
    
    # target 設定
    if 'target' not in df.columns and 'fatal' in df.columns:
        df['target'] = df['fatal']
    
    # 緯度経度の欠損チェック
    lat_lon_missing = df[['lat', 'lon']].isna().any(axis=1).sum()
    if lat_lon_missing > 0:
        print(f"   [WARN] Missing lat/lon: {lat_lon_missing:,} records (excluded from viz)")
    
    # 病院距離カラムの特定
    hospital_col = None
    for col in ['hospital_distance', 'nearest_hospital_dist']:
        if col in df.columns:
            hospital_col = col
            print(f"   Hospital distance column: {col}")
            break
    
    print(f"   Merged result: {len(df):,} records")
    
    return df, hospital_col


def find_optimal_threshold(y_true, y_pred_proba, method='f1'):
    """最適な閾値を自動計算"""
    thresholds = np.arange(0.01, 0.5, 0.01)
    best_score = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        
        if method == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif method == 'youden':
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            tn = ((y_pred == 0) & (y_true == 0)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


def classify_predictions(df, threshold=None, auto_optimize=True):
    """閾値に基づき予測をTP/FP/TN/FNに分類（ベクトル化版）"""
    df = df.copy()
    
    y_true = df['target'].values
    y_pred_proba = df['stacking_prob'].values
    
    # 閾値の決定
    if threshold is None:
        if auto_optimize:
            threshold, score = find_optimal_threshold(y_true, y_pred_proba, method='f1')
            print(f"\n[THRESHOLD] Auto-optimized (F1 max): {threshold:.3f} (F1={score:.4f})")
        else:
            mortality_rate = (y_true == 1).mean()
            threshold = max(0.05, min(0.2, mortality_rate * 2))
            print(f"\n[WARN] No threshold specified. Using mortality-based: {threshold:.3f}")
    else:
        print(f"\n[THRESHOLD] Specified: {threshold:.3f}")
    
    df['predicted_label'] = (y_pred_proba >= threshold).astype(int)
    
    # ベクトル化による分類（高速化）
    conditions = [
        (df['target'] == 1) & (df['predicted_label'] == 1),
        (df['target'] == 0) & (df['predicted_label'] == 1),
        (df['target'] == 1) & (df['predicted_label'] == 0),
    ]
    choices = ['TP', 'FP', 'FN']
    df['pred_class'] = np.select(conditions, choices, default='TN')
    
    # 統計出力
    print(f"\n[STATS] Classification at threshold {threshold:.3f}:")
    for cls in ['TP', 'FP', 'FN', 'TN']:
        count = (df['pred_class'] == cls).sum()
        pct = count / len(df) * 100
        print(f"   {cls}: {count:,} ({pct:.2f}%)")
    
    # Recall/Precision計算
    tp = (df['pred_class'] == 'TP').sum()
    fp = (df['pred_class'] == 'FP').sum()
    fn = (df['pred_class'] == 'FN').sum()
    fatal_total = (y_true == 1).sum()
    
    recall = tp / fatal_total if fatal_total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"\n   Recall: {recall:.1%} ({tp}/{fatal_total} fatal detected)")
    print(f"   Precision: {precision:.1%}")
    
    if fn > tp:
        print(f"   [WARN] FN > TP. Consider lowering threshold.")
    
    return df, threshold


def analyze_by_area(df, use_saved_kmeans=True):
    """エリアIDごとの精度分析（特性情報付き）"""
    print("\n[AREA] Analyzing by area...")
    
    if 'area_id' not in df.columns or df['area_id'].isna().all():
        print("   area_id not found. Running KMeans clustering...")
        from sklearn.cluster import MiniBatchKMeans
        
        valid_idx = df[['lat', 'lon']].notna().all(axis=1)
        coords = df.loc[valid_idx, ['lat', 'lon']].values
        
        if use_saved_kmeans and KMEANS_MODEL_PATH.exists():
            print(f"   Loading existing KMeans: {KMEANS_MODEL_PATH}")
            with open(KMEANS_MODEL_PATH, 'rb') as f:
                kmeans = pickle.load(f)
            df.loc[valid_idx, 'area_id'] = kmeans.predict(coords)
        else:
            print("   Creating new KMeans model...")
            kmeans = MiniBatchKMeans(n_clusters=50, random_state=RANDOM_SEED)
            df.loc[valid_idx, 'area_id'] = kmeans.fit_predict(coords)
            with open(KMEANS_MODEL_PATH, 'wb') as f:
                pickle.dump(kmeans, f)
            print(f"   Saved KMeans: {KMEANS_MODEL_PATH}")
    
    # エリアごとの集計
    agg_dict = {
        'target': 'sum',
        'predicted_label': 'sum',
        'stacking_prob': 'mean',
        'original_index': 'count',
        'pred_class': lambda x: (x == 'TP').sum(),
        'lat': 'mean',
        'lon': 'mean',
    }
    
    if 'road_type' in df.columns:
        agg_dict['road_type'] = lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan
    if 'hour' in df.columns:
        agg_dict['hour'] = 'mean'
    
    area_stats = df.groupby('area_id').agg(agg_dict)
    
    rename_dict = {
        'target': 'actual_fatal',
        'predicted_label': 'predicted_fatal',
        'stacking_prob': 'mean_prob',
        'original_index': 'total_accidents',
        'pred_class': 'tp_count',
        'lat': 'center_lat',
        'lon': 'center_lon'
    }
    if 'road_type' in agg_dict:
        rename_dict['road_type'] = 'dominant_road_type'
    if 'hour' in agg_dict:
        rename_dict['hour'] = 'avg_hour'
    
    area_stats = area_stats.rename(columns=rename_dict)
    
    area_stats['fn_count'] = area_stats['actual_fatal'] - area_stats['tp_count']
    area_stats['fp_count'] = area_stats['predicted_fatal'] - area_stats['tp_count']
    
    area_stats['recall'] = np.where(
        area_stats['actual_fatal'] > 0,
        area_stats['tp_count'] / area_stats['actual_fatal'],
        np.nan
    )
    area_stats['precision'] = np.where(
        area_stats['predicted_fatal'] > 0,
        area_stats['tp_count'] / area_stats['predicted_fatal'],
        np.nan
    )
    area_stats['mortality_rate'] = area_stats['actual_fatal'] / area_stats['total_accidents']
    
    area_stats = area_stats.reset_index()
    
    print(f"   Areas: {len(area_stats)}")
    print(f"   Areas with NaN Recall: {area_stats['recall'].isna().sum()}")
    
    return area_stats


def get_dynamic_map_bounds(df, margin=0.5):
    """データから動的に地図の表示範囲を計算"""
    valid_df = df.dropna(subset=['lat', 'lon'])
    
    if len(valid_df) == 0:
        return (122, 146, 24, 46)
    
    x_min = valid_df['lon'].min() - margin
    x_max = valid_df['lon'].max() + margin
    y_min = valid_df['lat'].min() - margin
    y_max = valid_df['lat'].max() + margin
    
    return (x_min, x_max, y_min, y_max)


def plot_error_distribution(df, output_dir, threshold):
    """FP/FNの地理的分布をプロット"""
    print("\n[PLOT] Creating error distribution plot...")
    
    valid_df = df.dropna(subset=['lat', 'lon'])
    
    if len(valid_df) == 0:
        print("   [WARN] No valid coordinates. Skipping.")
        return
    
    fp_df = valid_df[valid_df['pred_class'] == 'FP']
    fn_df = valid_df[valid_df['pred_class'] == 'FN']
    tp_df = valid_df[valid_df['pred_class'] == 'TP']
    
    x_min, x_max, y_min, y_max = get_dynamic_map_bounds(valid_df)
    mean_lat = valid_df['lat'].mean()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    def setup_ax(ax, title):
        ax.scatter(valid_df['lon'], valid_df['lat'], c='lightgray', s=1, alpha=0.3, label='All')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect(1.0 / np.cos(np.radians(mean_lat)))
    
    ax1 = axes[0]
    setup_ax(ax1, f'False Positive (Thresh={threshold:.3f})')
    ax1.scatter(fp_df['lon'], fp_df['lat'], c='red', s=10, alpha=0.6, label=f'FP ({len(fp_df):,})')
    ax1.legend()
    
    ax2 = axes[1]
    setup_ax(ax2, 'False Negative (CRITICAL)')
    ax2.scatter(fn_df['lon'], fn_df['lat'], c='blue', s=20, alpha=0.8, label=f'FN ({len(fn_df):,})')
    ax2.legend()
    
    ax3 = axes[2]
    setup_ax(ax3, 'True Positive (Detected)')
    ax3.scatter(tp_df['lon'], tp_df['lat'], c='green', s=20, alpha=0.8, label=f'TP ({len(tp_df):,})')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "error_geographic_distribution.png", dpi=150)
    plt.close()
    print(f"   Saved: {output_dir / 'error_geographic_distribution.png'}")


def plot_area_heatmap(area_stats, output_dir):
    """エリア別Recall/FNをヒートマップ表示"""
    print("\n[PLOT] Creating area heatmap...")
    
    valid_areas = area_stats[area_stats['actual_fatal'] > 0].copy()
    
    if len(valid_areas) == 0:
        print("   [WARN] No areas with fatal accidents")
        return
    
    x_min = valid_areas['center_lon'].min() - 0.5
    x_max = valid_areas['center_lon'].max() + 0.5
    y_min = valid_areas['center_lat'].min() - 0.5
    y_max = valid_areas['center_lat'].max() + 0.5
    mean_lat = valid_areas['center_lat'].mean()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for ax in axes:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect(1.0 / np.cos(np.radians(mean_lat)))
    
    ax1 = axes[0]
    scatter = ax1.scatter(
        valid_areas['center_lon'], valid_areas['center_lat'],
        c=valid_areas['fn_count'], s=valid_areas['actual_fatal'] * 10,
        cmap='Reds', alpha=0.7
    )
    ax1.set_title('FN Count by Area (Size=Fatal Count)', fontsize=12)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    plt.colorbar(scatter, ax=ax1, label='FN Count')
    
    ax2 = axes[1]
    scatter2 = ax2.scatter(
        valid_areas['center_lon'], valid_areas['center_lat'],
        c=valid_areas['recall'], s=valid_areas['actual_fatal'] * 10,
        cmap='RdYlGn', alpha=0.7, vmin=0, vmax=1
    )
    ax2.set_title('Recall by Area (Size=Fatal Count)', fontsize=12)
    ax2.set_xlabel('Longitude')
    ax2.set_ylabel('Latitude')
    plt.colorbar(scatter2, ax=ax2, label='Recall')
    
    plt.tight_layout()
    plt.savefig(output_dir / "area_performance_heatmap.png", dpi=150)
    plt.close()
    print(f"   Saved: {output_dir / 'area_performance_heatmap.png'}")


def analyze_hospital_distance(df, hospital_col, output_dir):
    """病院距離とエラーの関連分析"""
    if hospital_col is None or hospital_col not in df.columns:
        print("\n[HOSPITAL] No hospital distance data. Skipping.")
        return None
    
    print("\n[HOSPITAL] Analyzing hospital distance...")
    
    valid_df = df.dropna(subset=[hospital_col])
    
    if len(valid_df) == 0:
        print("   [WARN] No valid hospital distance data")
        return None
    
    stats = valid_df.groupby('pred_class')[hospital_col].agg(['mean', 'median', 'std', 'count'])
    stats = stats.round(2)
    
    print("\n   Hospital distance by class:")
    for cls in ['TP', 'FP', 'FN', 'TN']:
        if cls in stats.index:
            row = stats.loc[cls]
            print(f"      {cls}: mean={row['mean']:.1f}m, median={row['median']:.1f}m (n={int(row['count'])})")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    box_data = [valid_df[valid_df['pred_class'] == cls][hospital_col].values 
                for cls in ['TP', 'FP', 'FN', 'TN'] if cls in valid_df['pred_class'].values]
    box_labels = [cls for cls in ['TP', 'FP', 'FN', 'TN'] if cls in valid_df['pred_class'].values]
    ax1.boxplot(box_data, labels=box_labels)
    ax1.set_ylabel('Hospital Distance (m)')
    ax1.set_title('Hospital Distance by Prediction Class', fontsize=12)
    
    ax2 = axes[1]
    fn_dist = valid_df[valid_df['pred_class'] == 'FN'][hospital_col]
    tp_dist = valid_df[valid_df['pred_class'] == 'TP'][hospital_col]
    
    if len(fn_dist) > 0:
        ax2.hist(fn_dist, bins=30, alpha=0.5, label=f'FN (n={len(fn_dist)})', color='blue')
    if len(tp_dist) > 0:
        ax2.hist(tp_dist, bins=30, alpha=0.5, label=f'TP (n={len(tp_dist)})', color='green')
    
    ax2.set_xlabel('Hospital Distance (m)')
    ax2.set_ylabel('Count')
    ax2.set_title('Hospital Distance: FN vs TP', fontsize=12)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "hospital_distance_analysis.png", dpi=150)
    plt.close()
    print(f"   Saved: {output_dir / 'hospital_distance_analysis.png'}")
    
    stats.to_csv(output_dir / "hospital_distance_stats.csv")
    print(f"   Saved: {output_dir / 'hospital_distance_stats.csv'}")
    
    return stats


def generate_report(df, area_stats, threshold, hospital_stats, output_dir):
    """分析レポート生成"""
    print("\n[REPORT] Generating report...")
    
    total = len(df)
    fatal = (df['target'] == 1).sum()
    
    fp_count = (df['pred_class'] == 'FP').sum()
    fn_count = (df['pred_class'] == 'FN').sum()
    tp_count = (df['pred_class'] == 'TP').sum()
    tn_count = (df['pred_class'] == 'TN').sum()
    
    recall = tp_count / fatal if fatal > 0 else 0
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    
    worst_areas = area_stats[area_stats['actual_fatal'] > 0].nsmallest(10, 'recall')
    
    report = f"""# Geospatial Error Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 1. Analysis Parameters

| Item | Value |
|:---|---:|
| **Threshold** | **{threshold:.4f}** |
| Test accidents | {total:,} |
| Fatal accidents | {fatal:,} |
| Mortality rate | {fatal/total*100:.2f}% |

## 2. Overall Statistics

| Class | Count | Ratio |
|:---|---:|---:|
| True Positive | {tp_count:,} | {tp_count/total*100:.2f}% |
| False Positive | {fp_count:,} | {fp_count/total*100:.2f}% |
| False Negative | {fn_count:,} | {fn_count/total*100:.2f}% |
| True Negative | {tn_count:,} | {tn_count/total*100:.2f}% |

**Recall: {recall:.2%}** | **Precision: {precision:.2%}**

## 3. Area Analysis

Total areas: {len(area_stats)} clusters

### Worst Performing Areas (Top 10)

| Area ID | Fatal | FN | Recall | Coordinates |"""
    
    if 'dominant_road_type' in worst_areas.columns:
        report += " Road Type |"
    report += "\n|:---:|---:|---:|---:|:---|"
    if 'dominant_road_type' in worst_areas.columns:
        report += ":---|"
    report += "\n"
    
    for _, row in worst_areas.iterrows():
        line = f"| {int(row['area_id'])} | {int(row['actual_fatal'])} | {int(row['fn_count'])} | {row['recall']:.1%} | ({row['center_lat']:.2f}, {row['center_lon']:.2f}) |"
        if 'dominant_road_type' in worst_areas.columns:
            road_type = row['dominant_road_type'] if pd.notna(row['dominant_road_type']) else '-'
            line += f" {road_type} |"
        report += line + "\n"
    
    if hospital_stats is not None:
        report += """
## 4. Hospital Distance Analysis

| Class | Mean (m) | Median (m) | Count |
|:---:|---:|---:|---:|
"""
        for cls in ['TP', 'FP', 'FN', 'TN']:
            if cls in hospital_stats.index:
                row = hospital_stats.loc[cls]
                report += f"| {cls} | {row['mean']:.0f} | {row['median']:.0f} | {int(row['count'])} |\n"
    
    report += """
## 5. Visualizations

- `error_geographic_distribution.png`: Geographic distribution of FP/FN/TP
- `area_performance_heatmap.png`: Area-level performance
"""
    
    if hospital_stats is not None:
        report += "- `hospital_distance_analysis.png`: Hospital distance analysis\n"
    
    report += """
## 6. Findings

"""
    
    if fn_count > tp_count:
        report += f"> **WARNING**: FN ({fn_count:,}) exceeds TP ({tp_count:,}). Consider lowering threshold.\n\n"
    
    if len(worst_areas) > 0:
        avg_recall = worst_areas['recall'].mean()
        if avg_recall < 0.5:
            report += f"> Worst areas average Recall: **{avg_recall:.1%}**\n\n"
    
    with open(output_dir / "geospatial_analysis_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"   Saved: {output_dir / 'geospatial_analysis_report.md'}")
    
    area_stats.to_csv(output_dir / "area_statistics.csv", index=False)
    print(f"   Saved: {output_dir / 'area_statistics.csv'}")
    
    with open(output_dir / "analysis_config.json", 'w', encoding='utf-8') as f:
        json.dump({
            'threshold': float(threshold),
            'total_samples': int(total),
            'fatal_count': int(fatal),
            'recall': float(recall),
            'precision': float(precision),
            'generated_at': datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)
    print(f"   Saved: {output_dir / 'analysis_config.json'}")


def main():
    parser = argparse.ArgumentParser(description='Geospatial Error Analysis')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Prediction threshold (default: auto F1 optimization)')
    parser.add_argument('--no-auto-optimize', action='store_true',
                        help='Disable threshold auto-optimization')
    parser.add_argument('--reset-kmeans', action='store_true',
                        help='Recreate KMeans model')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Geospatial Error Analysis")
    print("=" * 70)
    
    setup_japanese_font()
    
    df, hospital_col = load_data()
    
    auto_optimize = not args.no_auto_optimize
    df, threshold = classify_predictions(df, threshold=args.threshold, auto_optimize=auto_optimize)
    
    use_saved_kmeans = not args.reset_kmeans
    area_stats = analyze_by_area(df, use_saved_kmeans=use_saved_kmeans)
    
    plot_error_distribution(df, OUTPUT_DIR, threshold)
    plot_area_heatmap(area_stats, OUTPUT_DIR)
    
    hospital_stats = analyze_hospital_distance(df, hospital_col, OUTPUT_DIR)
    
    generate_report(df, area_stats, threshold, hospital_stats, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("[DONE] Geospatial Error Analysis completed!")
    print(f"   Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
