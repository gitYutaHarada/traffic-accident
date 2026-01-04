"""
4モデルアンサンブル 誤差分析スクリプト (v3: 強化版)
==============================================================
目的: アンサンブルモデルが間違えた事例 (FP/FN) の特性を特定し、
      今後のモデル改善方針を決定するための根拠を提供する。

v3修正点:
- 元データとOOF予測値を正確に紐付け（多重検証付き）
- モデル間の相関係数分析を追加
- Seabornを使った可視化（KDEプロット）
- 特徴量別エラー率の棒グラフ可視化
- 混同行列のヒートマップ
- コードのクリーンアップ

使用方法:
    python scripts/experiment/analyze_ensemble_errors.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, 
    f1_score, confusion_matrix
)

# 特定の警告のみ抑制（FutureWarning等）
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# --- 設定 ---
DATA_PATH = Path("data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv")
STAGE1_OOF_PATH = Path("data/processed/stage1_oof_predictions.csv")
ENSEMBLE_OOF_PATH = Path("results/tabnet_optimized/oof_predictions.csv")
OUTPUT_DIR = Path("results/error_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
STAGE1_RECALL_TARGET = 0.98

# 日本語フォント設定 (Matplotlib/Seaborn)
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 分析対象の特徴量カラム
ANALYSIS_COLS = [
    '都道府県コード', '市区町村コード', '昼夜', '天候', '地形', '路面状態',
    '道路形状', '信号機', '事故類型', '曜日(発生年月日)', '時', '月',
    '歩車道区分', '中央分離帯施設等', 'road_type'
]

# モデル名のマッピング
MODEL_COLS = ['pred_lgbm', 'pred_catboost', 'pred_mlp', 'pred_tabnet_optimized']
MODEL_NAMES = ['LightGBM', 'CatBoost', 'MLP', 'TabNet']


def load_and_align_data():
    """
    元データとOOF予測値を正確に紐付ける
    
    処理フロー:
    1. 元データを読み込み
    2. 同じシードで80/20分割してTrainインデックスを取得
    3. Stage 1フィルタリングを再現してフィルタ通過インデックスを取得
    4. 多重検証後、元データの特徴量とOOF予測値を紐付け
    """
    print("📂 データ読み込み・紐付け中...")
    
    # 元データ読み込み
    df_full = pd.read_csv(DATA_PATH)
    df_full['fatal'] = df_full['fatal'].astype(int)
    print(f"   元データ: {len(df_full):,} 行")
    
    # Stage 1 OOF読み込み（Train部分の予測値）
    stage1_oof = pd.read_csv(STAGE1_OOF_PATH)
    print(f"   Stage1 OOF: {len(stage1_oof):,} 行")
    
    # Ensemble OOF読み込み（フィルタ済みデータの予測値）
    ensemble_oof = pd.read_csv(ENSEMBLE_OOF_PATH)
    print(f"   Ensemble OOF: {len(ensemble_oof):,} 行")
    
    # === Step 1: 80/20分割のインデックスを再現 ===
    all_indices = np.arange(len(df_full))
    train_indices, _ = train_test_split(
        all_indices, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=df_full['fatal']
    )
    print(f"   Train分割: {len(train_indices):,} 行")
    
    # === Step 2: Stage 1フィルタリング閾値を再計算 ===
    y_train = df_full.iloc[train_indices]['fatal'].values
    stage1_prob = 0.85 * stage1_oof['prob_catboost'].values + 0.15 * stage1_oof['prob_lgbm'].values
    
    precision, recall, thresholds = precision_recall_curve(y_train, stage1_prob)
    valid_idx = np.where(recall[:-1] >= STAGE1_RECALL_TARGET)[0]
    stage1_threshold = thresholds[valid_idx[-1]] if len(valid_idx) > 0 else 0.0
    print(f"   Stage1閾値: {stage1_threshold:.6f}")
    
    # === Step 3: フィルタ通過インデックスを取得 ===
    filter_mask = stage1_prob >= stage1_threshold
    filtered_train_indices = train_indices[filter_mask]
    print(f"   フィルタ通過: {len(filtered_train_indices):,} 行")
    
    # === Step 4: 多重検証 ===
    if len(filtered_train_indices) != len(ensemble_oof):
        raise ValueError(f"❌ 行数不一致: filtered={len(filtered_train_indices)}, oof={len(ensemble_oof)}")
    
    original_target = df_full.iloc[filtered_train_indices]['fatal'].values
    oof_target = ensemble_oof['target'].values
    
    if not np.array_equal(original_target, oof_target):
        raise ValueError("❌ ターゲット値が一致しません。紐付けに問題があります。")
    
    original_fatal_positions = np.where(original_target == 1)[0]
    oof_fatal_positions = np.where(oof_target == 1)[0]
    
    if not np.array_equal(original_fatal_positions, oof_fatal_positions):
        raise ValueError("❌ fatal=1の位置が一致しません。紐付けに問題があります。")
    
    print("   ✅ 多重検証パス:")
    print(f"      - ターゲット完全一致: True")
    print(f"      - fatal=1位置一致: True ({len(original_fatal_positions):,} 件)")
    print(f"      - 信頼度: HIGH")
    
    # === Step 5: 紐付けデータフレーム作成 ===
    df_aligned = df_full.iloc[filtered_train_indices].reset_index(drop=True).copy()
    
    for col in ensemble_oof.columns:
        df_aligned[f'pred_{col}'] = ensemble_oof[col].values
    
    return df_aligned


def find_optimal_threshold(y_true, y_prob):
    """F1スコア最大化閾値を見つける"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0
    )
    
    best_idx = np.argmax(f1_scores[:-1])
    return thresholds[best_idx], f1_scores[best_idx]


def extract_hard_examples(df, threshold):
    """FP/FN/TP/TN を抽出"""
    y_true = df['fatal'].values
    y_prob = df['pred_ensemble'].values
    y_pred = (y_prob >= threshold).astype(int)
    
    tp_mask = (y_true == 1) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    
    stats = {
        'TP': tp_mask.sum(),
        'TN': tn_mask.sum(),
        'FP': fp_mask.sum(),
        'FN': fn_mask.sum(),
    }
    
    print(f"\n📊 分類結果 (閾値: {threshold:.4f})")
    print(f"   TP (正しく検出): {stats['TP']:,}")
    print(f"   TN (正しく棄却): {stats['TN']:,}")
    print(f"   FP (誤検知):    {stats['FP']:,}")
    print(f"   FN (見逃し):    {stats['FN']:,}")
    
    return tp_mask, tn_mask, fp_mask, fn_mask, stats


def plot_confusion_matrix(stats, output_dir):
    """混同行列のヒートマップを作成"""
    print("\n📊 混同行列ヒートマップ作成...")
    
    cm = np.array([
        [stats['TN'], stats['FP']],
        [stats['FN'], stats['TP']]
    ])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt=',d', cmap='Blues',
        xticklabels=['予測: 非死亡', '予測: 死亡'],
        yticklabels=['実際: 非死亡', '実際: 死亡'],
        ax=ax
    )
    ax.set_title('混同行列', fontsize=14)
    ax.set_xlabel('予測', fontsize=12)
    ax.set_ylabel('実際', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"   保存: confusion_matrix.png")


def analyze_prediction_distribution(df, tp_mask, tn_mask, fp_mask, fn_mask, output_dir):
    """FP/FN vs TP/TN の予測確率分布を比較（Seaborn KDEプロット）"""
    print("\n📈 予測確率分布の分析 (Seaborn)...")
    
    y_prob = df['pred_ensemble'].values
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # FP vs TN の比較
    ax1 = axes[0]
    df_plot1 = pd.DataFrame({
        'pred': np.concatenate([y_prob[fp_mask], y_prob[tn_mask]]),
        'type': ['FP (誤検知)'] * fp_mask.sum() + ['TN (正解)'] * tn_mask.sum()
    })
    sns.histplot(
        data=df_plot1, x='pred', hue='type', kde=True,
        palette={'FP (誤検知)': 'red', 'TN (正解)': 'blue'},
        alpha=0.5, ax=ax1, stat='density'
    )
    ax1.set_xlabel('予測確率')
    ax1.set_ylabel('密度')
    ax1.set_title('FP vs TN の予測確率分布')
    ax1.legend(title='')
    
    # FN vs TP の比較
    ax2 = axes[1]
    df_plot2 = pd.DataFrame({
        'pred': np.concatenate([y_prob[fn_mask], y_prob[tp_mask]]),
        'type': ['FN (見逃し)'] * fn_mask.sum() + ['TP (正解)'] * tp_mask.sum()
    })
    sns.histplot(
        data=df_plot2, x='pred', hue='type', kde=True,
        palette={'FN (見逃し)': 'orange', 'TP (正解)': 'green'},
        alpha=0.5, ax=ax2, stat='density'
    )
    ax2.set_xlabel('予測確率')
    ax2.set_ylabel('密度')
    ax2.set_title('FN vs TP の予測確率分布')
    ax2.legend(title='')
    
    plt.tight_layout()
    plt.savefig(output_dir / "prediction_distribution.png", dpi=150)
    plt.close()
    print(f"   保存: prediction_distribution.png")


def analyze_model_correlation(df, tp_mask, tn_mask, fp_mask, fn_mask, output_dir):
    """
    エラー事例におけるモデル間の相関係数を分析
    
    相関が高い → 全モデルが同じ間違いをしている → データ自体の難易度が高い
    相関が低い → 特定のモデルだけ間違えている → そのモデルに改善余地あり
    """
    print("\n🔗 モデル間相関係数分析...")
    
    model_cols = [c for c in MODEL_COLS if c in df.columns]
    
    # 各ケースでの相関行列を計算
    cases = {
        'FP (誤検知)': fp_mask,
        'FN (見逃し)': fn_mask,
        'TP (正検出)': tp_mask,
        'TN (正棄却)': tn_mask,
        '全データ': np.ones(len(df), dtype=bool)
    }
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    correlation_results = {}
    
    for idx, (case_name, mask) in enumerate(cases.items()):
        if mask.sum() < 10:  # サンプル数が少ない場合はスキップ
            continue
        
        df_subset = df.loc[mask, model_cols]
        corr_matrix = df_subset.corr()
        
        # 相関行列の要約統計
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        avg_corr = upper_tri.stack().mean()
        
        correlation_results[case_name] = {
            'avg_correlation': avg_corr,
            'sample_count': mask.sum()
        }
        
        # ヒートマップ
        if idx < len(axes):
            ax = axes[idx]
            sns.heatmap(
                corr_matrix, annot=True, fmt='.2f', cmap='RdYlBu_r',
                vmin=0, vmax=1, center=0.5,
                xticklabels=MODEL_NAMES, yticklabels=MODEL_NAMES,
                ax=ax
            )
            ax.set_title(f'{case_name}\n(N={mask.sum():,}, 平均相関={avg_corr:.2f})')
    
    # 未使用のaxesを非表示
    for idx in range(len(cases), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_correlation.png", dpi=150)
    plt.close()
    print(f"   保存: model_correlation.png")
    
    # 相関分析の解釈
    print("\n   📊 相関分析結果:")
    for case_name, result in correlation_results.items():
        interpretation = ""
        if case_name in ['FP (誤検知)', 'FN (見逃し)']:
            if result['avg_correlation'] > 0.7:
                interpretation = "→ 全モデルが同じ間違い（データ難易度が高い）"
            elif result['avg_correlation'] < 0.4:
                interpretation = "→ 特定モデルの問題（改善余地あり）"
            else:
                interpretation = "→ 中程度の相関"
        print(f"      {case_name}: 平均相関={result['avg_correlation']:.3f} {interpretation}")
    
    return correlation_results


def analyze_model_disagreement(df, fp_mask, fn_mask, output_dir):
    """各モデルの予測値の不一致を分析（平均値比較）"""
    print("\n🔍 モデル間の予測不一致分析...")
    
    model_cols = [c for c in MODEL_COLS if c in df.columns]
    
    results = {'FP': {}, 'FN': {}}
    
    for model, name in zip(model_cols, MODEL_NAMES):
        results['FP'][name] = df.loc[fp_mask, model].mean()
        results['FN'][name] = df.loc[fn_mask, model].mean()
    
    print("\n   FP/FN 時の各モデル平均予測確率:")
    print("   " + "-" * 50)
    print(f"   {'Model':<20} {'FP Mean':>12} {'FN Mean':>12}")
    print("   " + "-" * 50)
    for name in MODEL_NAMES:
        if name in results['FP']:
            print(f"   {name:<20} {results['FP'][name]:>12.4f} {results['FN'][name]:>12.4f}")
    print("   " + "-" * 50)
    
    # 可視化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    fp_means = [results['FP'].get(n, 0) for n in MODEL_NAMES]
    colors = sns.color_palette("Reds", len(MODEL_NAMES))
    ax1.bar(MODEL_NAMES, fp_means, color=colors)
    ax1.axhline(df.loc[fp_mask, 'pred_ensemble'].mean(), color='darkred', linestyle='--', 
                label=f'Ensemble平均: {df.loc[fp_mask, "pred_ensemble"].mean():.4f}')
    ax1.set_ylabel('平均予測確率')
    ax1.set_title('False Positive: 各モデルの平均予測値')
    ax1.legend()
    
    ax2 = axes[1]
    fn_means = [results['FN'].get(n, 0) for n in MODEL_NAMES]
    colors = sns.color_palette("Oranges", len(MODEL_NAMES))
    ax2.bar(MODEL_NAMES, fn_means, color=colors)
    ax2.axhline(df.loc[fn_mask, 'pred_ensemble'].mean(), color='darkorange', linestyle='--',
                label=f'Ensemble平均: {df.loc[fn_mask, "pred_ensemble"].mean():.4f}')
    ax2.set_ylabel('平均予測確率')
    ax2.set_title('False Negative: 各モデルの平均予測値')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_disagreement.png", dpi=150)
    plt.close()
    print(f"   保存: model_disagreement.png")
    
    return results


def analyze_feature_distribution(df, tp_mask, tn_mask, fp_mask, fn_mask, output_dir):
    """FP/FN vs TP/TN の特徴量分布を比較分析"""
    print("\n🔍 特徴量別エラー率分析...")
    
    results = []
    
    for col in ANALYSIS_COLS:
        if col not in df.columns:
            continue
        
        for cat in df[col].dropna().unique():
            cat_mask = df[col] == cat
            
            n_total = cat_mask.sum()
            n_positive = (cat_mask & (df['fatal'] == 1)).sum()
            n_fp = (cat_mask & fp_mask).sum()
            n_fn = (cat_mask & fn_mask).sum()
            n_tp = (cat_mask & tp_mask).sum()
            n_tn = (cat_mask & tn_mask).sum()
            
            fp_rate = n_fp / (n_fp + n_tn) if (n_fp + n_tn) > 0 else 0
            fn_rate = n_fn / (n_fn + n_tp) if (n_fn + n_tp) > 0 else 0
            
            results.append({
                'feature': col,
                'category': cat,
                'total': n_total,
                'positive': n_positive,
                'positive_rate': n_positive / n_total if n_total > 0 else 0,
                'FP': n_fp, 'FN': n_fn, 'TP': n_tp, 'TN': n_tn,
                'FP_rate': fp_rate, 'FN_rate': fn_rate,
            })
    
    results_df = pd.DataFrame(results)
    
    # 全体平均エラー率
    overall_fp_rate = fp_mask.sum() / (fp_mask.sum() + tn_mask.sum())
    overall_fn_rate = fn_mask.sum() / (fn_mask.sum() + tp_mask.sum())
    
    results_df['FP_rate_ratio'] = results_df['FP_rate'] / overall_fp_rate
    results_df['FN_rate_ratio'] = results_df['FN_rate'] / overall_fn_rate
    
    # サンプル数が少ないものを除外
    results_df = results_df[results_df['total'] >= 100]
    
    high_fp_risk = results_df[results_df['FP_rate_ratio'] > 1.5].sort_values('FP_rate_ratio', ascending=False)
    high_fn_risk = results_df[results_df['FN_rate_ratio'] > 1.5].sort_values('FN_rate_ratio', ascending=False)
    
    print(f"\n   🔴 高FPリスク Top 10:")
    for _, row in high_fp_risk.head(10).iterrows():
        print(f"      {row['feature']}={row['category']}: FP率 {row['FP_rate']:.3f} ({row['FP_rate_ratio']:.1f}x)")
    
    print(f"\n   🟠 高FNリスク Top 10:")
    for _, row in high_fn_risk.head(10).iterrows():
        print(f"      {row['feature']}={row['category']}: FN率 {row['FN_rate']:.3f} ({row['FN_rate_ratio']:.1f}x)")
    
    results_df.to_csv(output_dir / "feature_error_analysis.csv", index=False)
    print(f"\n   保存: feature_error_analysis.csv")
    
    return results_df, high_fp_risk, high_fn_risk


def plot_feature_error_rates(results_df, output_dir):
    """特徴量別エラー率の棒グラフを作成"""
    print("\n� 特徴量別エラー率グラフ作成...")
    
    # 各特徴量の中で最もエラー率比が高いカテゴリを抽出
    top_fp_by_feature = results_df.loc[results_df.groupby('feature')['FP_rate_ratio'].idxmax()]
    top_fn_by_feature = results_df.loc[results_df.groupby('feature')['FN_rate_ratio'].idxmax()]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # FP率が高い特徴量×カテゴリ
    ax1 = axes[0]
    top_fp = top_fp_by_feature.nlargest(10, 'FP_rate_ratio')
    labels = [f"{row['feature']}\n({row['category']})" for _, row in top_fp.iterrows()]
    values = top_fp['FP_rate_ratio'].values
    colors = sns.color_palette("Reds_r", len(labels))
    ax1.barh(labels, values, color=colors)
    ax1.axvline(1.0, color='gray', linestyle='--', label='全体平均')
    ax1.set_xlabel('FP率比 (全体平均=1.0)')
    ax1.set_title('高FPリスク特徴 Top 10')
    ax1.legend()
    
    # FN率が高い特徴量×カテゴリ
    ax2 = axes[1]
    top_fn = top_fn_by_feature.nlargest(10, 'FN_rate_ratio')
    labels = [f"{row['feature']}\n({row['category']})" for _, row in top_fn.iterrows()]
    values = top_fn['FN_rate_ratio'].values
    colors = sns.color_palette("Oranges_r", len(labels))
    ax2.barh(labels, values, color=colors)
    ax2.axvline(1.0, color='gray', linestyle='--', label='全体平均')
    ax2.set_xlabel('FN率比 (全体平均=1.0)')
    ax2.set_title('高FNリスク特徴 Top 10')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "feature_error_rates.png", dpi=150)
    plt.close()
    print(f"   保存: feature_error_rates.png")


def save_hard_examples_with_features(df, fp_mask, fn_mask, output_dir):
    """高難易度事例を特徴量付きでCSVに保存"""
    print("\n💾 高難易度事例の保存...")
    
    save_cols = ANALYSIS_COLS + ['fatal'] + MODEL_COLS + ['pred_ensemble']
    save_cols = [c for c in save_cols if c in df.columns]
    
    fp_df = df.loc[fp_mask, save_cols].copy()
    fp_df['error_type'] = 'FP'
    fp_df = fp_df.sort_values('pred_ensemble', ascending=False)
    
    fn_df = df.loc[fn_mask, save_cols].copy()
    fn_df['error_type'] = 'FN'
    fn_df = fn_df.sort_values('pred_ensemble', ascending=True)
    
    fp_df.to_csv(output_dir / "false_positives_all.csv", index=False)
    fn_df.to_csv(output_dir / "false_negatives_all.csv", index=False)
    print(f"   保存: false_positives_all.csv ({len(fp_df):,} 件)")
    print(f"   保存: false_negatives_all.csv ({len(fn_df):,} 件)")
    
    high_confidence_fp = fp_df[fp_df['pred_ensemble'] > 0.5]
    if len(high_confidence_fp) > 0:
        high_confidence_fp.to_csv(output_dir / "high_confidence_fp.csv", index=False)
        print(f"   保存: high_confidence_fp.csv ({len(high_confidence_fp):,} 件)")
    
    low_confidence_fn = fn_df[fn_df['pred_ensemble'] < 0.1]
    if len(low_confidence_fn) > 0:
        low_confidence_fn.to_csv(output_dir / "low_confidence_fn.csv", index=False)
        print(f"   保存: low_confidence_fn.csv ({len(low_confidence_fn):,} 件)")


def generate_report(df, stats, threshold, best_f1, high_fp_risk, high_fn_risk, 
                    model_disagreement, correlation_results, output_dir):
    """分析レポートをMarkdown形式で生成"""
    print("\n📄 レポート生成中...")
    
    total = stats['TP'] + stats['TN'] + stats['FP'] + stats['FN']
    precision = stats['TP'] / (stats['TP'] + stats['FP']) if (stats['TP'] + stats['FP']) > 0 else 0
    recall = stats['TP'] / (stats['TP'] + stats['FN']) if (stats['TP'] + stats['FN']) > 0 else 0
    
    fp_risk_table = ""
    for _, row in high_fp_risk.head(15).iterrows():
        fp_risk_table += f"| {row['feature']} | {row['category']} | {row['total']:,} | {row['FP_rate']:.3f} | {row['FP_rate_ratio']:.1f}x |\n"
    
    fn_risk_table = ""
    for _, row in high_fn_risk.head(15).iterrows():
        fn_risk_table += f"| {row['feature']} | {row['category']} | {row['total']:,} | {row['FN_rate']:.3f} | {row['FN_rate_ratio']:.1f}x |\n"
    
    # 相関分析の解釈
    fp_corr = correlation_results.get('FP (誤検知)', {}).get('avg_correlation', 0)
    fn_corr = correlation_results.get('FN (見逃し)', {}).get('avg_correlation', 0)
    
    if fp_corr > 0.7:
        fp_interpretation = "全モデルが同じ間違いをしており、**データ自体の予測難易度が高い**と考えられます。"
    elif fp_corr < 0.4:
        fp_interpretation = "特定のモデルだけが間違えている傾向があり、**そのモデルの改善余地**があります。"
    else:
        fp_interpretation = "中程度の相関があり、モデル間で部分的に共通した間違いをしています。"
    
    if fn_corr > 0.7:
        fn_interpretation = "全モデルが同じ事例を見逃しており、**観測不可能な要因**がある可能性があります。"
    elif fn_corr < 0.4:
        fn_interpretation = "特定のモデルだけが見逃しており、**アンサンブルの多様性**で改善できる可能性があります。"
    else:
        fn_interpretation = "中程度の相関があり、モデル間で部分的に共通した見逃しをしています。"
    
    report = f"""# 4モデルアンサンブル 誤差分析レポート (v3)

## 概要

この分析は、LightGBM/CatBoost/MLP/TabNetの4モデルアンサンブルにおける
予測誤りのパターンを特定し、今後のモデル改善方針を決定することを目的としています。

## 分析対象データ

- **データソース**: honhyo_for_analysis_with_traffic_hospital_no_leakage.csv
- **OOF予測数**: {total:,} 件
- **最適閾値**: {threshold:.4f} (F1スコア: {best_f1:.4f})

## 分類結果サマリー

![混同行列](confusion_matrix.png)

| 分類 | 件数 | 全体比率 |
|------|------|----------|
| True Positive (正しく検出) | {stats['TP']:,} | {stats['TP']/total*100:.2f}% |
| True Negative (正しく棄却) | {stats['TN']:,} | {stats['TN']/total*100:.2f}% |
| **False Positive (誤検知)** | **{stats['FP']:,}** | **{stats['FP']/total*100:.2f}%** |
| **False Negative (見逃し)** | **{stats['FN']:,}** | **{stats['FN']/total*100:.2f}%** |

- **Precision**: {precision:.4f}
- **Recall**: {recall:.4f}
- **F1 Score**: {best_f1:.4f}

## � モデル間相関分析

エラー事例におけるモデル予測値の相関を分析することで、エラーの原因を特定できます。

![モデル相関](model_correlation.png)

### False Positive (誤検知) の相関
- **平均相関係数**: {fp_corr:.3f}
- **解釈**: {fp_interpretation}

### False Negative (見逃し) の相関
- **平均相関係数**: {fn_corr:.3f}
- **解釈**: {fn_interpretation}

## 📊 予測確率分布

![予測分布](prediction_distribution.png)

## �🔴 高FPリスク特徴 (誤検知が多発するパターン)

![特徴量エラー率](feature_error_rates.png)

| 特徴量 | カテゴリ | サンプル数 | FP率 | 全体比 |
|--------|----------|------------|------|--------|
{fp_risk_table}

## 🟠 高FNリスク特徴 (見逃しが多発するパターン)

| 特徴量 | カテゴリ | サンプル数 | FN率 | 全体比 |
|--------|----------|------------|------|--------|
{fn_risk_table}

## モデル間の予測不一致

FP/FN発生時に、各モデルがどのような予測をしていたかを分析しました。

![モデル不一致](model_disagreement.png)

| モデル | FP時の平均予測 | FN時の平均予測 |
|--------|----------------|----------------|
| LightGBM | {model_disagreement['FP'].get('LightGBM', 0):.4f} | {model_disagreement['FN'].get('LightGBM', 0):.4f} |
| CatBoost | {model_disagreement['FP'].get('CatBoost', 0):.4f} | {model_disagreement['FN'].get('CatBoost', 0):.4f} |
| MLP | {model_disagreement['FP'].get('MLP', 0):.4f} | {model_disagreement['FN'].get('MLP', 0):.4f} |
| TabNet | {model_disagreement['FP'].get('TabNet', 0):.4f} | {model_disagreement['FN'].get('TabNet', 0):.4f} |

## 生成ファイル

- `confusion_matrix.png`: 混同行列ヒートマップ
- `prediction_distribution.png`: FP/FN vs TP/TNの予測確率分布
- `model_correlation.png`: エラー事例におけるモデル間相関
- `model_disagreement.png`: モデル間の予測値比較
- `feature_error_rates.png`: 特徴量別エラー率グラフ
- `false_positives_all.csv`: 全FP事例（特徴量付き）
- `false_negatives_all.csv`: 全FN事例（特徴量付き）
- `feature_error_analysis.csv`: 特徴量別エラー率の詳細

## 次のステップ (推奨)

1. **高相関エラーへの対処**: モデル間相関が高いエラーは、新しい特徴量の追加や外部データの活用で対処
2. **低相関エラーへの対処**: 特定モデルの問題は、そのモデルのハイパーパラメータ調整や特徴量選択で対処
3. **高リスク特徴への MoE**: 見逃しやすいパターンに特化したExpertモデルを追加
4. **閾値調整**: 用途に応じてRecall/Precisionのトレードオフを調整
"""
    
    with open(output_dir / "error_analysis_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"   保存: error_analysis_report.md")


def main():
    """メイン処理"""
    print("=" * 70)
    print(" 🔍 4モデルアンサンブル 誤差分析 (v3: 強化版)")
    print("=" * 70)
    
    # データ読み込み・紐付け
    df = load_and_align_data()
    
    # 最適閾値の決定
    y_true = df['fatal'].values
    y_prob = df['pred_ensemble'].values
    
    threshold, best_f1 = find_optimal_threshold(y_true, y_prob)
    print(f"\n🎯 最適閾値 (F1最大化): {threshold:.4f} (F1: {best_f1:.4f})")
    
    # FP/FN抽出
    tp_mask, tn_mask, fp_mask, fn_mask, stats = extract_hard_examples(df, threshold)
    
    # 可視化・分析
    plot_confusion_matrix(stats, OUTPUT_DIR)
    analyze_prediction_distribution(df, tp_mask, tn_mask, fp_mask, fn_mask, OUTPUT_DIR)
    correlation_results = analyze_model_correlation(df, tp_mask, tn_mask, fp_mask, fn_mask, OUTPUT_DIR)
    model_disagreement = analyze_model_disagreement(df, fp_mask, fn_mask, OUTPUT_DIR)
    feature_results, high_fp_risk, high_fn_risk = analyze_feature_distribution(
        df, tp_mask, tn_mask, fp_mask, fn_mask, OUTPUT_DIR
    )
    plot_feature_error_rates(feature_results, OUTPUT_DIR)
    
    # 結果保存
    save_hard_examples_with_features(df, fp_mask, fn_mask, OUTPUT_DIR)
    generate_report(df, stats, threshold, best_f1, high_fp_risk, high_fn_risk, 
                    model_disagreement, correlation_results, OUTPUT_DIR)
    
    # 結果サマリー保存 (numpy型をPython型に変換)
    def convert_to_native(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        return obj
    
    summary = {
        'threshold': float(threshold),
        'best_f1': float(best_f1),
        'stats': convert_to_native(stats),
        'correlation_results': convert_to_native({k: v for k, v in correlation_results.items()}),
        'high_fp_risk_count': int(len(high_fp_risk)),
        'high_fn_risk_count': int(len(high_fn_risk)),
    }
    with open(OUTPUT_DIR / "analysis_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 70)
    print(" ✅ 分析完了！")
    print(f"    出力先: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
