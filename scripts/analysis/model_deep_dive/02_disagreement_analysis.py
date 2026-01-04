"""
モデル間不一致分析 (Disagreement Analysis)
===========================================
Stackingモデルの成功要因を解明する。

【機能】
- TabNet vs CatBoost の予測不一致を抽出
- Stackingが各モデルのどちらを採用したか分析
- 不一致ケースの正答率を算出

【出力】
- モデル間比較テーブル
- 不一致パターン分析結果

使用方法:
    python scripts/analysis/model_deep_dive/02_disagreement_analysis.py
    python scripts/analysis/model_deep_dive/02_disagreement_analysis.py --threshold 0.1
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
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import json
import argparse

# ========================================
# フォント設定（クロスプラットフォーム対応）
# ========================================
def setup_japanese_font():
    """日本語フォントを設定（環境に応じてフォールバック）"""
    try:
        import japanize_matplotlib
        return True
    except ImportError:
        pass
    
    import platform
    if platform.system() == 'Windows':
        try:
            plt.rcParams['font.family'] = 'MS Gothic'
            return True
        except:
            pass
    
    for font in ['IPAexGothic', 'IPAPGothic', 'Noto Sans CJK JP', 'DejaVu Sans']:
        try:
            plt.rcParams['font.family'] = font
            return True
        except:
            continue
    
    return False


# ========================================
# 定数
# ========================================
RANDOM_SEED = 42

# パス設定
DATA_DIR = Path("data")
SPATIO_TEMPORAL_DIR = DATA_DIR / "spatio_temporal"
RESULTS_DIR = Path("results")

SINGLE_STAGE_DIR = RESULTS_DIR / "spatio_temporal_ensemble"
TWO_STAGE_DIR = RESULTS_DIR / "twostage_spatiotemporal_ensemble"
STACKING_DIR = RESULTS_DIR / "stage3_stacking"

OUTPUT_DIR = RESULTS_DIR / "analysis" / "model_deep_dive" / "disagreement"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_optimal_threshold(y_true, y_pred_proba, method='f1'):
    """最適な閾値を自動計算"""
    thresholds = np.arange(0.01, 0.5, 0.01)
    best_score = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


def load_all_predictions():
    """全モデルの予測値を統合ロード"""
    print("[DATA] Loading predictions...")
    
    # Single-Stage
    single_preds = pd.read_csv(SINGLE_STAGE_DIR / "test_predictions.csv")
    single_preds = single_preds.rename(columns={
        'lgbm': 'single_lgbm', 'catboost': 'single_catboost',
        'mlp': 'single_mlp', 'tabnet': 'single_tabnet', 'ensemble': 'single_ensemble'
    })
    print(f"   Single-Stage: {len(single_preds):,} records")
    
    # Two-Stage
    two_stage_preds = pd.read_csv(TWO_STAGE_DIR / "test_predictions.csv")
    two_stage_preds = two_stage_preds.rename(columns={
        'lgbm': 'twostage_lgbm', 'catboost': 'twostage_catboost',
        'mlp': 'twostage_mlp', 'tabnet': 'twostage_tabnet', 'ensemble': 'twostage_ensemble'
    })
    print(f"   Two-Stage: {len(two_stage_preds):,} records (Hard Samples)")
    
    # Stacking
    stacking_preds = pd.read_csv(STACKING_DIR / "test_predictions.csv")
    print(f"   Stacking: {len(stacking_preds):,} records")
    
    # マージ
    df = single_preds[['original_index', 'single_tabnet', 'single_catboost', 'single_ensemble']].copy()
    
    twostage_cols = ['original_index', 'twostage_catboost', 'twostage_ensemble']
    if 'target' in two_stage_preds.columns:
        twostage_cols.append('target')
    
    try:
        df = df.merge(two_stage_preds[twostage_cols], on='original_index', how='left', validate='1:1')
    except pd.errors.MergeError:
        print("   [WARN] Two-Stage merge: duplicate detected. Disabling validate...")
        df = df.merge(two_stage_preds[twostage_cols], on='original_index', how='left')
    
    stacking_cols = ['original_index', 'stacking_prob']
    if 'target' in stacking_preds.columns:
        stacking_cols.append('target')
    
    try:
        df = df.merge(stacking_preds[stacking_cols], on='original_index', how='left', validate='1:1')
    except pd.errors.MergeError:
        print("   [WARN] Stacking merge: duplicate detected. Disabling validate...")
        df = df.merge(stacking_preds[stacking_cols], on='original_index', how='left')
    
    # target 統合
    if 'target_x' in df.columns:
        df['target'] = df['target_x'].combine_first(df['target_y'])
        df = df.drop(columns=['target_x', 'target_y'], errors='ignore')
    
    # 欠損補完
    if 'twostage_catboost' in df.columns:
        df['twostage_catboost'] = df['twostage_catboost'].fillna(df['single_catboost'])
    
    df['is_hard_sample'] = df['original_index'].isin(two_stage_preds['original_index']).astype(int)
    
    print(f"   Merged result: {len(df):,} records")
    print(f"   Hard Sample rate: {df['is_hard_sample'].mean():.1%}")
    
    return df


def classify_disagreement(df, threshold=None, auto_optimize=True):
    """TabNet と CatBoost の判断一致/不一致を分類（ベクトル化版）"""
    print("\n[ANALYSIS] Disagreement analysis...")
    
    df = df.copy()
    
    # 閾値決定
    if threshold is None:
        if auto_optimize:
            y_true = df['target'].values
            y_pred_proba = df['stacking_prob'].values
            threshold, score = find_optimal_threshold(y_true, y_pred_proba)
            print(f"   Threshold auto-optimized (F1 max): {threshold:.3f} (F1={score:.4f})")
        else:
            mortality_rate = (df['target'] == 1).mean()
            threshold = max(0.05, min(0.2, mortality_rate * 2))
            print(f"   [WARN] No threshold specified. Using mortality-based: {threshold:.3f}")
    else:
        print(f"   Specified threshold: {threshold:.3f}")
    
    # 2値化
    df['tabnet_pred'] = (df['single_tabnet'] >= threshold).astype(int)
    df['catboost_pred'] = (df['twostage_catboost'] >= threshold).astype(int)
    df['stacking_pred'] = (df['stacking_prob'] >= threshold).astype(int)
    
    # ベクトル化によるパターン分類
    conditions = [
        (df['tabnet_pred'] == 1) & (df['catboost_pred'] == 1),
        (df['tabnet_pred'] == 0) & (df['catboost_pred'] == 0),
        (df['tabnet_pred'] == 1) & (df['catboost_pred'] == 0),
    ]
    choices = ['Both Positive', 'Both Negative', 'TabNet Only']
    df['agreement_pattern'] = np.select(conditions, choices, default='CatBoost Only')
    
    # 統計
    pattern_counts = df['agreement_pattern'].value_counts()
    print("\n   Pattern distribution:")
    for pattern, count in pattern_counts.items():
        pct = count / len(df) * 100
        print(f"      {pattern}: {count:,} ({pct:.1f}%)")
    
    disagreement_rate = df['agreement_pattern'].isin(['TabNet Only', 'CatBoost Only']).mean()
    print(f"\n   Disagreement rate: {disagreement_rate:.1%}")
    
    return df, threshold


def analyze_disagreement_accuracy(df):
    """不一致ケースでのStackingの判断精度を分析"""
    print("\n[ACCURACY] Stacking accuracy on disagreement cases...")
    
    results = []
    
    for pattern in ['Both Positive', 'Both Negative', 'TabNet Only', 'CatBoost Only']:
        subset = df[df['agreement_pattern'] == pattern]
        if len(subset) == 0:
            continue
        
        total = len(subset)
        fatal = (subset['target'] == 1).sum()
        stacking_positive = (subset['stacking_pred'] == 1).sum()
        correct = (subset['stacking_pred'] == subset['target']).sum()
        accuracy = correct / total
        
        tp = ((subset['stacking_pred'] == 1) & (subset['target'] == 1)).sum()
        fn = ((subset['stacking_pred'] == 0) & (subset['target'] == 1)).sum()
        recall = tp / fatal if fatal > 0 else np.nan
        
        results.append({
            'pattern': pattern, 'total': total, 'fatal': fatal,
            'stacking_positive': stacking_positive, 'accuracy': accuracy,
            'tp': tp, 'fn': fn, 'recall': recall
        })
        
        print(f"\n   {pattern}:")
        print(f"      Count: {total:,} (fatal: {fatal})")
        print(f"      Stacking positive: {stacking_positive:,}")
        print(f"      Accuracy: {accuracy:.1%}")
        if not np.isnan(recall):
            print(f"      Recall: {recall:.1%}")
    
    return pd.DataFrame(results)


def analyze_who_wins(df):
    """不一致時に「誰が正しかったか」を分析"""
    print("\n[WINNER] Who was right in disagreements...")
    
    disagreement_df = df[df['agreement_pattern'].isin(['TabNet Only', 'CatBoost Only'])]
    
    if len(disagreement_df) == 0:
        print("   No disagreement cases")
        return None
    
    results = []
    
    for pattern in ['TabNet Only', 'CatBoost Only']:
        subset = disagreement_df[disagreement_df['agreement_pattern'] == pattern]
        if len(subset) == 0:
            continue
        
        total = len(subset)
        fatal = (subset['target'] == 1).sum()
        
        stacking_aggressive = (subset['stacking_pred'] == 1).sum()
        stacking_conservative = (subset['stacking_pred'] == 0).sum()
        stacking_correct = (subset['stacking_pred'] == subset['target']).sum()
        
        if pattern == 'TabNet Only':
            aggressive_model = 'TabNet'
            aggressive_correct = ((subset['target'] == 1)).sum()
        else:
            aggressive_model = 'CatBoost'
            aggressive_correct = ((subset['target'] == 1)).sum()
        
        results.append({
            'pattern': pattern, 'total': total, 'fatal': fatal,
            'aggressive_model': aggressive_model,
            'aggressive_correct': aggressive_correct,
            'stacking_aggressive': stacking_aggressive,
            'stacking_conservative': stacking_conservative,
            'stacking_correct': stacking_correct
        })
        
        print(f"\n   {pattern} (total: {total:,}, fatal: {fatal}):")
        print(f"      {aggressive_model} aggressive correct: {aggressive_correct:,}")
        print(f"      Stacking aggressive: {stacking_aggressive:,}")
        print(f"      Stacking conservative: {stacking_conservative:,}")
        print(f"      Stacking correct: {stacking_correct:,}")
    
    return pd.DataFrame(results)


def plot_disagreement_analysis(df, disagreement_results, output_dir):
    """不一致分析の可視化"""
    print("\n[PLOT] Creating disagreement plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. パターン分布
    ax1 = axes[0]
    pattern_counts = df['agreement_pattern'].value_counts()
    colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
    pattern_counts.plot(kind='bar', ax=ax1, color=colors[:len(pattern_counts)])
    ax1.set_title('Model Agreement Patterns', fontsize=14)
    ax1.set_xlabel('Pattern')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. パターン別Recall
    ax2 = axes[1]
    if disagreement_results is not None and len(disagreement_results) > 0:
        valid_results = disagreement_results[disagreement_results['fatal'] > 0]
        if len(valid_results) > 0:
            x_pos = range(len(valid_results))
            ax2.bar(x_pos, valid_results['recall'], color=colors[:len(valid_results)])
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(valid_results['pattern'], rotation=45, ha='right')
            ax2.set_title('Stacking Recall by Pattern', fontsize=14)
            ax2.set_ylabel('Recall')
            ax2.set_ylim(0, 1)
            
            for i, (_, row) in enumerate(valid_results.iterrows()):
                if not np.isnan(row['recall']):
                    ax2.text(i, row['recall'] + 0.02, f"{row['recall']:.1%}", ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / "disagreement_patterns.png", dpi=150)
    plt.close()
    print(f"   Saved: {output_dir / 'disagreement_patterns.png'}")


def generate_report(df, disagreement_results, winner_results, threshold, output_dir):
    """分析レポート生成"""
    print("\n[REPORT] Generating report...")
    
    total = len(df)
    disagreement_count = df['agreement_pattern'].isin(['TabNet Only', 'CatBoost Only']).sum()
    disagreement_rate = disagreement_count / total
    
    report = f"""# Model Disagreement Analysis Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 1. Analysis Parameters

| Item | Value |
|:---|---:|
| **Threshold** | **{threshold:.4f}** |
| Total records | {total:,} |

## 2. Pattern Distribution

| Pattern | Count | Ratio |
|:---|---:|---:|
"""
    
    pattern_counts = df['agreement_pattern'].value_counts()
    for pattern, count in pattern_counts.items():
        pct = count / total * 100
        report += f"| {pattern} | {count:,} | {pct:.1f}% |\n"
    
    report += f"""
**Disagreement rate: {disagreement_rate:.1%}** ({disagreement_count:,} / {total:,})

## 3. Stacking Accuracy on Disagreement Cases

"""
    
    if disagreement_results is not None and len(disagreement_results) > 0:
        report += "| Pattern | Count | Fatal | Stacking Accuracy | Recall |\n"
        report += "|:---|---:|---:|---:|---:|\n"
        for _, row in disagreement_results.iterrows():
            recall_str = f"{row['recall']:.1%}" if not np.isnan(row['recall']) else "N/A"
            report += f"| {row['pattern']} | {row['total']:,} | {row['fatal']} | {row['accuracy']:.1%} | {recall_str} |\n"
    
    if winner_results is not None and len(winner_results) > 0:
        report += """
## 4. Aggressive vs Conservative Model

| Pattern | Aggressive Model | Fatal | Aggressive Correct | Stacking Decision |
|:---|:---:|---:|---:|:---|
"""
        for _, row in winner_results.iterrows():
            stacking_choice = f"Agg:{row['stacking_aggressive']}, Cons:{row['stacking_conservative']}"
            report += f"| {row['pattern']} | {row['aggressive_model']} | {row['fatal']} | {row['aggressive_correct']} | {stacking_choice} |\n"
    
    report += """
## 5. Findings

"""
    
    if disagreement_rate > 0.1:
        report += f"> Model disagreement rate is **{disagreement_rate:.1%}**. Ensemble effect is expected.\n\n"
    
    both_positive = pattern_counts.get('Both Positive', 0)
    if both_positive > 0:
        report += f"> Both models agreed on positive: **{both_positive:,}** cases. High confidence alerts.\n\n"
    
    report += """## 6. Visualizations

- `disagreement_patterns.png`: Pattern distribution and accuracy comparison
"""
    
    with open(output_dir / "disagreement_analysis_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"   Saved: {output_dir / 'disagreement_analysis_report.md'}")
    
    if disagreement_results is not None:
        disagreement_results.to_csv(output_dir / "disagreement_statistics.csv", index=False)
    if winner_results is not None:
        winner_results.to_csv(output_dir / "winner_analysis.csv", index=False)
    
    with open(output_dir / "analysis_config.json", 'w', encoding='utf-8') as f:
        json.dump({'threshold': float(threshold), 'generated_at': datetime.now().isoformat()}, 
                 f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Model Disagreement Analysis')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Prediction threshold (default: auto F1 optimization)')
    parser.add_argument('--no-auto-optimize', action='store_true',
                        help='Disable threshold auto-optimization')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Model Disagreement Analysis")
    print("=" * 70)
    
    setup_japanese_font()
    
    df = load_all_predictions()
    
    auto_optimize = not args.no_auto_optimize
    df, threshold = classify_disagreement(df, threshold=args.threshold, auto_optimize=auto_optimize)
    
    disagreement_results = analyze_disagreement_accuracy(df)
    winner_results = analyze_who_wins(df)
    
    plot_disagreement_analysis(df, disagreement_results, OUTPUT_DIR)
    generate_report(df, disagreement_results, winner_results, threshold, OUTPUT_DIR)
    
    print("\n" + "=" * 70)
    print("[DONE] Model Disagreement Analysis completed!")
    print(f"   Output: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
