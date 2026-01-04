"""
閾値シミュレーション - 運用最適化のための閾値検討

使用方法:
    python scripts/analysis/model_deep_dive/04_threshold_simulation.py
"""

import sys, io
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
import json
import argparse

RESULTS_DIR = Path("results")
STACKING_DIR = RESULTS_DIR / "stage3_stacking"
OUTPUT_DIR = RESULTS_DIR / "analysis" / "model_deep_dive" / "threshold"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def setup_font():
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
    return False


def load_predictions():
    print("[DATA] Loading predictions...")
    df = pd.read_csv(STACKING_DIR / "test_predictions.csv")
    print(f"   Records: {len(df):,}, Fatal: {(df['target'] == 1).sum():,}")
    return df


def calc_metrics(y_true, y_proba, thresholds=None):
    if thresholds is None:
        thresholds = np.concatenate([np.arange(0.001, 0.01, 0.001), np.arange(0.01, 0.1, 0.01), np.arange(0.1, 0.51, 0.05)])
    
    total = len(y_true)
    total_fatal = (y_true == 1).sum()
    results = []
    
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        results.append({
            'threshold': t, 'tp': tp, 'fp': fp, 'fn': fn,
            'precision': prec, 'recall': rec, 'f1': f1,
            'alert_count': tp + fp, 'missed_fatal': fn
        })
    
    return pd.DataFrame(results)


def plot_curves(metrics_df, output_dir):
    print("\n[PLOT] Creating threshold curves...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax1 = axes[0, 0]
    ax1.plot(metrics_df['threshold'], metrics_df['precision'], 'b-', label='Precision', lw=2)
    ax1.plot(metrics_df['threshold'], metrics_df['recall'], 'r-', label='Recall', lw=2)
    ax1.plot(metrics_df['threshold'], metrics_df['f1'], 'g--', label='F1', lw=1.5)
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Score')
    ax1.set_title('Precision/Recall vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 0.5)
    ax1.set_ylim(0, 1)
    
    ax2 = axes[0, 1]
    ax2.plot(metrics_df['threshold'], metrics_df['alert_count'], 'purple', lw=2)
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Alert Count')
    ax2.set_title('Alert Count vs Threshold')
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.plot(metrics_df['threshold'], metrics_df['missed_fatal'], 'red', lw=2)
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Missed Fatal')
    ax3.set_title('Missed Fatal vs Threshold')
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    sc = ax4.scatter(metrics_df['recall'], metrics_df['precision'], c=metrics_df['threshold'], cmap='viridis', s=50)
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('PR Trade-off (color=threshold)')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(sc, ax=ax4, label='Threshold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_curves.png", dpi=150)
    plt.close()
    print(f"   Saved: {output_dir / 'threshold_curves.png'}")


def find_optimal(metrics_df, total_fatal, total):
    print("\n[OPTIMAL] Finding optimal thresholds...")
    scenarios = {}
    
    # Recall 95%+
    r95 = metrics_df[metrics_df['recall'] >= 0.95]
    if len(r95) > 0:
        best = r95.loc[r95['precision'].idxmax()]
        scenarios['recall_95'] = {'name': 'Recall 95%+', 'threshold': float(best['threshold']),
                                   'recall': float(best['recall']), 'precision': float(best['precision']),
                                   'alerts': int(best['alert_count']), 'missed': int(best['missed_fatal'])}
    
    # F1 max
    best_f1 = metrics_df.loc[metrics_df['f1'].idxmax()]
    scenarios['best_f1'] = {'name': 'F1 Max', 'threshold': float(best_f1['threshold']),
                             'recall': float(best_f1['recall']), 'precision': float(best_f1['precision']),
                             'alerts': int(best_f1['alert_count']), 'missed': int(best_f1['missed_fatal'])}
    
    for k, v in scenarios.items():
        print(f"   {v['name']}: threshold={v['threshold']:.3f}, recall={v['recall']:.1%}, alerts={v['alerts']:,}")
    
    return scenarios


def generate_report(metrics_df, scenarios, total_fatal, total, output_dir):
    print("\n[REPORT] Generating...")
    
    report = f"""# Threshold Simulation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Data Summary
- Total: {total:,}
- Fatal: {total_fatal:,} ({total_fatal/total*100:.2f}%)

## Recommended Thresholds
"""
    for k, v in scenarios.items():
        miss_pct = v['missed'] / total_fatal * 100 if total_fatal > 0 else 0
        report += f"""
### {v['name']}
- Threshold: {v['threshold']:.4f}
- Recall: {v['recall']:.1%}
- Precision: {v['precision']:.1%}
- Alerts: {v['alerts']:,}
- Missed: {v['missed']} ({miss_pct:.1f}%)
"""
    
    report += """
## Visualizations
- `threshold_curves.png`: Threshold analysis curves
"""
    
    with open(output_dir / "threshold_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    metrics_df.to_csv(output_dir / "threshold_metrics.csv", index=False)
    
    with open(output_dir / "optimal_thresholds.json", 'w') as f:
        json.dump(scenarios, f, indent=2)
    
    print(f"   Saved: {output_dir / 'threshold_report.md'}")


def main():
    print("=" * 60)
    print("Threshold Simulation")
    print("=" * 60)
    setup_font()
    
    df = load_predictions()
    y_true = df['target'].values
    y_proba = df['stacking_prob'].values
    total_fatal = (y_true == 1).sum()
    total = len(y_true)
    
    print("\n[CALC] Computing metrics at various thresholds...")
    metrics_df = calc_metrics(y_true, y_proba)
    print(f"   Thresholds evaluated: {len(metrics_df)}")
    
    plot_curves(metrics_df, OUTPUT_DIR)
    scenarios = find_optimal(metrics_df, total_fatal, total)
    generate_report(metrics_df, scenarios, total_fatal, total, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("[DONE] Output:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
