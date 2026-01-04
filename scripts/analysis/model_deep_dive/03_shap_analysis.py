"""
SHAP分析 - 個別予測の「なぜ」を説明する

使用方法:
    python scripts/analysis/model_deep_dive/03_shap_analysis.py
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
import argparse
import json
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import LabelEncoder

RANDOM_SEED = 42
DEFAULT_SAMPLE_SIZE = 5000

DATA_DIR = Path("data")
SPATIO_TEMPORAL_DIR = DATA_DIR / "spatio_temporal"
RESULTS_DIR = Path("results")
STACKING_DIR = RESULTS_DIR / "stage3_stacking"
OUTPUT_DIR = RESULTS_DIR / "analysis" / "model_deep_dive" / "shap"
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


def load_data():
    print("[DATA] Loading...")
    test_df = pd.read_parquet(SPATIO_TEMPORAL_DIR / "preprocessed_test.parquet")
    stacking_preds = pd.read_csv(STACKING_DIR / "test_predictions.csv")
    raw_test = pd.read_parquet(SPATIO_TEMPORAL_DIR / "raw_test.parquet")
    if 'fatal' in raw_test.columns:
        test_df['target'] = raw_test['fatal'].values
    test_df = test_df.reset_index(drop=True)
    min_len = min(len(test_df), len(stacking_preds))
    test_df = test_df.iloc[:min_len]
    test_df['stacking_prob'] = stacking_preds['stacking_prob'].values[:min_len]
    print(f"   Records: {len(test_df):,}")
    return test_df


def prepare_features(df):
    """特徴量を数値型のみに変換（カテゴリカルは除外）"""
    # 数値カラムのみを選択
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # target と stacking_prob を除外
    exclude = ['target', 'stacking_prob', 'fatal']
    feature_cols = [c for c in numeric_cols if c not in exclude]
    
    X = df[feature_cols].copy()
    
    # NaN/Inf を処理
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)
    
    print(f"   Numeric features: {len(feature_cols)}")
    return X


def train_model(X, y, stacking_prob):
    print("\n[MODEL] Training surrogate...")
    import lightgbm as lgb
    
    # 学習データも同様に処理
    train_df = pd.read_parquet(SPATIO_TEMPORAL_DIR / "preprocessed_train.parquet")
    raw_train = pd.read_parquet(SPATIO_TEMPORAL_DIR / "raw_train.parquet")
    
    # 同じカラムを選択
    train_X = train_df[X.columns].copy()
    train_X = train_X.replace([np.inf, -np.inf], np.nan)
    train_X = train_X.fillna(0)
    
    y_train = raw_train['fatal'].values
    
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        verbose=-1,
        n_jobs=-1
    )
    
    model.fit(train_X, y_train)
    
    y_pred = model.predict_proba(X)[:, 1]
    
    print("\n[FIDELITY] Checking...")
    pearson_r, _ = pearsonr(stacking_prob, y_pred)
    spearman_r, _ = spearmanr(stacking_prob, y_pred)
    mae = np.abs(stacking_prob - y_pred).mean()
    print(f"   Pearson: {pearson_r:.4f}, Spearman: {spearman_r:.4f}, MAE: {mae:.4f}")
    
    if spearman_r >= 0.7:
        print("   Fidelity: GOOD")
    else:
        print("   Fidelity: LOW - interpret with caution")
    
    fidelity = {'pearson_r': float(pearson_r), 'spearman_r': float(spearman_r), 'mae': float(mae)}
    return model, fidelity


def compute_shap(model, X, sample_size):
    print(f"\n[SHAP] Computing (sample: {sample_size})...")
    try:
        import shap
    except ImportError:
        print("   [ERROR] pip install shap")
        return None, None, None
    
    if len(X) > sample_size:
        idx = np.random.RandomState(RANDOM_SEED).choice(len(X), sample_size, replace=False)
        X_samp = X.iloc[idx].copy()
    else:
        X_samp = X.copy()
        idx = np.arange(len(X))
    
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_samp)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    print(f"   Shape: {shap_vals.shape}")
    return shap_vals, X_samp, idx


def plot_shap(shap_vals, X_samp, output_dir):
    print("\n[PLOT] Creating...")
    import shap
    
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top_n = min(20, len(mean_abs))
    top_idx = np.argsort(mean_abs)[-top_n:][::-1]
    sel_feats = [X_samp.columns[i] for i in top_idx]
    
    # Summary plot
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(shap_vals[:, top_idx], X_samp[sel_feats], show=False, max_display=top_n)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_dir / 'shap_summary_plot.png'}")
    
    # Bar plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_n), mean_abs[top_idx][::-1], color='steelblue')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([sel_feats[i] for i in range(top_n)][::-1])
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('Feature Importance by SHAP')
    plt.tight_layout()
    plt.savefig(output_dir / "shap_bar_plot.png", dpi=150)
    plt.close()
    print(f"   Saved: {output_dir / 'shap_bar_plot.png'}")
    
    # CSV
    imp_df = pd.DataFrame({'feature': X_samp.columns, 'importance': mean_abs})
    imp_df = imp_df.sort_values('importance', ascending=False)
    imp_df.to_csv(output_dir / "shap_feature_importance.csv", index=False)
    print(f"   Saved: {output_dir / 'shap_feature_importance.csv'}")
    
    return imp_df


def analyze_high_risk(model, X, stacking_prob, y, shap_vals, idx, output_dir):
    print("\n[HIGH-RISK] Analyzing...")
    s_prob = stacking_prob[idx]
    y_samp = y.iloc[idx].values if isinstance(y, pd.Series) else y[idx]
    
    high_idx = np.argsort(s_prob)[-10:][::-1]
    results = []
    for rank, i in enumerate(high_idx, 1):
        case = shap_vals[i]
        top_f = np.argsort(np.abs(case))[-3:][::-1]
        results.append({
            'rank': rank, 
            'stacking_prob': float(s_prob[i]), 
            'true_label': int(y_samp[i]),
            'top_1': str(X.columns[top_f[0]]), 
            'shap_1': float(case[top_f[0]]),
            'top_2': str(X.columns[top_f[1]]), 
            'shap_2': float(case[top_f[1]]),
            'top_3': str(X.columns[top_f[2]]), 
            'shap_3': float(case[top_f[2]]),
        })
    pd.DataFrame(results).to_csv(output_dir / "high_risk_cases.csv", index=False)
    print(f"   Saved: {output_dir / 'high_risk_cases.csv'}")
    return results


def generate_report(imp_df, fidelity, output_dir):
    print("\n[REPORT] Generating...")
    report = f"""# SHAP Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Surrogate Model Fidelity
- Pearson: {fidelity['pearson_r']:.4f}
- Spearman: {fidelity['spearman_r']:.4f}
- MAE: {fidelity['mae']:.4f}

## Top 20 Features
"""
    for i, (_, r) in enumerate(imp_df.head(20).iterrows()):
        report += f"{i+1}. {r['feature']}: {r['importance']:.4f}\n"
    
    report += """
## Files
- shap_summary_plot.png
- shap_bar_plot.png
- shap_feature_importance.csv
- high_risk_cases.csv
"""
    
    with open(output_dir / "shap_analysis_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    with open(output_dir / "config.json", 'w') as f:
        json.dump({'fidelity': fidelity, 'generated': datetime.now().isoformat()}, f, indent=2)
    print(f"   Saved: {output_dir / 'shap_analysis_report.md'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=DEFAULT_SAMPLE_SIZE)
    args = parser.parse_args()
    
    print("=" * 60)
    print("SHAP Analysis")
    print("=" * 60)
    setup_font()
    
    df = load_data()
    X = prepare_features(df)
    y = df['target'].copy()
    s_prob = df['stacking_prob'].values
    
    model, fidelity = train_model(X, y, s_prob)
    shap_vals, X_samp, idx = compute_shap(model, X, args.sample_size)
    
    if shap_vals is not None:
        imp_df = plot_shap(shap_vals, X_samp, OUTPUT_DIR)
        analyze_high_risk(model, X, s_prob, y, shap_vals, idx, OUTPUT_DIR)
        generate_report(imp_df, fidelity, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("[DONE] Output:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
