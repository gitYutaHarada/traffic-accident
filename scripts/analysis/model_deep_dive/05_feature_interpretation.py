"""
特徴量解釈分析 (Feature Interpretation)
=====================================
SHAPで重要と判定された特徴量について、
「具体的にどのような値がリスクを高めるのか」を分析する。

【機能】
- SHAP分析と同様の代理モデルを学習
- 上位特徴量について、元の値（Raw Data）ごとのSHAP値（リスク寄与度）を集計
- 具体的な危険パターンを出力

使用方法:
    python scripts/analysis/model_deep_dive/05_feature_interpretation.py
"""

import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path
import lightgbm as lgb
import shap
from sklearn.preprocessing import LabelEncoder

RANDOM_SEED = 42
SAMPLE_SIZE = 10000

DATA_DIR = Path("data")
SPATIO_TEMPORAL_DIR = DATA_DIR / "spatio_temporal"
RESULTS_DIR = Path("results")
STACKING_DIR = RESULTS_DIR / "stage3_stacking"
OUTPUT_DIR = RESULTS_DIR / "analysis" / "model_deep_dive" / "interpretation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 分析対象のペア (Preprocessed名 -> Raw名)
TARGET_FEATURES = {
    '速度規制（指定のみ）（当事者B）_scaled': '速度規制（指定のみ）（当事者B）',
    '車道幅員_scaled': '車道幅員',
    'geohash_accidents_past_365d_scaled': 'geohash_accidents_past_365d',
    '当事者種別（当事者A）_te': '当事者種別（当事者A）',
    'hour_cos_scaled': 'hour',  # hourから変換されていると推測
    '都道府県コード_te': '都道府県コード',
}

def load_data():
    print("[DATA] Loading...")
    
    # Preprocessed (X)
    test_df = pd.read_parquet(SPATIO_TEMPORAL_DIR / "preprocessed_test.parquet")
    stacking_preds = pd.read_csv(STACKING_DIR / "test_predictions.csv")
    
    # Raw (for interpretation)
    raw_test = pd.read_parquet(SPATIO_TEMPORAL_DIR / "raw_test.parquet")
    
    # Align lengths
    min_len = min(len(test_df), len(stacking_preds), len(raw_test))
    test_df = test_df.iloc[:min_len].reset_index(drop=True)
    stacking_preds = stacking_preds.iloc[:min_len].reset_index(drop=True)
    raw_test = raw_test.iloc[:min_len].reset_index(drop=True)
    
    # Add target for training
    y = raw_test['fatal'].values if 'fatal' in raw_test.columns else np.zeros(len(test_df))
    stacking_prob = stacking_preds['stacking_prob'].values
    
    return test_df, raw_test, y, stacking_prob

def train_surrogate_model(X, y):
    print("[MODEL] Training surrogate LightGBM...")
    
    # 数値カラムのみ使用（SHAPと同じ前処理）
    exclude = ['target', 'stacking_prob', 'fatal']
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude]
    
    X_train = X[feature_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 学習用データ（簡易的にテストデータ自身で学習してフィッティングさせる）
    # 本来はTrainデータを使うべきだが、傾向を見るだけならこれでも可
    # しかし、fatal除外の徹底のため、03_shap_analysis.pyと同じ構成にする
    
    train_df = pd.read_parquet(SPATIO_TEMPORAL_DIR / "preprocessed_train.parquet")
    raw_train = pd.read_parquet(SPATIO_TEMPORAL_DIR / "raw_train.parquet")
    
    X_train_real = train_df[feature_cols].copy().replace([np.inf, -np.inf], np.nan).fillna(0)
    y_train_real = raw_train['fatal'].values
    
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        class_weight='balanced',
        random_state=RANDOM_SEED,
        verbose=-1
    )
    model.fit(X_train_real, y_train_real)
    
    return model, X_train, feature_cols

def analyze_feature_trend(shap_values, feature_vals_raw, feature_name_raw, is_categorical=False):
    """
    特徴量の生値とSHAP値の関係を分析
    """
    df = pd.DataFrame({
        'raw_value': feature_vals_raw,
        'shap_value': shap_values
    })
    
    # Remove NaN
    df = df.dropna()
    
    if is_categorical or df['raw_value'].nunique() < 20:
        # カテゴリ別平均
        agg = df.groupby('raw_value')['shap_value'].agg(['mean', 'count', 'std'])
        agg = agg[agg['count'] > 10].sort_values('mean', ascending=False)
        return agg
    else:
        # 数値などの場合、ビン分割して傾向を見る
        try:
            df['bin'] = pd.qcut(df['raw_value'], q=10, duplicates='drop')
            agg = df.groupby('bin')['shap_value'].mean()
            return agg
        except:
            # qcut失敗時は単純に上位下位で
            high = df[df['raw_value'] > df['raw_value'].median()]['shap_value'].mean()
            low = df[df['raw_value'] <= df['raw_value'].median()]['shap_value'].mean()
            return f"High value mean SHAP: {high:.4f}, Low value mean SHAP: {low:.4f}"

def main():
    print("="*60)
    print("Feature Interpretation Analysis")
    print("="*60)
    
    test_df, raw_test, y, stacking_prob = load_data()
    model, X_test_processed, feature_cols = train_surrogate_model(test_df, y)
    
    # Sampling for SHAP
    if len(X_test_processed) > SAMPLE_SIZE:
        idx = np.random.RandomState(RANDOM_SEED).choice(len(X_test_processed), SAMPLE_SIZE, replace=False)
        X_sample = X_test_processed.iloc[idx]
        raw_sample = raw_test.iloc[idx]
    else:
        X_sample = X_test_processed
        raw_sample = raw_test
        idx = np.arange(len(X_test_processed))
        
    print(f"\n[SHAP] Computing SHAP values (sample: {len(X_sample)})...")
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_sample)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    
    print("\n[ANALYSIS] Interpreting top features...")
    
    results_txt = ""
    
    for processed_name, raw_name in TARGET_FEATURES.items():
        if processed_name not in X_sample.columns:
            print(f"   [SKIP] {processed_name} not found in features")
            continue
        
        if raw_name not in raw_sample.columns:
            # hourなどはrawにない可能性がある
            if raw_name == 'hour' and 'date' in raw_sample.columns:
                 # dateからhour作成などが必要だが、簡易的に省略または推測
                 pass
            print(f"   [SKIP] {raw_name} not found in raw data")
            continue

        print(f"\n   Analyzing: {raw_name} (from {processed_name})")
        results_txt += f"\n### {raw_name}\n"
        
        col_idx = X_sample.columns.get_loc(processed_name)
        vals_shap = shap_vals[:, col_idx]
        vals_raw = raw_sample[raw_name].values
        
        # カテゴリ判定
        is_cat = raw_sample[raw_name].dtype == 'object' or raw_sample[raw_name].nunique() < 50
        
        agg = analyze_feature_trend(vals_shap, vals_raw, raw_name, is_categorical=is_cat)
        
        print(agg)
        if isinstance(agg, pd.DataFrame):
            results_txt += agg.to_markdown() + "\n"
        else:
            results_txt += str(agg) + "\n"

    with open(OUTPUT_DIR / "interpretation_result.md", "w", encoding='utf-8') as f:
        f.write(results_txt)
    
    print(f"\nSaved results to {OUTPUT_DIR / 'interpretation_result.md'}")

if __name__ == "__main__":
    main()
