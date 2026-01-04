"""
GBDTベースラインスクリプト (Phase 2: Model Comparison)
=====================================================
Spatio-Temporal特徴量を用いてLightGBMベースラインを学習し、
MLPと比較する。
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, 
    precision_score, recall_score, f1_score, brier_score_loss
)
import matplotlib.pyplot as plt

# --- パス設定 ---
DATA_DIR = Path("data/spatio_temporal")
RESULTS_DIR = Path("results/spatio_temporal")
OUTPUT_DIR = RESULTS_DIR / "gbdt_baseline"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'


def load_data():
    """前処理済みデータを読み込む"""
    print("📂 データ読み込み中...")
    train_df = pd.read_parquet(DATA_DIR / "preprocessed_train.parquet")
    val_df = pd.read_parquet(DATA_DIR / "preprocessed_val.parquet")
    test_df = pd.read_parquet(DATA_DIR / "preprocessed_test.parquet")
    print(f"   Train: {len(train_df):,} rows")
    print(f"   Val:   {len(val_df):,} rows")
    print(f"   Test:  {len(test_df):,} rows")
    return train_df, val_df, test_df


def prepare_features(train_df, val_df, test_df):
    """特徴量を準備"""
    exclude_cols = ['fatal', 'geohash', 'geohash_fine', 'date', 'year', 'node_id']
    feature_cols = [
        c for c in train_df.columns 
        if c not in exclude_cols and train_df[c].dtype in ['int64', 'float64', 'float32', 'int32']
    ]
    
    X_train = train_df[feature_cols]
    y_train = train_df['fatal']
    
    X_val = val_df[feature_cols]
    y_val = val_df['fatal']
    
    X_test = test_df[feature_cols]
    y_test = test_df['fatal']
    
    print(f"   特徴量数: {len(feature_cols)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def train_lightgbm(X_train, y_train, X_val, y_val):
    """LightGBMモデルを学習"""
    print("\n🚀 LightGBM学習中...")
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    # ハイパーパラメータ
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1,
        'random_state': 42,
        'is_unbalance': True,  # クラス不均衡対応
    }
    
    model = lgb.train(
        params, dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(100)
        ]
    )
    
    print(f"   Best iteration: {model.best_iteration}")
    return model


def evaluate_model(model, X_test, y_test, feature_cols):
    """モデルを評価"""
    print("\n📊 モデル評価中...")
    
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # 基本指標
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    
    # Top-k Precision
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    top_k_results = {}
    for k in [100, 500, 1000]:
        top_k_idx = sorted_indices[:k]
        top_k_precision = y_test.iloc[top_k_idx].sum() / k
        top_k_results[f'precision_at_{k}'] = float(top_k_precision)
        top_k_results[f'recall_at_{k}'] = float(y_test.iloc[top_k_idx].sum() / y_test.sum())
    
    # 閾値別のスコア
    for thresh in [0.3, 0.5, 0.7]:
        y_pred_t = (y_pred_proba >= thresh).astype(int)
        p = precision_score(y_test, y_pred_t, zero_division=0)
        r = recall_score(y_test, y_pred_t, zero_division=0)
        f1 = f1_score(y_test, y_pred_t, zero_division=0)
        top_k_results[f'precision_at_{thresh}'] = float(p)
        top_k_results[f'recall_at_{thresh}'] = float(r)
        top_k_results[f'f1_at_{thresh}'] = float(f1)
    
    # 特定Recallでの閾値
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_proba)
    for target_recall in [0.99, 0.95, 0.90]:
        idx = np.searchsorted(recall_curve[::-1], target_recall)
        if idx < len(thresholds):
            thresh = thresholds[::-1][idx] if idx < len(thresholds) else 0.0
            prec = precision_curve[::-1][idx] if idx < len(precision_curve) else 0.0
            top_k_results[f'threshold_at_recall_{int(target_recall*100)}'] = float(thresh)
            top_k_results[f'precision_at_recall_{int(target_recall*100)}'] = float(prec)
    
    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier_score': brier,
        **top_k_results
    }
    
    print(f"\n   ROC-AUC: {roc_auc:.4f}")
    print(f"   PR-AUC:  {pr_auc:.4f}")
    print(f"   Top-100 Precision: {top_k_results['precision_at_100']:.1%}")
    print(f"   Top-500 Precision: {top_k_results['precision_at_500']:.1%}")
    print(f"   Top-1000 Precision: {top_k_results['precision_at_1000']:.1%}")
    
    # 特徴量重要度
    importance = model.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return metrics, importance_df


def compare_with_mlp(metrics):
    """MLPと結果を比較"""
    print("\n" + "=" * 70)
    print(" 🔄 MLP vs LightGBM 比較")
    print("=" * 70)
    
    mlp_path = RESULTS_DIR / "results_mlp.json"
    if not mlp_path.exists():
        print("   ⚠️ MLP結果ファイルが見つかりません")
        return None
    
    with open(mlp_path, 'r') as f:
        mlp_results = json.load(f)
    
    mlp_metrics = mlp_results['test_metrics']
    
    comparisons = [
        ('ROC-AUC', 'roc_auc', float(mlp_metrics.get('roc_auc', 0))),
        ('PR-AUC', 'pr_auc', float(mlp_metrics.get('pr_auc', 0))),
        ('Top-100 Precision', 'precision_at_100', float(mlp_metrics.get('precision_at_100', 0))),
        ('Top-500 Precision', 'precision_at_500', float(mlp_metrics.get('precision_at_500', 0))),
    ]
    
    print(f"\n   {'指標':<25} {'MLP':<15} {'LightGBM':<15} {'差分':<10}")
    print("   " + "-" * 65)
    
    comparison_results = {}
    for name, key, mlp_val in comparisons:
        lgb_val = metrics.get(key, 0)
        diff = lgb_val - mlp_val
        diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        print(f"   {name:<25} {mlp_val:<15.4f} {lgb_val:<15.4f} {diff_str:<10}")
        comparison_results[key] = {'mlp': mlp_val, 'lgb': lgb_val, 'diff': diff}
    
    return comparison_results


def save_results(model, metrics, importance_df, comparison_results):
    """結果を保存"""
    print("\n💾 結果を保存中...")
    
    # モデル保存
    model.save_model(str(OUTPUT_DIR / "lightgbm_model.txt"))
    
    # メトリクス保存
    results = {
        'model_type': 'lightgbm',
        'test_metrics': metrics,
        'comparison_with_mlp': comparison_results
    }
    with open(OUTPUT_DIR / "results_lightgbm.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # 特徴量重要度保存
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    
    # プロット
    plt.figure(figsize=(12, 8))
    top20 = importance_df.head(20)
    plt.barh(top20['feature'][::-1], top20['importance'][::-1])
    plt.xlabel('Importance (Gain)')
    plt.title('Top 20 Feature Importance (LightGBM Baseline)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close()
    
    print(f"   結果保存先: {OUTPUT_DIR}")


def main():
    print("=" * 70)
    print(" 🌲 LightGBM Baseline (Phase 2)")
    print("=" * 70)
    
    train_df, val_df, test_df = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = prepare_features(
        train_df, val_df, test_df
    )
    
    model = train_lightgbm(X_train, y_train, X_val, y_val)
    metrics, importance_df = evaluate_model(model, X_test, y_test, feature_cols)
    comparison_results = compare_with_mlp(metrics)
    save_results(model, metrics, importance_df, comparison_results)
    
    print("\n🎉 LightGBMベースライン完了！")


if __name__ == "__main__":
    main()
