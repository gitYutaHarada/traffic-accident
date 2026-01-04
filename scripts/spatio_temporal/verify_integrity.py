"""
健全性チェックスクリプト (Phase 1: Integrity Check)
==================================================
1. LightGBMによる特徴量重要度の確認
2. Top-k 予測の空間的多様性（ユニークGeohash数）の確認
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path

# --- パス設定 ---
DATA_DIR = Path("data/spatio_temporal")
RESULTS_DIR = Path("results/spatio_temporal")
OUTPUT_DIR = RESULTS_DIR / "integrity_check"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'


def load_data():
    """前処理済みデータを読み込む"""
    print("📂 データ読み込み中...")
    train_df = pd.read_parquet(DATA_DIR / "preprocessed_train.parquet")
    test_df = pd.read_parquet(DATA_DIR / "preprocessed_test.parquet")
    print(f"   Train: {len(train_df):,} rows, Test: {len(test_df):,} rows")
    return train_df, test_df


def analyze_feature_importance(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """LightGBMで特徴量重要度を算出"""
    print("\n🔎 [1/2] 特徴量重要度を分析中...")
    
    # 特徴量の特定
    exclude_cols = ['fatal', 'geohash', 'geohash_fine', 'date', 'year', 'node_id']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols and train_df[c].dtype in ['int64', 'float64', 'float32', 'int32']]
    
    X_train = train_df[feature_cols].values
    y_train = train_df['fatal'].values
    
    # LightGBM学習
    dtrain = lgb.Dataset(X_train, label=y_train)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
    }
    model = lgb.train(
        params, dtrain,
        num_boost_round=200,
        valid_sets=[dtrain],
        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(0)]
    )
    
    # 重要度取得
    importance = model.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # 上位20件の表示
    print("\n📊 特徴量重要度 Top 20:")
    print("-" * 60)
    for i, row in importance_df.head(20).iterrows():
        pct = row['importance'] / importance_df['importance'].sum() * 100
        bar = "█" * int(pct / 2)
        print(f"   {row['feature']:40s} {pct:5.1f}% {bar}")
    
    # リーケージの懸念チェック
    leakage_keywords = ['past_30d', 'past_365d', 'fatal', 'te_', 'target_enc']
    top5_features = importance_df.head(5)['feature'].tolist()
    leakage_suspects = [f for f in top5_features if any(kw in f.lower() for kw in leakage_keywords)]
    
    if leakage_suspects:
        print("\n⚠️ 【警告】リーケージの疑いがある特徴量が上位に存在します:")
        for f in leakage_suspects:
            print(f"   - {f}")
    else:
        print("\n✅ 上位5特徴量に明らかなリーケージは見当たりません。")
    
    # 保存
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
    
    # プロット
    plt.figure(figsize=(12, 8))
    top20 = importance_df.head(20)
    plt.barh(top20['feature'][::-1], top20['importance'][::-1])
    plt.xlabel('Importance (Gain)')
    plt.title('Top 20 Feature Importance (LightGBM)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print(f"   保存: {OUTPUT_DIR / 'feature_importance.png'}")
    
    return importance_df, leakage_suspects


def analyze_spatial_diversity(test_df: pd.DataFrame):
    """Top-k予測の空間的多様性を確認"""
    print("\n🌍 [2/2] 空間的多様性を分析中...")
    
    # 予測結果の読み込み
    pred_path = RESULTS_DIR / "test_predictions.parquet"
    if not pred_path.exists():
        print(f"   ❌ 予測ファイルが見つかりません: {pred_path}")
        return None
    
    pred_df = pd.read_parquet(pred_path)
    print(f"   予測データ: {len(pred_df):,} rows")
    
    # 上位k件の分析
    k_values = [100, 500, 1000]
    results = {}
    
    # geohashカラムの確認
    geohash_col = None
    for col in ['geohash', 'geohash_fine']:
        if col in pred_df.columns:
            geohash_col = col
            break
    
    if geohash_col is None:
        # test_dfからgeohashを取得（インデックスが一致すると仮定）
        if 'geohash' in test_df.columns:
            pred_df['geohash'] = test_df['geohash'].values[:len(pred_df)]
            geohash_col = 'geohash'
        else:
            print("   ❌ geohashカラムが見つかりません。")
            return None
    
    # probまたはpredictionカラムを使う
    prob_col = 'prediction' if 'prediction' in pred_df.columns else 'prob'
    pred_df_sorted = pred_df.sort_values(prob_col, ascending=False)
    
    for k in k_values:
        top_k = pred_df_sorted.head(k)
        unique_geohash = top_k[geohash_col].nunique()
        fatal_count = top_k['label'].sum() if 'label' in top_k.columns else top_k['fatal'].sum()
        precision = fatal_count / k
        
        results[k] = {
            'total': k,
            'unique_geohash': unique_geohash,
            'diversity_ratio': unique_geohash / k,
            'fatal_count': int(fatal_count),
            'precision': precision
        }
        
        print(f"\n   Top-{k}:")
        print(f"      - ユニークGeohash数: {unique_geohash} / {k} ({unique_geohash/k*100:.1f}%)")
        print(f"      - 正解（fatal=1）: {int(fatal_count)} 件 (Precision: {precision:.1%})")
    
    # 集中度の警告
    top100_diversity = results[100]['diversity_ratio']
    if top100_diversity < 0.3:
        print("\n⚠️ 【警告】Top-100の空間的多様性が非常に低いです（< 30%）")
        print("      特定の地点（Geohash）に予測が集中している可能性があります。")
    elif top100_diversity < 0.5:
        print("\n⚠️ 【注意】Top-100の空間的多様性がやや低いです（< 50%）")
    else:
        print("\n✅ Top-100の空間的多様性は適切なレベルです。")
    
    # 結果保存
    with open(OUTPUT_DIR / "spatial_diversity.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n   保存: {OUTPUT_DIR / 'spatial_diversity.json'}")
    
    return results


def main():
    print("=" * 70)
    print(" 🔍 Spatio-Temporal Model 健全性チェック (Phase 1)")
    print("=" * 70)
    
    train_df, test_df = load_data()
    
    # 1. 特徴量重要度
    importance_df, leakage_suspects = analyze_feature_importance(train_df, test_df)
    
    # 2. 空間的多様性
    diversity_results = analyze_spatial_diversity(test_df)
    
    # サマリー
    print("\n" + "=" * 70)
    print(" 📋 健全性チェック サマリー")
    print("=" * 70)
    
    if leakage_suspects:
        print(f"   ⚠️ リーケージ疑惑: {leakage_suspects}")
    else:
        print("   ✅ リーケージ疑惑: なし")
    
    if diversity_results:
        dr = diversity_results[100]['diversity_ratio']
        print(f"   📍 Top-100 多様性: {dr:.1%} (ユニークGeohash比率)")
        if dr >= 0.5:
            print("   ✅ 空間的多様性: 良好")
        else:
            print("   ⚠️ 空間的多様性: 要確認")
    
    print("\n🎉 健全性チェック完了！")
    print(f"   結果は {OUTPUT_DIR} に保存されました。")


if __name__ == "__main__":
    main()
