"""
GBDT vs TabNet 公平比較スクリプト
=================================
TabNetと同じ特徴量セット (honhyo_clean_with_features.csv) を使用して
LightGBMを学習し、公平な比較を行う。

比較条件を統一:
- 同じデータソース: honhyo_clean_with_features.csv
- 同じTrain/Test分割: 80/20 random split (stratified)
- 同じ5-Fold CV
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, 
    precision_score, recall_score, f1_score, brier_score_loss
)
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# --- パス設定 ---
DATA_PATH = Path("data/processed/honhyo_clean_with_features.csv")
RESULTS_DIR = Path("results/spatio_temporal")
OUTPUT_DIR = RESULTS_DIR / "gbdt_tabnet_comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'

# --- リーク防止 ---
FORBIDDEN_COLUMNS = [
    '事故内容',
    '人身損傷程度（当事者A）', '人身損傷程度（当事者B）',
    '負傷者数',
    '車両の損壊程度（当事者A）', '車両の損壊程度（当事者B）',
    '車両の衝突部位（当事者A）', '車両の衝突部位（当事者B）',
    'エアバッグの装備（当事者A）', 'エアバッグの装備（当事者B）',
    'サイドエアバッグの装備（当事者A）', 'サイドエアバッグの装備（当事者B）',
]


def load_data():
    """TabNetと同じデータを読み込む"""
    print("📂 データ読み込み中 (honhyo_clean_with_features.csv)...")
    df = pd.read_csv(DATA_PATH)
    print(f"   データ: {len(df):,} 行, {len(df.columns)} 列")
    
    # ターゲット列
    target_col = '死者数'
    y = (df[target_col] > 0).astype(int)
    
    # 特徴量
    X = df.drop(columns=[target_col])
    if '発生日時' in X.columns:
        X = X.drop(columns=['発生日時'])
    
    # リークチェック
    leaked = [col for col in FORBIDDEN_COLUMNS if col in X.columns]
    if leaked:
        print(f"   ⚠️ リーク警告: {leaked}")
        X = X.drop(columns=leaked)
    
    print(f"   特徴量: {len(X.columns)} 列")
    print(f"   ターゲット分布: 0={sum(y==0):,}, 1={sum(y==1):,} ({sum(y==1)/len(y)*100:.2f}%)")
    
    return X, y


def prepare_features(X):
    """TabNetと同様の前処理（LightGBM用）"""
    print("\n🔧 特徴量前処理中...")
    
    # カテゴリ列と数値列を識別
    known_categoricals = [
        '都道府県コード', '市区町村コード', '警察署等コード',
        '昼夜', '天候', '地形', '路面状態', '道路形状', '信号機',
        '衝突地点', 'ゾーン規制', '中央分離帯施設等', '歩車道区分',
        '事故類型', '曜日(発生年月日)', '祝日(発生年月日)',
        'road_type', 'area_id', '地点コード', '道路線形',
        '一時停止規制　標識（当事者A）', '一時停止規制　標識（当事者B）',
        '一時停止規制　表示（当事者A）', '一時停止規制　表示（当事者B）'
    ]
    
    categorical_cols = []
    numeric_cols = []
    
    for col in X.columns:
        if col in known_categoricals or X[col].dtype == 'object':
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)
    
    print(f"   カテゴリ列: {len(categorical_cols)}")
    print(f"   数値列: {len(numeric_cols)}")
    
    # LightGBM用: category型に変換
    X_lgb = X.copy()
    for col in categorical_cols:
        X_lgb[col] = X_lgb[col].astype('category')
    for col in numeric_cols:
        X_lgb[col] = pd.to_numeric(X_lgb[col], errors='coerce').astype(np.float32)
    
    # 欠損値補完 (数値列のみ)
    for col in numeric_cols:
        if X_lgb[col].isna().any():
            X_lgb[col] = X_lgb[col].fillna(X_lgb[col].median())
    
    return X_lgb, categorical_cols, numeric_cols


def train_lightgbm_cv(X_train, y_train, n_folds=5, random_state=42):
    """5-Fold CVでLightGBMを学習"""
    print(f"\n🌲 LightGBM {n_folds}-Fold CV 学習中...")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    oof_proba = np.zeros(len(y_train))
    feature_importances = np.zeros(X_train.shape[1])
    models = []
    
    lgb_params = {
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
        'random_state': random_state,
        'is_unbalance': True,
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"   Fold {fold+1}/{n_folds}...")
        
        X_tr = X_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_tr = y_train.iloc[train_idx]
        y_val = y_train.iloc[val_idx]
        
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            lgb_params, dtrain,
            num_boost_round=1000,
            valid_sets=[dtrain, dval],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(0)
            ]
        )
        
        oof_proba[val_idx] = model.predict(X_val)
        feature_importances += model.feature_importance(importance_type='gain') / n_folds
        models.append(model)
        
        fold_auc = roc_auc_score(y_val, oof_proba[val_idx])
        print(f"      Fold {fold+1} AUC: {fold_auc:.4f}")
    
    # 特徴量重要度
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importances
    }).sort_values('importance', ascending=False)
    
    return models, oof_proba, importance_df


def evaluate_metrics(y_true, y_pred_proba):
    """詳細な評価指標を計算"""
    # 基本指標
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)
    
    # Top-k Precision
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    top_k_results = {}
    for k in [100, 500, 1000]:
        if k <= len(y_true):
            top_k_idx = sorted_indices[:k]
            top_k_precision = y_true.iloc[top_k_idx].sum() / k
            top_k_results[f'precision_at_{k}'] = float(top_k_precision)
    
    # 特定Recallでの閾値とPrecision
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_pred_proba)
    recall_targets = {}
    for target_recall in [0.99, 0.95, 0.90]:
        idx = np.searchsorted(recall_curve[::-1], target_recall)
        if idx < len(thresholds):
            thresh = thresholds[::-1][idx] if idx < len(thresholds) else 0.0
            prec = precision_curve[::-1][idx] if idx < len(precision_curve) else 0.0
            recall_targets[f'threshold_at_recall_{int(target_recall*100)}'] = float(thresh)
            recall_targets[f'precision_at_recall_{int(target_recall*100)}'] = float(prec)
    
    # Best F1
    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-15)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_thresh = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
    best_prec = precision_curve[best_f1_idx]
    best_rec = recall_curve[best_f1_idx]
    
    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier_score': brier,
        'best_f1': best_f1,
        'best_f1_threshold': best_thresh,
        'best_f1_precision': best_prec,
        'best_f1_recall': best_rec,
        **top_k_results,
        **recall_targets,
    }
    
    return metrics


def compare_with_tabnet(lgb_metrics):
    """TabNet結果と比較"""
    print("\n" + "=" * 70)
    print(" 🔄 LightGBM vs TabNet 比較 (同一データ・同一条件)")
    print("=" * 70)
    
    # TabNet結果ファイルを探す
    tabnet_paths = [
        Path("results/two_stage_model/tabnet_pipeline/experiment_report.md"),
        Path("results/oof/oof_stage2_tabnet.csv"),
    ]
    
    # TabNet動的閾値評価結果 (comparison_mlp_tabnet.md から)
    # Recall 95%時のPrecision: 2.58%
    tabnet_results = {
        'roc_auc': 0.8393,  # Stage 2 (フィルタリング後), from comparison report
        'precision_at_recall_95': 0.0258,  # from comparison report
    }
    
    comparisons = [
        ('ROC-AUC', 'roc_auc', tabnet_results.get('roc_auc', 0)),
        ('Recall 95% Precision', 'precision_at_recall_95', tabnet_results.get('precision_at_recall_95', 0)),
        ('Best F1', 'best_f1', 0),  # TabNetのBest F1は別途取得必要
    ]
    
    print(f"\n   {'指標':<30} {'TabNet':<15} {'LightGBM':<15} {'差分':<10}")
    print("   " + "-" * 70)
    
    comparison_results = {}
    for name, key, tabnet_val in comparisons:
        lgb_val = lgb_metrics.get(key, 0)
        diff = lgb_val - tabnet_val
        diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        print(f"   {name:<30} {tabnet_val:<15.4f} {lgb_val:<15.4f} {diff_str:<10}")
        comparison_results[key] = {'tabnet': tabnet_val, 'lgb': lgb_val, 'diff': diff}
    
    return comparison_results


def save_results(models, oof_proba, y_train, metrics, importance_df, comparison_results):
    """結果を保存"""
    print("\n💾 結果を保存中...")
    
    # モデル保存
    for i, model in enumerate(models):
        model.save_model(str(OUTPUT_DIR / f"lightgbm_fold{i+1}.txt"))
    
    # メトリクス保存
    results = {
        'model_type': 'lightgbm',
        'data_source': 'honhyo_clean_with_features.csv',
        'comparison_note': 'TabNetと同じ特徴量セットで学習',
        'oof_metrics': metrics,
        'comparison_with_tabnet': comparison_results,
    }
    with open(OUTPUT_DIR / "results_lightgbm_tabnet_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 特徴量重要度保存
    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False, encoding='utf-8-sig')
    
    # OOF予測保存
    oof_df = pd.DataFrame({
        'true_label': y_train.values,
        'prob': oof_proba
    })
    oof_df.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    
    # プロット
    plt.figure(figsize=(12, 8))
    top20 = importance_df.head(20)
    plt.barh(top20['feature'][::-1], top20['importance'][::-1])
    plt.xlabel('Importance (Gain)')
    plt.title('Top 20 Feature Importance (LightGBM - TabNet Comparison)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150)
    plt.close()
    
    print(f"   結果保存先: {OUTPUT_DIR}")


def main():
    print("=" * 70)
    print(" 🌲 LightGBM vs TabNet 公平比較")
    print(" (同じデータ: honhyo_clean_with_features.csv)")
    print("=" * 70)
    
    # 1. データ読み込み
    X, y = load_data()
    
    # 2. 前処理
    X_lgb, categorical_cols, numeric_cols = prepare_features(X)
    
    # 3. Train/Test分割 (TabNetと同じ: 80/20)
    print("\n✂️ Train/Test分割 (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_lgb, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Train: {len(y_train):,} (Fatal: {y_train.sum():,})")
    print(f"   Test:  {len(y_test):,} (Fatal: {y_test.sum():,})")
    
    # 4. 5-Fold CVで学習
    models, oof_proba, importance_df = train_lightgbm_cv(X_train, y_train)
    
    # 5. OOF評価
    print("\n📊 OOF評価 (Cross Validation)...")
    oof_metrics = evaluate_metrics(y_train, oof_proba)
    
    print(f"\n   ROC-AUC: {oof_metrics['roc_auc']:.4f}")
    print(f"   PR-AUC:  {oof_metrics['pr_auc']:.4f}")
    print(f"   Best F1: {oof_metrics['best_f1']:.4f} (閾値: {oof_metrics['best_f1_threshold']:.4f})")
    print(f"   Recall 95% Precision: {oof_metrics.get('precision_at_recall_95', 0):.4f}")
    
    # 6. テストセット評価
    print("\n📊 テストセット評価...")
    test_proba = np.zeros(len(y_test))
    for model in models:
        test_proba += model.predict(X_test) / len(models)
    
    test_metrics = evaluate_metrics(y_test, test_proba)
    print(f"   Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"   Test PR-AUC:  {test_metrics['pr_auc']:.4f}")
    print(f"   Test Best F1: {test_metrics['best_f1']:.4f}")
    print(f"   Test Recall 95% Precision: {test_metrics.get('precision_at_recall_95', 0):.4f}")
    
    # 7. TabNetと比較
    comparison_results = compare_with_tabnet(oof_metrics)
    
    # 8. 結果保存
    save_results(models, oof_proba, y_train, oof_metrics, importance_df, comparison_results)
    
    print("\n🎉 LightGBM vs TabNet 比較完了！")
    
    # サマリー出力
    print("\n" + "=" * 70)
    print(" 📋 サマリー")
    print("=" * 70)
    print(f"   データ: honhyo_clean_with_features.csv ({len(X):,} 件)")
    print(f"   特徴量: {len(X.columns)} 列 (カテゴリ: {len(categorical_cols)}, 数値: {len(numeric_cols)})")
    print(f"   LightGBM OOF ROC-AUC: {oof_metrics['roc_auc']:.4f}")
    print(f"   LightGBM OOF PR-AUC:  {oof_metrics['pr_auc']:.4f}")
    print(f"   LightGBM Test ROC-AUC: {test_metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
