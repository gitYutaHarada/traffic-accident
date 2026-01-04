"""
Stage 1 OOF予測値を保存するスクリプト（Intel最適化版）
=====================================================
既存のtrain_two_stage_or_ensemble.pyのStage 1部分のみを実行し、
OOF予測値とテストセット予測値をCSVとして保存する。

最適化:
- Intel Extension for Scikit-learn (sklearnex) を使用
- LightGBM/CatBoost の n_jobs/thread_count を 8 に制限（P-core最適化）

出力:
    data/processed/stage1_oof_predictions.csv   (Train OOF)
    data/processed/stage1_test_predictions.csv  (Test)

実行方法:
    python scripts/experiment/save_stage1_oof.py
"""

# Intel Extension for Scikit-learn（最初に読み込む）
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("✅ Intel Extension for Scikit-learn が有効化されました")
except ImportError:
    print("⚠️ sklearnex がインストールされていません。'pip install scikit-learn-intelex' を推奨")

import pandas as pd
import numpy as np
import os
import gc
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import recall_score, roc_auc_score, precision_recall_curve
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')

# Intel Core Ultra 9 285K 向け最適化パラメータ
# P-core 8個を中心に使用（E-coreを無理に使わない）
N_JOBS_OPTIMAL = 8


def save_stage1_oof(
    data_path: str = "data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv",
    target_col: str = "fatal",
    n_folds: int = 5,
    random_state: int = 42,
    undersample_ratio: float = 2.0,
    n_seeds: int = 3,
    test_size: float = 0.2,
    output_dir: str = "data/processed",
):
    """Stage 1のOOF予測値とテスト予測値を生成して保存"""
    
    print("=" * 70)
    print("Stage 1 OOF予測値 生成・保存 (Intel最適化版)")
    print(f"CPU最適化: n_jobs={N_JOBS_OPTIMAL} (P-core向け)")
    print("=" * 70)
    
    # データ読み込み
    print("\n📂 データ読み込み中...")
    df = pd.read_csv(data_path)
    
    # 元のインデックスを保持
    df['original_index'] = df.index
    
    y_all = df[target_col].values
    X_all = df.drop(columns=[target_col])
    
    if '発生日時' in X_all.columns:
        X_all = X_all.drop(columns=['発生日時'])
    
    known_categoricals = [
        '都道府県コード', '市区町村コード', '警察署等コード',
        '昼夜', '天候', '地形', '路面状態', '道路形状', '信号機',
        '衝突地点', 'ゾーン規制', '中央分離帯施設等', '歩車道区分',
        '事故類型', '曜日(発生年月日)', '祝日(発生年月日)',
        'road_type', 'area_id', '地点コード'
    ]
    
    categorical_cols = []
    numerical_cols = []
    
    for col in X_all.columns:
        if col == 'original_index':
            continue
        if col in known_categoricals or X_all[col].dtype == 'object':
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)
            X_all[col] = X_all[col].astype(np.float32)
    
    # Train/Test分割
    print(f"\n📊 データ分割 (Train: {1-test_size:.0%} / Test: {test_size:.0%})")
    X, X_test, y, y_test, idx_train, idx_test = train_test_split(
        X_all, y_all, X_all['original_index'].values,
        test_size=test_size, random_state=random_state, stratify=y_all
    )
    X = X.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    
    print(f"   Train: 正例 {y.sum():,} / {len(y):,}")
    print(f"   Test:  正例 {y_test.sum():,} / {len(y_test):,}")
    
    # OOF/テスト予測値を格納
    oof_proba_lgbm = np.zeros(len(y))
    oof_proba_catboost = np.zeros(len(y))
    test_proba_lgbm = np.zeros(len(y_test))
    test_proba_catboost = np.zeros(len(y_test))
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    # LightGBM用データ準備
    X_lgbm = X.drop(columns=['original_index']).copy()
    X_test_lgbm = X_test.drop(columns=['original_index']).copy()
    for col in categorical_cols:
        if col in X_lgbm.columns:
            X_lgbm[col] = X_lgbm[col].astype('category')
            X_test_lgbm[col] = X_test_lgbm[col].astype('category')
    
    # CatBoost用データ準備
    X_cat = X.drop(columns=['original_index']).copy()
    X_test_cat = X_test.drop(columns=['original_index']).copy()
    for col in categorical_cols:
        if col in X_cat.columns:
            X_cat[col] = X_cat[col].astype(str)
            X_test_cat[col] = X_test_cat[col].astype(str)
    cat_feature_indices = [X_cat.columns.get_loc(c) for c in categorical_cols if c in X_cat.columns]
    
    # ========== LightGBM ==========
    print("\n🌲 LightGBM 学習中...")
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'num_leaves': 31,
        'max_depth': 8,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'n_jobs': N_JOBS_OPTIMAL  # P-core最適化
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_lgbm, y)):
        print(f"   Fold {fold+1}/{n_folds}...")
        X_train_full = X_lgbm.iloc[train_idx]
        y_train_full = y[train_idx]
        X_val = X_lgbm.iloc[val_idx]
        y_val = y[val_idx]
        
        fold_proba = np.zeros(len(val_idx))
        fold_test_proba = np.zeros(len(y_test))
        
        for seed_offset in range(n_seeds):
            seed = random_state + fold * 100 + seed_offset
            
            # Under-sampling
            pos_idx = np.where(y_train_full == 1)[0]
            neg_idx = np.where(y_train_full == 0)[0]
            n_neg_sample = int(len(pos_idx) * undersample_ratio)
            np.random.seed(seed)
            sampled_neg_idx = np.random.choice(neg_idx, size=min(n_neg_sample, len(neg_idx)), replace=False)
            sampled_idx = np.concatenate([pos_idx, sampled_neg_idx])
            np.random.shuffle(sampled_idx)
            
            X_train_under = X_train_full.iloc[sampled_idx].copy()
            y_train_under = y_train_full[sampled_idx]
            
            for col in categorical_cols:
                if col in X_train_under.columns:
                    X_train_under[col] = X_train_under[col].astype('category')
            
            model = lgb.LGBMClassifier(**lgb_params, random_state=seed)
            model.fit(X_train_under, y_train_under, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            
            fold_proba += model.predict_proba(X_val)[:, 1] / n_seeds
            fold_test_proba += model.predict_proba(X_test_lgbm)[:, 1] / n_seeds
        
        oof_proba_lgbm[val_idx] = fold_proba
        test_proba_lgbm += fold_test_proba / n_folds
        gc.collect()
    
    lgbm_auc = roc_auc_score(y, oof_proba_lgbm)
    print(f"   LightGBM OOF AUC: {lgbm_auc:.4f}")
    
    # ========== CatBoost ==========
    print("\n🐱 CatBoost 学習中...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cat, y)):
        print(f"   Fold {fold+1}/{n_folds}...")
        X_train_full = X_cat.iloc[train_idx]
        y_train_full = y[train_idx]
        X_val = X_cat.iloc[val_idx]
        y_val = y[val_idx]
        
        fold_proba = np.zeros(len(val_idx))
        fold_test_proba = np.zeros(len(y_test))
        
        for seed_offset in range(n_seeds):
            seed = random_state + fold * 100 + seed_offset
            
            # Under-sampling
            pos_idx = np.where(y_train_full == 1)[0]
            neg_idx = np.where(y_train_full == 0)[0]
            n_neg_sample = int(len(pos_idx) * undersample_ratio)
            np.random.seed(seed)
            sampled_neg_idx = np.random.choice(neg_idx, size=min(n_neg_sample, len(neg_idx)), replace=False)
            sampled_idx = np.concatenate([pos_idx, sampled_neg_idx])
            np.random.shuffle(sampled_idx)
            
            X_train_under = X_train_full.iloc[sampled_idx]
            y_train_under = y_train_full[sampled_idx]
            
            model = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.05,
                depth=8,
                l2_leaf_reg=3,
                loss_function='Logloss',
                eval_metric='AUC',
                random_seed=seed,
                verbose=False,
                early_stopping_rounds=50,
                task_type='CPU',
                thread_count=N_JOBS_OPTIMAL,  # P-core最適化
                cat_features=cat_feature_indices
            )
            model.fit(X_train_under, y_train_under, eval_set=(X_val, y_val), verbose=False)
            
            fold_proba += model.predict_proba(X_val)[:, 1] / n_seeds
            fold_test_proba += model.predict_proba(X_test_cat)[:, 1] / n_seeds
        
        oof_proba_catboost[val_idx] = fold_proba
        test_proba_catboost += fold_test_proba / n_folds
        gc.collect()
    
    catboost_auc = roc_auc_score(y, oof_proba_catboost)
    print(f"   CatBoost OOF AUC: {catboost_auc:.4f}")
    
    # ========== 保存 ==========
    print("\n💾 予測値を保存中...")
    os.makedirs(output_dir, exist_ok=True)
    
    # OOF (Train)
    oof_df = pd.DataFrame({
        'original_index': idx_train,
        'prob_lgbm': oof_proba_lgbm,
        'prob_catboost': oof_proba_catboost,
        'target': y
    })
    oof_path = os.path.join(output_dir, "stage1_oof_predictions.csv")
    oof_df.to_csv(oof_path, index=False)
    print(f"   OOF保存完了: {oof_path}")
    print(f"   データ件数: {len(oof_df):,}, 正例: {oof_df['target'].sum():,}")
    
    # Test
    test_df = pd.DataFrame({
        'original_index': idx_test,
        'prob_lgbm': test_proba_lgbm,
        'prob_catboost': test_proba_catboost,
        'target': y_test
    })
    test_path = os.path.join(output_dir, "stage1_test_predictions.csv")
    test_df.to_csv(test_path, index=False)
    print(f"   Test保存完了: {test_path}")
    print(f"   データ件数: {len(test_df):,}, 正例: {test_df['target'].sum():,}")
    
    # ========== 現状評価（参考用）==========
    print("\n📈 現状評価（参考用）...")
    print("⚠️ 注意: アンダーサンプリングにより確率の絶対値はキャリブレーションが必要です")
    
    target_recall = 0.995
    prob_max = np.maximum(oof_proba_lgbm, oof_proba_catboost)
    precision_arr, recall_arr, thresh_arr = precision_recall_curve(y, prob_max)
    
    valid_idx = np.where(recall_arr >= target_recall)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[-1]
        thresh_max = thresh_arr[best_idx] if best_idx < len(thresh_arr) else 0
        actual_recall = recall_arr[best_idx]
        pass_rate = (prob_max >= thresh_max).mean()
    else:
        thresh_max, actual_recall, pass_rate = 0, 1.0, 1.0
    
    print(f"   Max Probability閾値: {thresh_max:.4f}")
    print(f"   Recall: {actual_recall:.4f}")
    print(f"   Pass Rate: {pass_rate:.2%}")
    
    print("\n" + "=" * 70)
    print("✅ 完了！")
    print("=" * 70)
    
    return oof_df, test_df


if __name__ == "__main__":
    save_stage1_oof()
