"""
Stage 2用学習データ生成スクリプト
================================
train_two_stage_final.py のStage 1部分を実行し、
OOF予測値を含むStage 2用データを保存する。

Optuna最適化の事前準備として使用。
"""

import pandas as pd
import numpy as np
import os
import gc
import pickle
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


def prepare_stage2_data(
    data_path: str = "data/processed/honhyo_clean_with_features.csv",
    target_col: str = "死者数",
    n_folds: int = 5,
    random_state: int = 42,
    stage1_recall_target: float = 0.99,
    undersample_ratio: float = 2.0,
    n_seeds: int = 3,
    top_k_interactions: int = 5,
    output_dir: str = "results/two_stage_model/optuna_data"
):
    """Stage 2用学習データを生成・保存"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Stage 2用データ生成（OOF予測値付き）")
    print("=" * 60)
    
    # データ読み込み
    print("\n📂 データ読み込み中...")
    df = pd.read_csv(data_path)
    y = df[target_col].values
    X = df.drop(columns=[target_col])
    
    if '発生日時' in X.columns:
        X = X.drop(columns=['発生日時'])
    
    known_categoricals = [
        '都道府県コード', '市区町村コード', '警察署等コード',
        '昼夜', '天候', '地形', '路面状態', '道路形状', '信号機',
        '衝突地点', 'ゾーン規制', '中央分離帯施設等', '歩車道区分',
        '事故類型', '曜日(発生年月日)', '祝日(発生年月日)',
        'road_type', 'area_id', '地点コード'
    ]
    
    categorical_cols = []
    for col in X.columns:
        if col in known_categoricals or X[col].dtype == 'object':
            categorical_cols.append(col)
            X[col] = X[col].astype('category')
        else:
            X[col] = X[col].astype(np.float32)
    
    feature_names = list(X.columns)
    print(f"   正例: {y.sum():,} / {len(y):,}")
    
    # Stage 1: OOF学習
    print("\n🌿 Stage 1: OOF予測値生成（リーク防止）")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_proba = np.zeros(len(y))
    feature_importances = np.zeros(len(feature_names))
    
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
        'n_jobs': -1
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"   Fold {fold+1}/{n_folds}...")
        X_train_full = X.iloc[train_idx]
        y_train_full = y[train_idx]
        X_val = X.iloc[val_idx]
        y_val = y[val_idx]
        
        fold_proba = np.zeros(len(val_idx))
        
        for seed_offset in range(n_seeds):
            seed = random_state + fold * 100 + seed_offset
            
            # アンダーサンプリング
            pos_idx = np.where(y_train_full == 1)[0]
            neg_idx = np.where(y_train_full == 0)[0]
            n_neg_sample = int(len(pos_idx) * undersample_ratio)
            np.random.seed(seed)
            sampled_neg_idx = np.random.choice(neg_idx, size=min(n_neg_sample, len(neg_idx)), replace=False)
            sampled_idx = np.concatenate([pos_idx, sampled_neg_idx])
            np.random.shuffle(sampled_idx)
            X_train_under = X_train_full.iloc[sampled_idx]
            y_train_under = y_train_full[sampled_idx]
            
            model = lgb.LGBMClassifier(**lgb_params, random_state=seed)
            model.fit(
                X_train_under, y_train_under,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            fold_proba += model.predict_proba(X_val)[:, 1] / n_seeds
            feature_importances += model.feature_importances_ / (n_folds * n_seeds)
            
            del model
            gc.collect()
        
        oof_proba[val_idx] = fold_proba
    
    # Feature Importance
    feature_importance_df = pd.DataFrame({
        'feature': feature_names, 'importance': feature_importances
    }).sort_values('importance', ascending=False)
    top_features = feature_importance_df.head(top_k_interactions)['feature'].tolist()
    
    print(f"   OOF AUC: {roc_auc_score(y, oof_proba):.4f}")
    
    # 閾値探索
    for thresh in np.arange(0.50, 0.001, -0.005):
        y_pred = (oof_proba >= thresh).astype(int)
        recall = recall_score(y, y_pred)
        if recall >= stage1_recall_target:
            threshold = thresh
            break
    else:
        threshold = 0.001
    
    stage2_mask = oof_proba >= threshold
    n_candidates = stage2_mask.sum()
    filter_rate = 1 - (n_candidates / len(y))
    print(f"   閾値: {threshold:.4f}, フィルタ率: {filter_rate*100:.2f}%")
    print(f"   Stage 2 候補数: {n_candidates:,}")
    
    # Stage 2用特徴量生成
    print("\n🔧 Stage 2用特徴量生成...")
    X_s2 = X[stage2_mask].copy()
    y_s2 = y[stage2_mask]
    prob_s2 = oof_proba[stage2_mask]
    
    # prob_stage1 追加
    X_s2['prob_stage1'] = prob_s2
    
    # Categorical Interaction Features
    top_cat_features = [f for f in top_features if f in categorical_cols]
    for i, f1 in enumerate(top_cat_features[:top_k_interactions]):
        for f2 in top_cat_features[i+1:top_k_interactions]:
            name = f"{f1}_{f2}"
            X_s2[name] = (X[stage2_mask][f1].astype(str) + "_" + X[stage2_mask][f2].astype(str)).astype('category')
    
    print(f"   特徴量数: {len(X_s2.columns)}")
    
    # 保存
    print("\n💾 データ保存...")
    save_data = {
        'X_s2': X_s2,
        'y_s2': y_s2,
        'prob_s2': prob_s2,
        'threshold': threshold,
        'top_features': top_features,
        'categorical_cols': categorical_cols,
        'feature_importance': feature_importance_df
    }
    
    save_path = os.path.join(output_dir, "stage2_train_data.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"   保存完了: {save_path}")
    print("=" * 60)
    
    return save_data


if __name__ == "__main__":
    prepare_stage2_data()
