import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
import cleanlab
from cleanlab.filter import find_label_issues
import os
import gc
import joblib

# ============================================================================
# 設定
# ============================================================================
FEATURES_PATH = "data/processed/honhyo_clean_with_features.csv"
RAW_DATA_PATH = "honhyo_all/csv/honhyo_all_with_datetime.csv"
TARGET_COL = "死者数"
OUTPUT_DIR = "results/data_quality/cleanlab"
RANDOM_STATE = 42
N_FOLDS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# データ読み込み (train_stage2_multiclass.py から拝借)
# ============================================================================
def load_and_preprocess():
    print("📂 データ読み込み中...")
    
    # 特徴量データ
    df_features = pd.read_csv(FEATURES_PATH)
    
    # ラベル生成 (0:負傷, 1:死亡)
    # ユーザー指摘により無傷(Class 0)は存在aしない前提
    y_binary = (df_features[TARGET_COL] > 0).astype(int)
    
    # 特徴量
    X = df_features.drop(columns=[TARGET_COL])
    if '発生日時' in X.columns:
        X = X.drop(columns=['発生日時'])
        
    print(f"   データ件数: {len(X):,}")
    print(f"   クラス分布 (0:負傷, 1:死亡): {np.bincount(y_binary)}")
    
    # カテゴリ変数の処理
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category')
            
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=RANDOM_STATE, stratify=y_binary
    )
    
    return X_train, y_train, X_test, y_test

# ============================================================================
# Cleanlab 実行
# ============================================================================
def run_cleanlab():
    X_train, y_train, _, _ = load_and_preprocess()
    print(f"DEBUG: X_train type: {type(X_train)}")
    print(f"DEBUG: y_train type: {type(y_train)}")
    
    print("\n🚀 Cross-Validation で予測確率を算出中 (Binary LightGBM)...")
    
    # LightGBM パラメータ (Binary)
    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'scale_pos_weight': float(np.sum(y_train==0) / np.sum(y_train==1)) # Balanced Weight
    }
    
    # CVで確率算出
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    final_probs = np.zeros((len(y_train), 2))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"   Fold {fold+1}/{N_FOLDS}...")
        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        y_val = y_train.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**lgb_params, random_state=RANDOM_STATE+fold)
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # 予測確率 (N, 2)
        probs = model.predict_proba(X_val)
        final_probs[val_idx] = probs
        
    print("\n🔍 Confident Learning (Cleanlab) でラベルノイズ探索中...")
    
    # Cleanlab実行
    # find_label_issues with return_indices_ranked_by returns a numpy array of indices, NOT a DataFrame!
    issue_indices = find_label_issues(
        labels=y_train.values,
        pred_probs=final_probs,
        return_indices_ranked_by='self_confidence',
        n_jobs=1
    )
    
    print(f"\n✅ 発見されたラベル品質問題: {len(issue_indices):,} 件")
    
    # DataFrameを構築 (issue_indices は numpy array of integer indices into y_train/X_train)
    # 元のDataFrameのindex (honhyo_all.csvの行番号に相当) を取得
    original_indices = X_train.index[issue_indices]
    given_labels = y_train.values[issue_indices]
    predicted_labels = np.argmax(final_probs[issue_indices], axis=1)
    
    # ラベル品質スコアを計算
    from cleanlab.rank import get_label_quality_scores
    quality_scores = get_label_quality_scores(y_train.values, final_probs)
    issue_quality_scores = quality_scores[issue_indices]
    
    issues_df = pd.DataFrame({
        'issue_index': issue_indices,  # 0-indexed position in X_train
        'original_index': original_indices,  # Original DataFrame index
        'given_label': given_labels,
        'predicted_label': predicted_labels,
        'label_quality': issue_quality_scores
    })
    
    # 詳細分析: "Label=0 (Injury) but Predicted=1 (Fatal)" (High Confidence)
    # これが「死亡事故に見える負傷事故」
    
    fatal_lookalikes = issues_df[
        (issues_df['given_label'] == 0) & 
        (issues_df['predicted_label'] == 1)
    ]
    
    print(f"   ⚠️ 「負傷(0)」ラベルだが「死亡(1)」と高確信度で予測されたデータ: {len(fatal_lookalikes):,} 件")
    print(f"       (これらがモデルの境界を歪めている可能性が高い)")
    
    # 結果保存
    save_path = os.path.join(OUTPUT_DIR, "label_issues.csv")
    issues_df.to_csv(save_path, index=False)
    
    # ノイズを除去するためのリストも保存
    noise_indices = fatal_lookalikes['original_index'].values
    np.savetxt(os.path.join(OUTPUT_DIR, "noise_indices_fatal_lookalike.txt"), noise_indices, fmt='%d')

    print(f"\n   💾 保存完了: {save_path}")
    print(f"   💾 ノイズインデックス(Fatal Lookalike): {os.path.join(OUTPUT_DIR, 'noise_indices_fatal_lookalike.txt')}")
    
    # 具体的な例を表示
    print("\n   [Examples (Top 5 Fatal Look-alikes)]")
    print(fatal_lookalikes[['original_index', 'given_label', 'predicted_label', 'label_quality']].head())

if __name__ == "__main__":
    run_cleanlab()
