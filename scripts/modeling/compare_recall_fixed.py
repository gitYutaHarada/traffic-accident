"""
Recall固定でのPrecision比較分析
================================
同じRecall目標（99%, 95%）に対して、
手動設定モデル vs Optuna最適化モデル の Precision を比較する。
"""

import pandas as pd
import numpy as np
import os
import gc
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


def find_threshold_for_recall(y_true, y_prob, target_recall):
    """指定Recallを達成する閾値を探索"""
    for thresh in np.arange(0.99, 0.001, -0.005):
        y_pred = (y_prob >= thresh).astype(int)
        recall = recall_score(y_true, y_pred)
        if recall >= target_recall:
            return thresh
    return 0.001


def run_comparison():
    """手動モデル vs Optuna最適化モデル の比較"""
    
    print("=" * 70)
    print("Recall固定でのPrecision比較分析")
    print("=" * 70)
    
    # Stage 2用データ読み込み
    print("\n📂 データ読み込み...")
    data_path = "results/two_stage_model/optuna_data/stage2_train_data.pkl"
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    X_s2 = data['X_s2']
    y_s2 = data['y_s2']
    
    print(f"   データ数: {len(y_s2):,} (Pos: {y_s2.sum():,})")
    
    # 2つのモデル設定
    models_config = {
        '手動設定': {
            'num_leaves': 63,
            'max_depth': -1,
            'min_child_samples': 10,
            'reg_alpha': 2.0,
            'reg_lambda': 2.0,
            'colsample_bytree': 0.7,
            'is_unbalance': True,
            'n_estimators': 500,
            'learning_rate': 0.05,
        },
        'Optuna最適化': {
            'num_leaves': 118,
            'max_depth': 8,
            'min_child_samples': 32,
            'reg_alpha': 7.289881162161227,
            'reg_lambda': 0.7394666125185072,
            'colsample_bytree': 0.7059404335368878,
            'subsample': 0.5385972873574277,
            'learning_rate': 0.048867878885592735,
            'scale_pos_weight': 1.1345607321720075,
            'is_unbalance': False,
            'n_estimators': 500,
        }
    }
    
    recall_targets = [0.99, 0.95]
    results = []
    
    # 5-Fold CVでOOF予測を取得
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, params in models_config.items():
        print(f"\n🌿 {model_name} モデル学習...")
        
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'n_jobs': -1,
            'random_state': 42,
            **params
        }
        
        oof_proba = np.zeros(len(y_s2))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_s2, y_s2)):
            X_train = X_s2.iloc[train_idx]
            y_train = y_s2[train_idx]
            X_val = X_s2.iloc[val_idx]
            y_val = y_s2[val_idx]
            
            model = lgb.LGBMClassifier(**lgb_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            oof_proba[val_idx] = model.predict_proba(X_val)[:, 1]
            
            del model
            gc.collect()
        
        # 各Recall目標でPrecision算出
        for target_recall in recall_targets:
            thresh = find_threshold_for_recall(y_s2, oof_proba, target_recall)
            y_pred = (oof_proba >= thresh).astype(int)
            precision = precision_score(y_s2, y_pred)
            recall = recall_score(y_s2, y_pred)
            
            results.append({
                'model': model_name,
                'target_recall': f"{int(target_recall*100)}%",
                'threshold': thresh,
                'actual_recall': recall,
                'precision': precision
            })
            
            print(f"   Recall {int(target_recall*100)}%: 閾値={thresh:.4f}, Prec={precision:.4f}, Rec={recall:.4f}")
    
    # 結果表示
    print("\n" + "=" * 70)
    print("📊 比較結果")
    print("=" * 70)
    
    df = pd.DataFrame(results)
    
    for target in ['99%', '95%']:
        print(f"\n【Recall {target} 目標】")
        subset = df[df['target_recall'] == target]
        for _, row in subset.iterrows():
            print(f"   {row['model']}: Precision = {row['precision']:.4f} (閾値: {row['threshold']:.4f})")
        
        # 改善率
        manual = subset[subset['model'] == '手動設定']['precision'].values[0]
        optuna = subset[subset['model'] == 'Optuna最適化']['precision'].values[0]
        improvement = (optuna - manual) / manual * 100 if manual > 0 else 0
        print(f"   → 改善率: {improvement:+.2f}%")
    
    # 結果保存
    output_dir = "results/two_stage_model/comparison"
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, "recall_fixed_comparison.csv"), index=False)
    print(f"\n💾 結果保存: {output_dir}/recall_fixed_comparison.csv")
    
    return df


if __name__ == "__main__":
    run_comparison()
