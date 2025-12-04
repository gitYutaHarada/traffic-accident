import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os

# 日本語フォントの設定 (Windows向け)
mpl.rcParams['font.family'] = 'MS Gothic'

def main():
    """
    SMOTEとLightGBMを使用し、閾値調整を行ってRecall改善を目指すスクリプト
    """
    
    print("=" * 80)
    print("高度なモデル改善: SMOTE × LightGBM × 閾値調整")
    print("=" * 80)
    
    # データ読み込み
    file_path = 'data/raw/honhyo_all_shishasuu_binary.csv'
    print(f"\n📂 データ読み込み中: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ データ読み込み完了: {len(df):,} 件")
    except Exception as e:
        print(f"❌ エラー: {e}")
        return
    
    # 目的変数
    target_col = '死者数'
    
    # 除外する列（事後情報・データリーク原因を徹底排除）
    drop_cols = [
        '資料区分', '本票番号',
        '人身損傷程度（当事者A）', '人身損傷程度（当事者B）',
        '車両の損壊程度（当事者A）', '車両の損壊程度（当事者B）',
        '負傷者数',
        '車両の衝突部位（当事者A）', '車両の衝突部位（当事者B）',
        'エアバッグの装備（当事者A）', 'エアバッグの装備（当事者B）',
        'サイドエアバッグの装備（当事者A）', 'サイドエアバッグの装備（当事者B）',
        '事故内容'  # データリーク原因
    ]
    
    print("\n🔧 データ前処理中（事後情報の除外）...")
    df_clean = df.drop(columns=drop_cols, errors='ignore')
    
    # 特徴量と目的変数
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # 欠損値処理
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
    
    # エンコーディング
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))
        
    print(f"✓ 前処理完了 - 特徴量数: {X.shape[1]}")
    
    # LightGBMのパラメータ
    lgbm_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42,
        'n_jobs': -1
    }
    
    # 交差検証 (5-fold)
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    print(f"\n🔄 {k_folds}-fold 交差検証を開始 (SMOTE適用)...")
    
    fold_metrics = []
    threshold_metrics = [] # 閾値ごとの性能を記録
    
    # 全体の予測結果を格納する配列
    y_true_all = []
    y_prob_all = []
    
    for i, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {i+1}/{k_folds} ---")
        
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # パイプライン構築: SMOTE -> LightGBM
        # Pipelineを使うことで、検証データにはSMOTEを適用せず、訓練データのみに適用できる（リーク防止）
        model = Pipeline([
            ('smote', SMOTE(random_state=42)),
            ('lgbm', lgb.LGBMClassifier(**lgbm_params))
        ])
        
        # 学習
        model.fit(X_train, y_train)
        
        # 予測（確率）
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # 全体の結果に蓄積
        y_true_all.extend(y_val)
        y_prob_all.extend(y_prob)
        
        # デフォルト閾値(0.5)での評価
        y_pred_default = (y_prob >= 0.5).astype(int)
        
        acc = accuracy_score(y_val, y_pred_default)
        prec = precision_score(y_val, y_pred_default, average='binary', zero_division=0)
        rec = recall_score(y_val, y_pred_default, average='binary')
        f1 = f1_score(y_val, y_pred_default, average='binary')
        
        print(f"  [Threshold 0.5] Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        fold_metrics.append({
            'Fold': i+1,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })

    # 全データでのPR曲線と最適閾値の探索
    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all)
    
    precisions, recalls, thresholds = precision_recall_curve(y_true_all, y_prob_all)
    
    # F1スコアが最大になる閾値を探す
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print("\n" + "=" * 80)
    print("🎯 最適閾値の探索結果")
    print("=" * 80)
    print(f"Best Threshold (Max F1): {best_threshold:.4f}")
    print(f"Max F1 Score: {best_f1:.4f}")
    print(f"Precision at Best: {precisions[best_idx]:.4f}")
    print(f"Recall at Best: {recalls[best_idx]:.4f}")
    
    # Recall重視の閾値設定（例: Recall >= 0.5 を満たす中で最大のPrecision）
    target_recall = 0.5
    valid_indices = np.where(recalls >= target_recall)[0]
    if len(valid_indices) > 0:
        # valid_indicesの中でPrecisionが最大のインデックスを探す
        # recallsは降順ではない可能性があるため注意が必要だが、通常PR曲線ではトレードオフ
        # ここでは単純にvalidな中でPrecision最大を選ぶ
        best_prec_idx = valid_indices[np.argmax(precisions[valid_indices])]
        recall_threshold = thresholds[best_prec_idx] if best_prec_idx < len(thresholds) else thresholds[-1]
        
        print(f"\n[Recall重視設定 (Target >= {target_recall})]")
        print(f"Threshold: {recall_threshold:.4f}")
        print(f"Precision: {precisions[best_prec_idx]:.4f}")
        print(f"Recall: {recalls[best_prec_idx]:.4f}")
    
    # PR曲線のプロット
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, marker='.', label='LightGBM + SMOTE')
    plt.xlabel('Recall (再現率)')
    plt.ylabel('Precision (適合率)')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    pr_path = 'results/visualizations/pr_curve_advanced.png'
    plt.savefig(pr_path)
    print(f"\n✓ PR曲線を保存: {pr_path}")
    
    # 最適閾値での混同行列
    y_pred_best = (y_prob_all >= best_threshold).astype(int)
    cm = confusion_matrix(y_true_all, y_pred_best)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['非死亡', '死亡'], yticklabels=['非死亡', '死亡'])
    plt.title(f'Confusion Matrix (Threshold={best_threshold:.4f})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    cm_path = 'results/visualizations/confusion_matrix_advanced.png'
    plt.savefig(cm_path)
    print(f"✓ 混同行列を保存: {cm_path}")
    
    # 評価メトリクスの保存
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv('results/analysis/advanced_model_metrics.csv', index=False)
    print("✓ 評価メトリクスを保存: results/analysis/advanced_model_metrics.csv")
    
    print("\n✅ 実験完了")

if __name__ == "__main__":
    main()
