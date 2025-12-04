import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os

# 日本語フォントの設定 (Windows向け)
mpl.rcParams['font.family'] = 'MS Gothic'

def main():
    """
    事後情報を除外し、交差検証を用いてランダムフォレストモデルを評価する
    """
    
    print("=" * 80)
    print("過学習検証: 事後情報除外モデルの交差検証")
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
    
    # 除外する列（ID、管理番号、および事後情報）
    # ここで「事故が起きた後でないと分からない情報」を徹底的に排除する
    drop_cols = [
        '資料区分', '本票番号',           # 管理情報
        '人身損傷程度（当事者A）',        # 事後情報
        '人身損傷程度（当事者B）',        # 事後情報
        '車両の損壊程度（当事者A）',      # 事後情報
        '車両の損壊程度（当事者B）',      # 事後情報
        '負傷者数',                       # 事後情報
        '車両の衝突部位（当事者A）',      # 事後情報（事故態様によるが、結果に近い）
        '車両の衝突部位（当事者B）',      # 事後情報
        'エアバッグの装備（当事者A）',    # 作動状況が含まれる可能性があるため除外
        'エアバッグの装備（当事者B）',
        'サイドエアバッグの装備（当事者A）',
        'サイドエアバッグの装備（当事者B）',
        '事故内容'                        # データリーク原因（死亡/負傷の区分そのもの）
    ]
    
    print("\n🚫 除外する特徴量（事後情報など）:")
    for col in drop_cols:
        print(f"  - {col}")
    
    # データ前処理
    print("\n🔧 データ前処理中...")
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
    
    # 交差検証 (5-fold)
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    print(f"\n🔄 {k_folds}-fold 交差検証を開始...")
    
    fold_metrics = []
    
    for i, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {i+1}/{k_folds} ---")
        
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # 訓練データのみアップサンプリング
        # 少数派クラスを特定
        X_train_minority = X_train[y_train == 1]
        y_train_minority = y_train[y_train == 1]
        
        X_train_majority = X_train[y_train == 0]
        y_train_majority = y_train[y_train == 0]
        
        # アップサンプリング実行
        X_minority_upsampled, y_minority_upsampled = resample(
            X_train_minority, y_train_minority,
            replace=True,
            n_samples=len(X_train_majority),
            random_state=42
        )
        
        # 結合
        X_train_res = pd.concat([X_train_majority, X_minority_upsampled])
        y_train_res = pd.concat([y_train_majority, y_minority_upsampled])
        
        # モデル学習
        rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1) # 時間短縮のため50木
        rf.fit(X_train_res, y_train_res)
        
        # 評価
        y_pred = rf.predict(X_val)
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='binary', zero_division=0)
        rec = recall_score(y_val, y_pred, average='binary')
        f1 = f1_score(y_val, y_pred, average='binary')
        
        print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        fold_metrics.append({
            'Fold': i+1,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })
    
    # 平均スコアの計算
    metrics_df = pd.DataFrame(fold_metrics)
    mean_metrics = metrics_df.mean()
    
    print("\n" + "=" * 80)
    print("📊 交差検証結果 (平均)")
    print("=" * 80)
    print(f"Accuracy: {mean_metrics['Accuracy']:.4f}")
    print(f"Precision: {mean_metrics['Precision']:.4f}")
    print(f"Recall:    {mean_metrics['Recall']:.4f}")
    print(f"F1 Score:  {mean_metrics['F1 Score']:.4f}")
    
    # 結果の保存
    output_csv = 'results/analysis/refined_model_cv_metrics.csv'
    metrics_df.to_csv(output_csv, index=False)
    print(f"\n✓ 詳細結果を保存: {output_csv}")
    
    # 全データでの再学習と特徴量重要度の確認（参考用）
    print("\n🔍 全データで再学習して特徴量重要度を確認中...")
    
    # 全データアップサンプリング（可視化用）
    X_minority = X[y == 1]
    y_minority = y[y == 1]
    X_majority = X[y == 0]
    y_majority = y[y == 0]
    
    X_min_up, y_min_up = resample(X_minority, y_minority, replace=True, n_samples=len(X_majority), random_state=42)
    X_full = pd.concat([X_majority, X_min_up])
    y_full = pd.concat([y_majority, y_min_up])
    
    rf_full = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_full.fit(X_full, y_full)
    
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_full.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 重要度の可視化
    plt.figure(figsize=(12, 10))
    sns.barplot(
        x='importance', 
        y='feature', 
        data=feature_importances.head(20),
        palette='viridis'
    )
    plt.title('特徴量重要度 Top 20 (事後情報除外モデル)', fontsize=16, pad=20)
    plt.xlabel('重要度', fontsize=12)
    plt.ylabel('特徴量', fontsize=12)
    plt.tight_layout()
    
    fi_path = 'results/visualizations/feature_importance_refined.png'
    plt.savefig(fi_path, dpi=300, bbox_inches='tight')
    print(f"✓ 特徴量重要度グラフを保存: {fi_path}")

    print("\n特徴量重要度 (Top 20):")
    print("-" * 80)
    for idx, row in feature_importances.head(20).iterrows():
        print(f"{row['feature']:45s}: {row['importance']:.6f}")
    
    print("\n✅ 検証完了")

if __name__ == "__main__":
    main()
