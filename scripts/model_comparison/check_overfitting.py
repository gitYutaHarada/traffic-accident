"""
過学習確認用: 訓練データと検証データの両方で性能を測定

既存の1%サンプル結果に対して、訓練データでの性能も測定し、
過学習の有無を確認する。
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

warnings.filterwarnings('ignore')

# 日本語フォントの設定
mpl.rcParams['font.family'] = 'MS Gothic'

def main():
    """
    過学習確認: 訓練データと検証データの性能を比較
    """
    
    print("=" * 80)
    print("過学習確認: ロジスティック回帰 (1%サンプル)")
    print("=" * 80)
    
    # データ読み込み
    file_path = 'data/processed/honhyo_model_ready.csv'
    print(f"\n📂 データ読み込み中: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"✓ データ読み込み完了: {len(df):,} 件")
    
    # 目的変数
    target_col = '死者数'
    
    # 1%サンプリング(層化サンプリング)
    sample_rate = 0.01
    print(f"\n🎲 {sample_rate*100:.1f}% サンプリング中(層化サンプリング)...")
    df_0 = df[df[target_col] == 0].sample(frac=sample_rate, random_state=42)
    df_1 = df[df[target_col] == 1].sample(frac=sample_rate, random_state=42)
    df = pd.concat([df_0, df_1], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"✓ サンプリング完了: {len(df):,} 件")
    
    # 除外する列
    drop_cols = [
        '資料区分', '本票番号',
        '人身損傷程度(当事者A)', '人身損傷程度(当事者B)',
        '車両の損壊程度(当事者A)', '車両の損壊程度(当事者B)',
        '負傷者数',
        '車両の衝突部位(当事者A)', '車両の衝突部位(当事者B)',
        'エアバッグの装備(当事者A)', 'エアバッグの装備(当事者B)',
        'サイドエアバッグの装備(当事者A)', 'サイドエアバッグの装備(当事者B)',
        '事故内容'
    ]
    
    # カラム名の正規化
    df.columns = df.columns.str.replace('(', '(').str.replace(')', ')')
    
    print("\n🔧 データ前処理中...")
    df_clean = df.drop(columns=drop_cols, errors='ignore')
    
    # 特徴量と目的変数
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # カテゴリカル変数と数値変数の分類
    count_encoding_cols = [col for col in X.columns if col.endswith('_count')]
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    explicit_cat_cols = [
        '都道府県コード', '路線コード', '地点コード', '市区町村コード',
        '昼夜', '天候', '地形', '路面状態', '道路形状', '信号機',
        '一時停止規制 標識', '一時停止規制 表示', '車道幅員', '道路線形',
        '衝突地点', 'ゾーン規制', '中央分離帯施設等', '歩車道区分',
        '事故類型', '年齢', '当事者種別', '用途別', '車両形状',
        'オートマチック車', 'サポカー', '速度規制(指定のみ)',
        '曜日', '祝日', '発生月', '発生時', '発生年', 'Area_Cluster_ID'
    ]
    
    explicit_cat_cols = [c for c in explicit_cat_cols if c in X.columns and c not in count_encoding_cols]
    final_cat_cols = list(set(categorical_cols + explicit_cat_cols))
    final_numeric_cols = [c for c in numeric_cols if c not in final_cat_cols]
    
    # カテゴリカル変数を文字列型に変換
    for col in final_cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)
    
    # 高カーディナリティ処理
    high_cardinality_threshold = 50
    for col in final_cat_cols:
        if col in X.columns:
            nunique = X[col].nunique()
            if nunique > high_cardinality_threshold:
                top_categories = X[col].value_counts().head(high_cardinality_threshold).index
                X[col] = X[col].apply(lambda x: x if x in top_categories else 'その他')
    
    # 前処理パイプライン
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=30))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, final_numeric_cols),
            ('cat', categorical_transformer, final_cat_cols)
        ],
        remainder='drop'
    )
    
    # モデル
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='saga',
            max_iter=500,
            class_weight='balanced',
            random_state=42,
            verbose=0,  # 進捗表示を抑制
            n_jobs=-1
        ))
    ])
    
    # 5-fold交差検証
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    print(f"\n🔄 {k_folds}-fold 交差検証を開始(訓練・検証データの両方で評価)...\n")
    
    fold_metrics = []
    
    for i, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"--- Fold {i+1}/{k_folds} ---")
        
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # 学習
        print(f"  学習中...")
        model.fit(X_train, y_train)
        
        # 訓練データでの予測
        y_train_prob = model.predict_proba(X_train)[:, 1]
        y_train_pred = (y_train_prob >= 0.5).astype(int)
        
        # 検証データでの予測
        y_val_prob = model.predict_proba(X_val)[:, 1]
        y_val_pred = (y_val_prob >= 0.5).astype(int)
        
        # 訓練データの評価
        train_acc = accuracy_score(y_train, y_train_pred)
        train_prec = precision_score(y_train, y_train_pred, average='binary', zero_division=0)
        train_rec = recall_score(y_train, y_train_pred, average='binary')
        train_f1 = f1_score(y_train, y_train_pred, average='binary')
        train_auc = roc_auc_score(y_train, y_train_prob)
        
        # 検証データの評価
        val_acc = accuracy_score(y_val, y_val_pred)
        val_prec = precision_score(y_val, y_val_pred, average='binary', zero_division=0)
        val_rec = recall_score(y_val, y_val_pred, average='binary')
        val_f1 = f1_score(y_val, y_val_pred, average='binary')
        val_auc = roc_auc_score(y_val, y_val_prob)
        
        print(f"\n  📚 訓練データ:")
        print(f"     Acc: {train_acc:.4f}, Prec: {train_prec:.4f}, Recall: {train_rec:.4f}, F1: {train_f1:.4f}, AUC: {train_auc:.4f}")
        
        print(f"  📊 検証データ:")
        print(f"     Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, Recall: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        
        # 差分(過学習の指標)
        diff_acc = train_acc - val_acc
        diff_prec = train_prec - val_prec
        diff_rec = train_rec - val_rec
        diff_f1 = train_f1 - val_f1
        diff_auc = train_auc - val_auc
        
        print(f"  📉 差分(訓練 - 検証):")
        print(f"     Acc: {diff_acc:+.4f}, Prec: {diff_prec:+.4f}, Recall: {diff_rec:+.4f}, F1: {diff_f1:+.4f}, AUC: {diff_auc:+.4f}")
        
        fold_metrics.append({
            'Fold': i+1,
            'Train_Accuracy': train_acc,
            'Val_Accuracy': val_acc,
            'Diff_Accuracy': diff_acc,
            'Train_Precision': train_prec,
            'Val_Precision': val_prec,
            'Diff_Precision': diff_prec,
            'Train_Recall': train_rec,
            'Val_Recall': val_rec,
            'Diff_Recall': diff_rec,
            'Train_F1': train_f1,
            'Val_F1': val_f1,
            'Diff_F1': diff_f1,
            'Train_AUC': train_auc,
            'Val_AUC': val_auc,
            'Diff_AUC': diff_auc
        })
        print()
    
    # 結果の集計
    metrics_df = pd.DataFrame(fold_metrics)
    
    print("=" * 80)
    print("📊 過学習分析結果")
    print("=" * 80)
    
    print("\n【平均性能】")
    avg_train_auc = metrics_df['Train_AUC'].mean()
    avg_val_auc = metrics_df['Val_AUC'].mean()
    avg_diff_auc = metrics_df['Diff_AUC'].mean()
    
    avg_train_f1 = metrics_df['Train_F1'].mean()
    avg_val_f1 = metrics_df['Val_F1'].mean()
    avg_diff_f1 = metrics_df['Diff_F1'].mean()
    
    print(f"\n訓練データ:")
    print(f"  AUC:       {avg_train_auc:.4f}")
    print(f"  F1 Score:  {avg_train_f1:.4f}")
    print(f"  Recall:    {metrics_df['Train_Recall'].mean():.4f}")
    print(f"  Precision: {metrics_df['Train_Precision'].mean():.4f}")
    
    print(f"\n検証データ:")
    print(f"  AUC:       {avg_val_auc:.4f}")
    print(f"  F1 Score:  {avg_val_f1:.4f}")
    print(f"  Recall:    {metrics_df['Val_Recall'].mean():.4f}")
    print(f"  Precision: {metrics_df['Val_Precision'].mean():.4f}")
    
    print(f"\n差分(訓練 - 検証):")
    print(f"  AUC:       {avg_diff_auc:+.4f} ({abs(avg_diff_auc)/avg_train_auc*100:.2f}%)")
    print(f"  F1 Score:  {avg_diff_f1:+.4f} ({abs(avg_diff_f1)/avg_train_f1*100:.2f}%)")
    
    # 過学習の判定
    print("\n" + "=" * 80)
    print("🔍 過学習の判定")
    print("=" * 80)
    
    # 判定基準
    auc_threshold = 0.05  # AUCの差が5%以内なら健全
    f1_threshold = 0.10   # F1の差が10%以内なら健全
    
    is_overfitting_auc = abs(avg_diff_auc) > auc_threshold
    is_overfitting_f1 = abs(avg_diff_f1) > f1_threshold
    
    print(f"\nAUCベース:")
    if is_overfitting_auc:
        print(f"  ⚠️  過学習の可能性あり (差分: {abs(avg_diff_auc):.4f} > 閾値: {auc_threshold})")
    else:
        print(f"  ✅ 健全 (差分: {abs(avg_diff_auc):.4f} <= 閾値: {auc_threshold})")
    
    print(f"\nF1スコアベース:")
    if is_overfitting_f1:
        print(f"  ⚠️  過学習の可能性あり (差分: {abs(avg_diff_f1):.4f} > 閾値: {f1_threshold})")
    else:
        print(f"  ✅ 健全 (差分: {abs(avg_diff_f1):.4f} <= 閾値: {f1_threshold})")
    
    # 総合判定
    print(f"\n総合判定:")
    if is_overfitting_auc or is_overfitting_f1:
        print("  ⚠️  過学習の兆候が見られます")
    else:
        print("  ✅ 過学習は見られません(健全なモデル)")
    
    # 可視化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # AUCの比較
    ax1 = axes[0, 0]
    folds = metrics_df['Fold']
    ax1.plot(folds, metrics_df['Train_AUC'], marker='o', label='訓練データ', linewidth=2)
    ax1.plot(folds, metrics_df['Val_AUC'], marker='s', label='検証データ', linewidth=2)
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('AUC')
    ax1.set_title('AUCの推移(訓練 vs 検証)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.8, 1.0])
    
    # F1スコアの比較
    ax2 = axes[0, 1]
    ax2.plot(folds, metrics_df['Train_F1'], marker='o', label='訓練データ', linewidth=2)
    ax2.plot(folds, metrics_df['Val_F1'], marker='s', label='検証データ', linewidth=2)
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1スコアの推移(訓練 vs 検証)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Recallの比較
    ax3 = axes[1, 0]
    ax3.plot(folds, metrics_df['Train_Recall'], marker='o', label='訓練データ', linewidth=2)
    ax3.plot(folds, metrics_df['Val_Recall'], marker='s', label='検証データ', linewidth=2)
    ax3.set_xlabel('Fold')
    ax3.set_ylabel('Recall')
    ax3.set_title('Recallの推移(訓練 vs 検証)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 差分の可視化
    ax4 = axes[1, 1]
    width = 0.25
    x = np.arange(len(folds))
    ax4.bar(x - width, metrics_df['Diff_AUC'], width, label='AUC差分')
    ax4.bar(x, metrics_df['Diff_F1'], width, label='F1差分')
    ax4.bar(x + width, metrics_df['Diff_Recall'], width, label='Recall差分')
    ax4.set_xlabel('Fold')
    ax4.set_ylabel('差分(訓練 - 検証)')
    ax4.set_title('各指標の差分')
    ax4.set_xticks(x)
    ax4.set_xticklabels(folds)
    ax4.legend()
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    output_path = 'results/model_comparison/logistic_regression_1pct/overfitting_analysis.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n✓ 過学習分析グラフを保存: {output_path}")
    plt.close()
    
    # CSVに保存
    csv_path = 'results/model_comparison/logistic_regression_1pct/overfitting_metrics.csv'
    metrics_df.to_csv(csv_path, index=False)
    print(f"✓ 詳細メトリクスを保存: {csv_path}")
    
    print("\n✅ 過学習分析完了")

if __name__ == "__main__":
    main()
