import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os

# 日本語フォントの設定 (Windows向け)
mpl.rcParams['font.family'] = 'MS Gothic'

def main():
    """
    ランダムフォレストモデルを訓練・評価する
    注意: データリークを防ぐため、訓練データのみに対してアップサンプリングを行う
    """
    
    print("=" * 80)
    print("ランダムフォレスト分析（訓練データのみアップサンプリング）")
    print("=" * 80)
    
    # 元データの読み込み（アップサンプリング前のデータを使用）
    file_path = 'data/raw/honhyo_all_shishasuu_binary.csv'
    print(f"\n📂 元データ読み込み中: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ データ読み込み完了: {len(df):,} 件")
    except FileNotFoundError:
        print(f"❌ エラー: ファイルが見つかりません - {file_path}")
        return
    except Exception as e:
        print(f"❌ エラー: {e}")
        return
    
    # 目的変数と不要な列の定義
    target_col = '死者数'
    drop_cols = ['資料区分', '本票番号']
    
    # 不要な列を削除
    print("\n🔧 データ前処理中...")
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # 特徴量と目的変数に分離
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 欠損値の処理
    print("  ・欠損値を処理しています...")
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
    
    # カテゴリ変数のエンコーディング
    print("  ・カテゴリ変数をエンコーディングしています...")
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))
    
    # データの分割 (学習データ: 80%, テストデータ: 20%)
    # stratify=y を指定して、分割後のクラス比率を維持する
    print("\n🔀 データを訓練用とテスト用に分割中...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  訓練データ: {len(X_train):,} 件")
    print(f"  テストデータ: {len(X_test):,} 件")
    
    # 訓練データのみアップサンプリング
    print("\n🔄 訓練データのアップサンプリングを実行中...")
    
    # 訓練データを結合して一時的なDataFrameを作成
    train_df = pd.concat([X_train, y_train], axis=1)
    
    # クラスごとに分離
    train_majority = train_df[train_df[target_col] == 0]
    train_minority = train_df[train_df[target_col] == 1]
    
    print(f"  アップサンプリング前（訓練データ）:")
    print(f"    多数派（0）: {len(train_majority):,} 件")
    print(f"    少数派（1）: {len(train_minority):,} 件")
    
    # 少数派クラスをアップサンプリング
    train_minority_upsampled = resample(
        train_minority,
        replace=True,
        n_samples=len(train_majority), # 多数派と同数に
        random_state=42
    )
    
    # アップサンプリング後の訓練データを結合
    train_upsampled = pd.concat([train_majority, train_minority_upsampled])
    
    # X_train, y_train を更新
    X_train_res = train_upsampled.drop(columns=[target_col])
    y_train_res = train_upsampled[target_col]
    
    print(f"  アップサンプリング後（訓練データ）: {len(X_train_res):,} 件")
    print(f"    多数派（0）: {sum(y_train_res==0):,} 件")
    print(f"    少数派（1）: {sum(y_train_res==1):,} 件")
    
    # ランダムフォレストモデルの構築と学習
    print("\n🌲 ランダムフォレストモデルを学習中...")
    rf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X_train_res, y_train_res)
    
    print("✓ モデル学習完了")
    
    # 予測と評価（テストデータは元の分布のまま評価）
    print("\n📈 テストデータで評価中...")
    y_pred = rf.predict(X_test)
    
    # 評価指標の算出
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print("\n" + "=" * 80)
    print("📊 評価結果 (テストデータ)")
    print("=" * 80)
    print(f"Accuracy (精度):    {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision (適合率): {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall (再現率):    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 Score:           {f1:.4f}")
    
    print("\n" + "-" * 80)
    print("詳細な分類レポート:")
    print("-" * 80)
    print(classification_report(y_test, y_pred, target_names=['非死亡事故', '死亡事故']))
    
    # 混同行列の作成と保存
    print("\n📉 混同行列を作成中...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=['非死亡事故 (0)', '死亡事故 (1)'],
        yticklabels=['非死亡事故 (0)', '死亡事故 (1)'],
        cbar_kws={'label': '件数'}
    )
    plt.title('混同行列 (訓練データのみアップサンプリング)', fontsize=16, pad=20)
    plt.ylabel('実際のクラス', fontsize=12)
    plt.xlabel('予測されたクラス', fontsize=12)
    plt.tight_layout()
    
    cm_path = 'results/visualizations/confusion_matrix_upsampled.png'
    os.makedirs(os.path.dirname(cm_path), exist_ok=True)
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✓ 混同行列を保存: {cm_path}")
    plt.close()
    
    # 特徴量重要度の表示と保存
    print("\n🔍 特徴量重要度を分析中...")
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n特徴量重要度 (Top 20):")
    print("-" * 80)
    for idx, row in feature_importances.head(20).iterrows():
        print(f"{row['feature']:45s}: {row['importance']:.6f}")
    
    # 重要度の可視化
    plt.figure(figsize=(12, 10))
    sns.barplot(
        x='importance', 
        y='feature', 
        data=feature_importances.head(20),
        palette='viridis'
    )
    plt.title('特徴量重要度 Top 20', fontsize=16, pad=20)
    plt.xlabel('重要度', fontsize=12)
    plt.ylabel('特徴量', fontsize=12)
    plt.tight_layout()
    
    fi_path = 'results/visualizations/feature_importance_upsampled.png'
    plt.savefig(fi_path, dpi=300, bbox_inches='tight')
    print(f"✓ 特徴量重要度グラフを保存: {fi_path}")
    plt.close()
    
    # 評価メトリクスをCSVに保存
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
        'Value': [accuracy, precision, recall, f1]
    })
    
    metrics_path = 'results/analysis/upsampled_model_metrics.csv'
    metrics_df.to_csv(metrics_path, index=False, encoding='utf-8-sig')
    print(f"✓ 評価メトリクスを保存: {metrics_path}")
    
    print("\n" + "=" * 80)
    print("✅ 分析完了")
    print("=" * 80)

if __name__ == "__main__":
    main()
