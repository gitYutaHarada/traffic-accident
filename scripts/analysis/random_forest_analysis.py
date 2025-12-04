import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# 日本語フォントの設定 (Windows向け)
mpl.rcParams['font.family'] = 'MS Gothic'

def main():
    # データの読み込み
    file_path = r'c:\Users\socce\software-lab\traffic-accident\honhyo_all\csv\honhyo_all_shishasuu_binary.csv'
    print("データを読み込んでいます...")
    try:
        df = pd.read_csv(file_path, encoding='cp932')
    except UnicodeDecodeError:
        print("cp932での読み込みに失敗しました。utf-8で試行します。")
        df = pd.read_csv(file_path, encoding='utf-8')

    # 目的変数と不要な列の定義
    target_col = '死者数'
    drop_cols = ['資料区分', '本票番号'] # IDや管理番号など分析に不要な列

    # 不要な列を削除
    print("不要な列を削除しています...")
    df = df.drop(columns=drop_cols, errors='ignore')

    # 特徴量と目的変数に分離
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 欠損値の処理
    print("欠損値を処理しています...")
    # 数値列は中央値で埋める
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())

    # カテゴリ列（オブジェクト型）は最頻値で埋める
    cat_cols = x.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0])

    # カテゴリ変数のエンコーディング (Label Encoding)
    print("カテゴリ変数をエンコーディングしています...")
    le = LabelEncoder()
    for col in cat_cols:
        # 文字列型に変換してからエンコード（混在データ対策）
        X[col] = le.fit_transform(X[col].astype(str))

    # データの分割 (学習データ: 80%, テストデータ: 20%)
    print("データを分割しています...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ランダムフォレストモデルの構築と学習
    print("ランダムフォレストモデルを学習しています...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 予測と評価
    print("評価を行っています...")
    y_pred = rf.predict(X_test)

    print("\n=== 評価レポート ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 特徴量重要度の表示（上位20個）
    feature_importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n=== 特徴量重要度 (Top 20) ===")
    print(feature_importances.head(20))

    # 重要度の可視化（オプション: 画像保存）
    plt.figure(figsize=(10, 12))
    sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
    plt.title('Random Forest Feature Importance (Top 20)')
    plt.tight_layout()
    plt.savefig('C:\Users\socce\software-lab\traffic-accident\results\visualizations\feature_importance.png')
    print("\n特徴量重要度のグラフを 'C:\Users\socce\software-lab\traffic-accident\results\visualizations\feature_importance.png' に保存しました。")

if __name__ == "__main__":
    main()


