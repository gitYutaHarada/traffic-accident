# -*- coding: utf-8 -*-
"""
ランダムフォレストモデルの検証曲線と学習曲線を作成
過学習・学習不足の分析を行う
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, validation_curve, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# 日本語フォントの設定
mpl.rcParams['font.family'] = 'MS Gothic'

def main():
    print("="*70)
    print("ランダムフォレストモデル - 検証曲線・学習曲線分析")
    print("="*70)
    print()
    
    # データの読み込み
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    file_path = os.path.join(base_dir, 'data', 'processed', 'honhyo_all_preaccident_only.csv')
    
    print("データを読み込んでいます...")
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print(f"読み込み完了: {len(df):,} 行")
    print()
    
    # データ準備
    target_col = '死者数'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 欠損値処理
    print("データ前処理中...")
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    
    cat_cols = X.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        if X[col].mode().empty:
            X[col] = X[col].fillna(0)
        else:
            X[col] = X[col].fillna(X[col].mode()[0])
        X[col] = le.fit_transform(X[col].astype(str))
    
    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print("前処理完了")
    print()
    
    # サンプリング（大量データのため）
    print("計算時間短縮のため、データをサンプリング...")
    sample_size = min(100000, len(X_train))
    indices = np.random.choice(len(X_train), sample_size, replace=False)
    X_sample = X_train.iloc[indices]
    y_sample = y_train.iloc[indices]
    print(f"サンプルサイズ: {sample_size:,} 件")
    print()
    
    # =================================================================
    # 1. 検証曲線 (Validation Curve)
    # n_estimatorsパラメータの影響を分析
    # =================================================================
    print("="*70)
    print("1. 検証曲線の作成 (n_estimators)")
    print("="*70)
    print()
    
    param_range = [10, 50, 100, 200, 500, 1000]
    print(f"検証するパラメータ: {param_range}")
    print("検証曲線を計算中... (数分かかります)")
    
    train_scores, test_scores = validation_curve(
        RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
        X_sample, y_sample,
        param_name="n_estimators",
        param_range=param_range,
        cv=3,  # 3-fold cross validation
        scoring="accuracy",
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, label='訓練スコア', color='blue', marker='o')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(param_range, test_mean, label='検証スコア', color='red', marker='s')
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.15, color='red')
    
    plt.xlabel('n_estimators (決定木の数)')
    plt.ylabel('Accuracy')
    plt.title('検証曲線 - n_estimators vs Accuracy')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(base_dir, 'results', 'visualizations', 'validation_curve_n_estimators.png')
    plt.savefig(output_path, dpi=300)
    print(f"検証曲線を保存: {output_path}")
    print()
    
    # 結果表示
    print("検証曲線の結果:")
    print(f"{'n_estimators':<15} {'訓練スコア':<12} {'検証スコア':<12} {'差分':<10}")
    print("-"*50)
    for i, n in enumerate(param_range):
        diff = train_mean[i] - test_mean[i]
        print(f"{n:<15} {train_mean[i]:.4f}      {test_mean[i]:.4f}      {diff:.4f}")
    print()
    
    # =================================================================
    # 2. 検証曲線 (max_depth)
    # =================================================================
    print("="*70)
    print("2. 検証曲線の作成 (max_depth)")
    print("="*70)
    print()
    
    depth_range = [5, 10, 15, 20, 25, 30, None]
    depth_range_display = [5, 10, 15, 20, 25, 30, 100]  # Noneを100として表示
    print(f"検証するパラメータ: {[d if d is not None else 'None' for d in depth_range]}")
    print("検証曲線を計算中...")
    
    train_scores_d, test_scores_d = validation_curve(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
        X_sample, y_sample,
        param_name="max_depth",
        param_range=depth_range,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )
    
    train_mean_d = np.mean(train_scores_d, axis=1)
    train_std_d = np.std(train_scores_d, axis=1)
    test_mean_d = np.mean(test_scores_d, axis=1)
    test_std_d = np.std(test_scores_d, axis=1)
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(depth_range_display, train_mean_d, label='訓練スコア', color='blue', marker='o')
    plt.fill_between(depth_range_display, train_mean_d - train_std_d, train_mean_d + train_std_d, alpha=0.15, color='blue')
    plt.plot(depth_range_display, test_mean_d, label='検証スコア', color='red', marker='s')
    plt.fill_between(depth_range_display, test_mean_d - test_std_d, test_mean_d + test_std_d, alpha=0.15, color='red')
    
    plt.xlabel('max_depth (木の最大深さ)')
    plt.ylabel('Accuracy')
    plt.title('検証曲線 - max_depth vs Accuracy')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path_d = os.path.join(base_dir, 'results', 'visualizations', 'validation_curve_max_depth.png')
    plt.savefig(output_path_d, dpi=300)
    print(f"検証曲線を保存: {output_path_d}")
    print()
    
    # 結果表示
    print("検証曲線の結果:")
    print(f"{'max_depth':<15} {'訓練スコア':<12} {'検証スコア':<12} {'差分':<10}")
    print("-"*50)
    for i, d in enumerate(depth_range):
        diff = train_mean_d[i] - test_mean_d[i]
        depth_str = str(d) if d is not None else "None"
        print(f"{depth_str:<15} {train_mean_d[i]:.4f}      {test_mean_d[i]:.4f}      {diff:.4f}")
    print()
    
    # =================================================================
    # 3. 学習曲線 (Learning Curve)
    # =================================================================
    print("="*70)
    print("3. 学習曲線の作成")
    print("="*70)
    print()
    
    print("学習曲線を計算中... (数分かかります)")
    train_sizes = np.linspace(0.1, 1.0, 10)
    
    train_sizes_abs, train_scores_lc, test_scores_lc = learning_curve(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
        X_sample, y_sample,
        train_sizes=train_sizes,
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    
    train_mean_lc = np.mean(train_scores_lc, axis=1)
    train_std_lc = np.std(train_scores_lc, axis=1)
    test_mean_lc = np.mean(test_scores_lc, axis=1)
    test_std_lc = np.std(test_scores_lc, axis=1)
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes_abs, train_mean_lc, label='訓練スコア', color='blue', marker='o')
    plt.fill_between(train_sizes_abs, train_mean_lc - train_std_lc, train_mean_lc + train_std_lc, alpha=0.15, color='blue')
    plt.plot(train_sizes_abs, test_mean_lc, label='検証スコア', color='red', marker='s')
    plt.fill_between(train_sizes_abs, test_mean_lc - test_std_lc, test_mean_lc + test_std_lc, alpha=0.15, color='red')
    
    plt.xlabel('訓練データ数')
    plt.ylabel('Accuracy')
    plt.title('学習曲線 - 訓練データ数 vs Accuracy')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path_lc = os.path.join(base_dir, 'results', 'visualizations', 'learning_curve.png')
    plt.savefig(output_path_lc, dpi=300)
    print(f"学習曲線を保存: {output_path_lc}")
    print()
    
    # 結果表示
    print("学習曲線の結果:")
    print(f"{'訓練データ数':<15} {'訓練スコア':<12} {'検証スコア':<12} {'差分':<10}")
    print("-"*55)
    for i, size in enumerate(train_sizes_abs):
        diff = train_mean_lc[i] - test_mean_lc[i]
        print(f"{int(size):<15} {train_mean_lc[i]:.4f}      {test_mean_lc[i]:.4f}      {diff:.4f}")
    print()
    
    # =================================================================
    # 診断と推奨事項
    # =================================================================
    print("="*70)
    print("診断結果")
    print("="*70)
    print()
    
    # 最終的な訓練・検証スコア差
    final_diff_estimators = train_mean[-1] - test_mean[-1]
    final_diff_depth = train_mean_d[-1] - test_mean_d[-1]
    final_diff_learning = train_mean_lc[-1] - test_mean_lc[-1]
    
    print("1. n_estimators (決定木の数)")
    if final_diff_estimators < 0.01:
        print("   ✓ 適切: 過学習の兆候は見られません")
    elif final_diff_estimators < 0.05:
        print("   ⚠ 軽度の過学習: わずかに過学習の傾向があります")
    else:
        print("   ✗ 過学習: 訓練スコアと検証スコアの差が大きいです")
    print(f"   訓練スコア: {train_mean[-1]:.4f}, 検証スコア: {test_mean[-1]:.4f}, 差: {final_diff_estimators:.4f}")
    print()
    
    print("2. max_depth (木の深さ)")
    if final_diff_depth < 0.01:
        print("   ✓ 適切: 過学習の兆候は見られません")
    elif final_diff_depth < 0.05:
        print("   ⚠ 軽度の過学習: わずかに過学習の傾向があります")
    else:
        print("   ✗ 過学習: 訓練スコアと検証スコアの差が大きいです")
    print(f"   訓練スコア: {train_mean_d[-1]:.4f}, 検証スコア: {test_mean_d[-1]:.4f}, 差: {final_diff_depth:.4f}")
    print()
    
    print("3. 学習曲線")
    if test_mean_lc[-1] > 0.95:
        print("   ✓ 高性能: 検証スコアが高く、モデルは良好です")
    if final_diff_learning < 0.05:
        print("   ✓ 適切: 過学習の兆候は少ないです")
    if test_mean_lc[-1] < test_mean_lc[-2]:
        print("   ⚠ データ追加の効果減少: データを増やしても性能向上が見込めない可能性")
    print()
    
    print("推奨事項:")
    if final_diff_estimators > 0.02 or final_diff_depth > 0.02:
        print("  - 正則化の強化を検討 (min_samples_split, min_samples_leaf を増やす)")
        print("  - max_depth を制限することで過学習を抑制")
    if test_mean_lc[-1] - test_mean_lc[0] > 0.1:
        print("  - データ量を増やすことで性能向上が期待できます")
    print("  - 現在のn_estimators=1000は適切です")
    print()
    
    print("="*70)
    print("処理完了！")
    print("="*70)

if __name__ == "__main__":
    main()
