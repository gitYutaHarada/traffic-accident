"""
最終モデル構築・検証スクリプト
==============================
チューニング済みLightGBMパラメータで最終モデルを構築し、
独立したテストセットで性能を検証する。

実行方法:
    python scripts/model/train_final_model.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    precision_recall_curve, average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

# 日本語フォント設定
plt.rcParams['font.family'] = 'Meiryo'

# ==============================================================================
# 設定
# ==============================================================================

# データパス
DATA_PATH = 'data/processed/honhyo_clean_predictable_only.csv'

# 出力ディレクトリ
OUTPUT_DIR = 'results/final_model'

# Trial 153 の最良パラメータ
BEST_PARAMS = {
    'learning_rate': 0.07658346283890378,
    'num_leaves': 125,
    'max_depth': 8,
    'min_child_samples': 278,
    'subsample': 0.6147706754536576,
    'colsample_bytree': 0.6267708320804088,
    'reg_alpha': 0.9961403311275829,
    'reg_lambda': 8.228908331551605,
    'min_child_weight': 0.12646850234127796,
    'min_split_gain': 0.24303906753172422,
    'path_smooth': 2.254892007170922,
    'scale_pos_weight': 61.47728365878301,
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'n_estimators': 10000,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# Hold-out設定
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_and_prepare_data(data_path):
    """データを読み込み、特徴量とターゲットに分割"""
    print("=" * 60)
    print("Step 1: データ読み込み")
    print("=" * 60)
    
    df = pd.read_csv(data_path)
    print(f"データサイズ: {df.shape[0]:,} 行 × {df.shape[1]} 列")
    
    # ターゲット変数
    target_col = '死者数'
    
    # 発生日時があれば削除
    drop_cols = []
    if '発生日時' in df.columns:
        drop_cols.append('発生日時')
    
    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col]
    
    # クラス分布
    print(f"\nクラス分布:")
    print(f"  非死亡 (0): {(y == 0).sum():,} ({(y == 0).mean()*100:.2f}%)")
    print(f"  死亡   (1): {(y == 1).sum():,} ({(y == 1).mean()*100:.2f}%)")
    print(f"  不均衡比: {(y == 0).sum() / (y == 1).sum():.2f}")
    
    # カテゴリカル変数の変換
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category')
    
    print(f"\n特徴量数: {X.shape[1]}")
    
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """データをHold-out分割"""
    print("\n" + "=" * 60)
    print("Step 2: データ分割 (Hold-out)")
    print("=" * 60)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n訓練+検証データ: {X_train_val.shape[0]:,} 件 ({(1-test_size)*100:.0f}%)")
    print(f"  - 非死亡: {(y_train_val == 0).sum():,}")
    print(f"  - 死亡:   {(y_train_val == 1).sum():,}")
    
    print(f"\nテストデータ: {X_test.shape[0]:,} 件 ({test_size*100:.0f}%)")
    print(f"  - 非死亡: {(y_test == 0).sum():,}")
    print(f"  - 死亡:   {(y_test == 1).sum():,}")
    
    return X_train_val, X_test, y_train_val, y_test


def train_final_model(X_train_val, y_train_val, params):
    """最終モデルを学習"""
    print("\n" + "=" * 60)
    print("Step 3: 最終モデルの学習")
    print("=" * 60)
    
    # 検証用に訓練データをさらに分割
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.1, random_state=42, stratify=y_train_val
    )
    
    print(f"\n学習データ: {X_train.shape[0]:,} 件")
    print(f"検証データ: {X_val.shape[0]:,} 件")
    
    # モデル作成
    model = lgb.LGBMClassifier(**params)
    
    print("\n学習開始...")
    start_time = datetime.now()
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    
    elapsed = datetime.now() - start_time
    print(f"\n学習完了! (所要時間: {elapsed})")
    print(f"最適イテレーション数: {model.best_iteration_}")
    
    # 検証データでの性能
    y_val_pred_proba = model.predict_proba(X_val)[:, 1]
    val_pr_auc = average_precision_score(y_val, y_val_pred_proba)
    val_roc_auc = roc_auc_score(y_val, y_val_pred_proba)
    
    print(f"\n検証データでの性能:")
    print(f"  PR-AUC:  {val_pr_auc:.4f}")
    print(f"  ROC-AUC: {val_roc_auc:.4f}")
    
    return model


def evaluate_on_test(model, X_test, y_test, output_dir):
    """テストセットで最終評価"""
    print("\n" + "=" * 60)
    print("Step 4: テストセットでの最終評価")
    print("=" * 60)
    
    # 予測
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # 主要指標
    pr_auc = average_precision_score(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n【最終評価結果】")
    print(f"  PR-AUC:  {pr_auc:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    
    # 閾値別の評価
    print("\n閾値別の評価:")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
    threshold_results = []
    
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        threshold_results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        print(f"  閾値={thresh:.1f}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    # 結果を保存
    results = {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'threshold_results': threshold_results,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba
    }
    
    return results


def analyze_thresholds(y_test, y_pred_proba, output_dir):
    """閾値分析と可視化"""
    print("\n" + "=" * 60)
    print("Step 5: 閾値分析")
    print("=" * 60)
    
    # Precision-Recall曲線
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # F1スコアの計算
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    
    print(f"\n最適閾値 (F1最大化): {best_threshold:.4f}")
    print(f"  - F1:        {best_f1:.4f}")
    print(f"  - Precision: {precision[best_f1_idx]:.4f}")
    print(f"  - Recall:    {recall[best_f1_idx]:.4f}")
    
    # 可視化: Precision-Recall曲線
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # PR曲線
    ax1 = axes[0]
    ax1.plot(recall, precision, 'b-', linewidth=2)
    ax1.scatter([recall[best_f1_idx]], [precision[best_f1_idx]], 
                color='red', s=100, zorder=5, label=f'最適閾値 ({best_threshold:.3f})')
    ax1.set_xlabel('Recall (再現率)', fontsize=12)
    ax1.set_ylabel('Precision (適合率)', fontsize=12)
    ax1.set_title('Precision-Recall 曲線', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 閾値 vs 各指標
    ax2 = axes[1]
    ax2.plot(thresholds, precision[:-1], 'b-', label='Precision', linewidth=2)
    ax2.plot(thresholds, recall[:-1], 'g-', label='Recall', linewidth=2)
    ax2.plot(thresholds, f1_scores, 'r-', label='F1', linewidth=2)
    ax2.axvline(x=best_threshold, color='gray', linestyle='--', label=f'最適閾値 ({best_threshold:.3f})')
    ax2.set_xlabel('閾値', fontsize=12)
    ax2.set_ylabel('スコア', fontsize=12)
    ax2.set_title('閾値とPrecision/Recall/F1の関係', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n閾値分析グラフを保存: {output_dir}/threshold_analysis.png")
    
    return best_threshold, best_f1


def plot_confusion_matrix(y_test, y_pred_proba, threshold, output_dir):
    """混同行列の可視化"""
    y_pred = (y_pred_proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['非死亡 (予測)', '死亡 (予測)'],
                yticklabels=['非死亡 (実際)', '死亡 (実際)'])
    ax.set_title(f'混同行列 (閾値={threshold:.3f})', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"混同行列を保存: {output_dir}/confusion_matrix.png")
    
    # 詳細レポート
    print(f"\n混同行列 (閾値={threshold:.3f}):")
    print(f"  TN (正しく非死亡と予測): {cm[0, 0]:,}")
    print(f"  FP (誤って死亡と予測):   {cm[0, 1]:,}")
    print(f"  FN (見逃し):             {cm[1, 0]:,}")
    print(f"  TP (正しく死亡と予測):   {cm[1, 1]:,}")
    
    return cm


def plot_feature_importance(model, output_dir, top_n=20):
    """特徴量重要度の可視化"""
    importance_df = pd.DataFrame({
        'feature': model.feature_name_,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # 保存
    importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
    
    # 可視化
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = importance_df.head(top_n)
    ax.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('重要度', fontsize=12)
    ax.set_title(f'特徴量重要度 (Top {top_n})', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"特徴量重要度を保存: {output_dir}/feature_importance.png")
    
    return importance_df


def save_results(model, results, best_threshold, output_dir):
    """結果を保存"""
    print("\n" + "=" * 60)
    print("Step 6: 結果の保存")
    print("=" * 60)
    
    # モデル保存
    model_path = os.path.join(output_dir, 'final_model.pkl')
    joblib.dump(model, model_path)
    print(f"\nモデルを保存: {model_path}")
    
    # パラメータ保存
    params_df = pd.DataFrame([BEST_PARAMS])
    params_df.to_csv(os.path.join(output_dir, 'best_params.csv'), index=False)
    
    # 評価結果をMarkdownレポートとして保存
    report_path = os.path.join(output_dir, 'final_evaluation_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 最終モデル評価レポート\n\n")
        f.write(f"**評価日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}\n\n")
        
        f.write("## 1. 最終評価結果\n\n")
        f.write("| 指標 | 値 |\n")
        f.write("|------|-----|\n")
        f.write(f"| **PR-AUC** | **{results['pr_auc']:.4f}** |\n")
        f.write(f"| ROC-AUC | {results['roc_auc']:.4f} |\n")
        f.write(f"| 最適閾値 (F1最大化) | {best_threshold:.4f} |\n\n")
        
        f.write("## 2. 閾値別の性能\n\n")
        f.write("| 閾値 | Precision | Recall | F1 |\n")
        f.write("|------|-----------|--------|----|\n")
        for r in results['threshold_results']:
            f.write(f"| {r['threshold']:.1f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |\n")
        
        f.write("\n## 3. データセット情報\n\n")
        f.write(f"- **データソース**: {DATA_PATH}\n")
        f.write(f"- **テストデータサイズ**: {len(results['y_test']):,} 件\n")
        f.write(f"- **テストデータの死亡件数**: {results['y_test'].sum():,} 件\n\n")
        
        f.write("## 4. 使用パラメータ\n\n")
        f.write("```json\n")
        import json
        f.write(json.dumps(BEST_PARAMS, indent=2))
        f.write("\n```\n")
    
    print(f"評価レポートを保存: {report_path}")


def main():
    """メイン実行"""
    print("=" * 60)
    print("最終モデル構築・検証")
    print("=" * 60)
    print(f"実行開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 出力ディレクトリ作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Step 1: データ読み込み
    X, y = load_and_prepare_data(DATA_PATH)
    
    # Step 2: Hold-out分割
    X_train_val, X_test, y_train_val, y_test = split_data(X, y, TEST_SIZE, RANDOM_STATE)
    
    # Step 3: モデル学習
    model = train_final_model(X_train_val, y_train_val, BEST_PARAMS)
    
    # Step 4: テストセット評価
    results = evaluate_on_test(model, X_test, y_test, OUTPUT_DIR)
    
    # Step 5: 閾値分析
    best_threshold, best_f1 = analyze_thresholds(
        results['y_test'], results['y_pred_proba'], OUTPUT_DIR
    )
    
    # 混同行列
    plot_confusion_matrix(results['y_test'], results['y_pred_proba'], best_threshold, OUTPUT_DIR)
    
    # 特徴量重要度
    plot_feature_importance(model, OUTPUT_DIR)
    
    # Step 6: 結果保存
    save_results(model, results, best_threshold, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("完了!")
    print("=" * 60)
    print(f"\n出力先: {OUTPUT_DIR}/")
    print("  - final_model.pkl (学習済みモデル)")
    print("  - final_evaluation_report.md (評価レポート)")
    print("  - threshold_analysis.png (閾値分析)")
    print("  - confusion_matrix.png (混同行列)")
    print("  - feature_importance.png (特徴量重要度)")
    print("  - feature_importance.csv (特徴量重要度CSV)")
    print("  - best_params.csv (パラメータ)")


if __name__ == '__main__':
    main()
