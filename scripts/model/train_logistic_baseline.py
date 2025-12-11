"""
ロジスティック回帰によるベースラインモデル。
LightGBM最終モデルと同一データ分割(80/20, stratify, random_state=42)で評価し、
PR-AUC/ROC-AUC/閾値別指標を出力する。

実行: python scripts/model/train_logistic_baseline.py
"""

import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Meiryo'

DATA_PATH = 'data/processed/honhyo_clean_predictable_only.csv'
OUTPUT_DIR = 'results/model_comparison/logistic_regression'
TEST_SIZE = 0.2
RANDOM_STATE = 42
LOW_CARD_THRESHOLD = 30  # この値以下はOne-Hot、それ以上はOrdinal


def load_data(path):
    print('=' * 60)
    print('Step 1: データ読み込み')
    print('=' * 60)
    df = pd.read_csv(path)
    print(f"データサイズ: {df.shape[0]:,} 行 × {df.shape[1]} 列")

    target_col = '死者数'
    drop_cols = []
    if '発生日時' in df.columns:
        drop_cols.append('発生日時')

    X = df.drop(columns=[target_col] + drop_cols)
    y = df[target_col].astype(int)

    print('\nクラス分布:')
    print(f"  非死亡 (0): {(y == 0).sum():,} ({(y == 0).mean()*100:.2f}%)")
    print(f"  死亡   (1): {(y == 1).sum():,} ({(y == 1).mean()*100:.2f}%)")
    print(f"  不均衡比: {(y == 0).sum() / (y == 1).sum():.2f}")

    return X, y


def build_preprocessor(X):
    print('\n' + '=' * 60)
    print('Step 2: 前処理パイプライン構築')
    print('=' * 60)

    cat_cols = [c for c in X.columns if X[c].dtype == 'object' or str(X[c].dtype).startswith('category')]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # 型をcategoryに変換
    for col in cat_cols:
        X[col] = X[col].astype('category')

    low_card_cols = [c for c in cat_cols if X[c].nunique() <= LOW_CARD_THRESHOLD]
    high_card_cols = [c for c in cat_cols if X[c].nunique() > LOW_CARD_THRESHOLD]

    print(f"カテゴリ列: {len(cat_cols)} (低カーディナリティ {len(low_card_cols)}, 高カーディナリティ {len(high_card_cols)})")
    print(f"数値列: {len(num_cols)}")

    num_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ]
    )

    low_card_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore')),
        ]
    )

    # OrdinalEncoderは未知カテゴリを-1で埋める
    high_card_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
            ('scaler', StandardScaler()),
        ]
    )

    transformers = []
    if num_cols:
        transformers.append(('num', num_transformer, num_cols))
    if low_card_cols:
        transformers.append(('low_cat', low_card_transformer, low_card_cols))
    if high_card_cols:
        transformers.append(('high_cat', high_card_transformer, high_card_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    return preprocessor


def train_and_evaluate(X, y, preprocessor):
    print('\n' + '=' * 60)
    print('Step 3: 学習・評価')
    print('=' * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    clf = Pipeline(
        steps=[
            ('preprocess', preprocessor),
            ('model', LogisticRegression(
                class_weight='balanced',
                max_iter=200,
                solver='lbfgs',
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ))
        ]
    )

    print('\n学習開始...')
    start = datetime.now()
    clf.fit(X_train, y_train)
    elapsed = datetime.now() - start
    print(f"学習完了! (所要時間: {elapsed})")

    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print('\n【テスト評価】')
    print(f"  PR-AUC:  {pr_auc:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")

    thresholds_fixed = [0.1, 0.2, 0.3, 0.4, 0.5]
    threshold_results = []
    print('\n閾値別の評価:')
    for thresh in thresholds_fixed:
        y_pred = (y_pred_proba >= thresh).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        threshold_results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        })
        print(f"  閾値={thresh:.1f}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")

    precision_arr, recall_arr, thresholds_arr = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * (precision_arr[:-1] * recall_arr[:-1]) / (precision_arr[:-1] + recall_arr[:-1] + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds_arr[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"\n最適閾値 (F1最大化): {best_threshold:.4f}")
    print(f"  - F1:        {best_f1:.4f}")
    print(f"  - Precision: {precision_arr[best_idx]:.4f}")
    print(f"  - Recall:    {recall_arr[best_idx]:.4f}")

    results = {
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'threshold_results': threshold_results,
        'best_threshold': float(best_threshold),
        'best_f1': float(best_f1),
        'y_test': y_test,
        'y_pred_proba': y_pred_proba,
        'precision_curve': precision_arr,
        'recall_curve': recall_arr,
        'thresholds_curve': thresholds_arr,
        'elapsed': str(elapsed),
    }

    return clf, results


def plot_and_save(results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # PR曲線
    plt.figure(figsize=(7, 5))
    plt.plot(results['recall_curve'], results['precision_curve'], label='Logistic Regression')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 閾値分析
    precision_arr = results['precision_curve']
    recall_arr = results['recall_curve']
    thresholds_arr = results['thresholds_curve']
    f1_scores = 2 * (precision_arr[:-1] * recall_arr[:-1]) / (precision_arr[:-1] + recall_arr[:-1] + 1e-10)

    plt.figure(figsize=(7, 5))
    plt.plot(thresholds_arr, precision_arr[:-1], label='Precision')
    plt.plot(thresholds_arr, recall_arr[:-1], label='Recall')
    plt.plot(thresholds_arr, f1_scores, label='F1')
    plt.axvline(results['best_threshold'], color='gray', linestyle='--', label=f"Best {results['best_threshold']:.3f}")
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Threshold vs Precision/Recall/F1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'threshold_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 混同行列
    y_pred = (results['y_pred_proba'] >= results['best_threshold']).astype(int)
    cm = confusion_matrix(results['y_test'], y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['非死亡 (予測)', '死亡 (予測)'],
                yticklabels=['非死亡 (実際)', '死亡 (実際)'])
    plt.title(f"Confusion Matrix (thr={results['best_threshold']:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"グラフを保存: {output_dir}/pr_curve.png, threshold_analysis.png, confusion_matrix.png")


def save_reports(clf, results, preprocessor, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # モデル保存
    model_path = os.path.join(output_dir, 'logreg_model.pkl')
    joblib.dump(clf, model_path)

    # 係数の展開
    feature_names = clf.named_steps['preprocess'].get_feature_names_out()
    coefs = clf.named_steps['model'].coef_[0]
    coef_df = pd.DataFrame({'feature': feature_names, 'coefficient': coefs})
    coef_df.sort_values('coefficient', ascending=False).to_csv(os.path.join(output_dir, 'coefficients.csv'), index=False)

    # 閾値別表をCSVで保存
    pd.DataFrame(results['threshold_results']).to_csv(os.path.join(output_dir, 'threshold_metrics.csv'), index=False)

    # メトリクスJSON
    metrics = {
        'pr_auc': results['pr_auc'],
        'roc_auc': results['roc_auc'],
        'best_threshold': results['best_threshold'],
        'best_f1': results['best_f1'],
        'elapsed': results['elapsed'],
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Markdownレポート
    report_path = os.path.join(output_dir, 'logistic_regression_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# ロジスティック回帰 ベースライン評価\n\n')
        f.write(f"**評価日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}\n\n")
        f.write('## 1. 最終評価結果\n\n')
        f.write('| 指標 | 値 |\n')
        f.write('|------|-----|\n')
        f.write(f"| PR-AUC | {results['pr_auc']:.4f} |\n")
        f.write(f"| ROC-AUC | {results['roc_auc']:.4f} |\n")
        f.write(f"| 最適閾値 (F1最大化) | {results['best_threshold']:.4f} |\n")
        f.write(f"| 学習時間 | {results['elapsed']} |\n\n")

        f.write('## 2. 閾値別の性能\n\n')
        f.write('| 閾値 | Precision | Recall | F1 |\n')
        f.write('|------|-----------|--------|----|\n')
        for r in results['threshold_results']:
            f.write(f"| {r['threshold']:.1f} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1']:.4f} |\n")

        f.write('\n## 3. 使用ハイパーパラメータ\n\n')
        f.write('- ロジスティック回帰: lbfgs, C=1.0, penalty=L2, class_weight=balanced, max_iter=200\n')
        f.write(f"- 前処理: One-Hot (<= {LOW_CARD_THRESHOLD}カテゴリ), Ordinal + StandardScaler\n")
        f.write('\n## 4. データセット\n\n')
        f.write(f"- ソース: {DATA_PATH}\n")
        f.write(f"- テストサイズ: {int(len(results['y_test'])):,} 件\n")
        f.write(f"- テスト死亡件数: {int(results['y_test'].sum()):,} 件\n")

    print(f"モデル・レポートを保存: {output_dir}")


def main():
    print('=' * 60)
    print('ロジスティック回帰 ベースライン')
    print('=' * 60)
    print(f"実行開始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    X, y = load_data(DATA_PATH)
    preprocessor = build_preprocessor(X)
    clf, results = train_and_evaluate(X, y, preprocessor)
    plot_and_save(results, OUTPUT_DIR)
    save_reports(clf, results, preprocessor, OUTPUT_DIR)

    print('\n完了')
    print(f"出力先: {OUTPUT_DIR}/")
    print('  - logreg_model.pkl')
    print('  - metrics.json')
    print('  - threshold_metrics.csv')
    print('  - coefficients.csv')
    print('  - logistic_regression_report.md')
    print('  - pr_curve.png, threshold_analysis.png, confusion_matrix.png')


if __name__ == '__main__':
    main()
