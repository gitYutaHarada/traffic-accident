"""
Stage 1 ロジスティックメタモデル実験（Intel最適化版）
====================================================
OOF予測値を入力とするロジスティック回帰を学習し、
Max Probability をベースラインとして比較する。

最適化:
- Intel Extension for Scikit-learn (sklearnex) を使用
- joblib.Parallel で 5-Fold CV を並列化

入力:
    data/processed/stage1_oof_predictions.csv
    data/processed/stage1_test_predictions.csv

出力:
    results/stage1_experiments/meta_model_report.md

実行方法:
    python scripts/experiment/stage1_meta_model_test.py
"""

# Intel Extension for Scikit-learn（最初に読み込む）
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("✅ Intel Extension for Scikit-learn が有効化されました")
except ImportError:
    print("⚠️ sklearnex がインストールされていません。'pip install scikit-learn-intelex' を推奨")

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, roc_auc_score, precision_recall_curve
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')


def find_threshold_for_recall_fast(proba: np.ndarray, y: np.ndarray, target_recall: float = 0.995) -> tuple:
    """precision_recall_curveを使用して高速に閾値探索"""
    precision_arr, recall_arr, thresh_arr = precision_recall_curve(y, proba)
    
    valid_idx = np.where(recall_arr >= target_recall)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[-1]
        if best_idx < len(thresh_arr):
            best_thresh = thresh_arr[best_idx]
        else:
            best_thresh = 0.0
        actual_recall = recall_arr[best_idx]
        actual_precision = precision_arr[best_idx]
    else:
        best_thresh = 0.0
        actual_recall = 1.0
        actual_precision = y.mean()
    
    pred = (proba >= best_thresh).astype(int)
    pass_rate = pred.mean()
    
    return best_thresh, actual_recall, pass_rate, actual_precision


def train_fold(fold, train_idx, val_idx, X_meta, y, random_state):
    """1つのFoldを処理する関数（並列化用）"""
    X_train, X_val = X_meta[train_idx], X_meta[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model = LogisticRegression(
        solver='lbfgs',
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    val_proba = model.predict_proba(X_val)[:, 1]
    
    return {
        'fold': fold,
        'val_idx': val_idx,
        'val_proba': val_proba,
        'model': model,
    }


def run_meta_model_experiment(
    oof_path: str = "data/processed/stage1_oof_predictions.csv",
    test_path: str = "data/processed/stage1_test_predictions.csv",
    target_recall: float = 0.995,
    output_dir: str = "results/stage1_experiments",
):
    """ロジスティックメタモデル実験を実行"""
    
    print("=" * 70)
    print("Stage 1 ロジスティックメタモデル実験 (Intel最適化版)")
    print("=" * 70)
    
    # データ読み込み
    print("\n📂 データを読み込み中...")
    oof_df = pd.read_csv(oof_path)
    test_df = pd.read_csv(test_path)
    
    prob_lgbm_oof = oof_df['prob_lgbm'].values
    prob_catboost_oof = oof_df['prob_catboost'].values
    y_oof = oof_df['target'].values
    
    prob_lgbm_test = test_df['prob_lgbm'].values
    prob_catboost_test = test_df['prob_catboost'].values
    y_test = test_df['target'].values
    
    print(f"   OOF: {len(oof_df):,} 件 (正例: {y_oof.sum():,})")
    print(f"   Test: {len(test_df):,} 件 (正例: {y_test.sum():,})")
    
    # 特徴量行列（多重共線性を避けるためシンプルに）
    X_meta_oof = np.column_stack([prob_lgbm_oof, prob_catboost_oof])
    X_meta_test = np.column_stack([prob_lgbm_test, prob_catboost_test])
    feature_names = ['prob_lgbm', 'prob_catboost']
    
    # 5-Fold CV（並列化）
    n_folds = 5
    random_state = 42
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    print("\n🧠 ロジスティック回帰メタモデル学習中（並列処理）...")
    
    # 全Foldを並列実行
    fold_results = Parallel(n_jobs=-1, verbose=1)(
        delayed(train_fold)(fold, train_idx, val_idx, X_meta_oof, y_oof, random_state)
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_meta_oof, y_oof))
    )
    
    # 結果を集約
    meta_oof_proba = np.zeros(len(y_oof))
    meta_test_proba = np.zeros(len(y_test))
    models = []
    
    for result in sorted(fold_results, key=lambda x: x['fold']):
        val_idx = result['val_idx']
        meta_oof_proba[val_idx] = result['val_proba']
        models.append(result['model'])
        meta_test_proba += result['model'].predict_proba(X_meta_test)[:, 1] / n_folds
    
    meta_oof_auc = roc_auc_score(y_oof, meta_oof_proba)
    meta_test_auc = roc_auc_score(y_test, meta_test_proba)
    print(f"   メタモデル OOF AUC: {meta_oof_auc:.4f}")
    print(f"   メタモデル Test AUC: {meta_test_auc:.4f}")
    
    # ========== 比較 ==========
    print("\n📊 各手法の比較...")
    
    results = []
    
    # 1. Max Probability（真のベースライン）
    prob_max_oof = np.maximum(prob_lgbm_oof, prob_catboost_oof)
    prob_max_test = np.maximum(prob_lgbm_test, prob_catboost_test)
    
    thresh_max, rec_max, pass_max, prec_max = find_threshold_for_recall_fast(prob_max_oof, y_oof, target_recall)
    
    pred_test_max = (prob_max_test >= thresh_max).astype(int)
    rec_test_max = recall_score(y_test, pred_test_max)
    pass_test_max = pred_test_max.mean()
    prec_test_max = precision_score(y_test, pred_test_max) if pred_test_max.sum() > 0 else 0
    
    results.append({
        'method': 'Max Probability (Baseline)',
        'oof_threshold': thresh_max,
        'oof_recall': rec_max,
        'oof_pass_rate': pass_max,
        'oof_precision': prec_max,
        'oof_auc': roc_auc_score(y_oof, prob_max_oof),
        'test_recall': rec_test_max,
        'test_pass_rate': pass_test_max,
        'test_precision': prec_test_max,
        'test_auc': roc_auc_score(y_test, prob_max_test),
    })
    
    # 2. メタモデル
    thresh_meta, rec_meta, pass_meta, prec_meta = find_threshold_for_recall_fast(meta_oof_proba, y_oof, target_recall)
    
    pred_test_meta = (meta_test_proba >= thresh_meta).astype(int)
    rec_test_meta = recall_score(y_test, pred_test_meta)
    pass_test_meta = pred_test_meta.mean()
    prec_test_meta = precision_score(y_test, pred_test_meta) if pred_test_meta.sum() > 0 else 0
    
    results.append({
        'method': 'Logistic Meta-Model',
        'oof_threshold': thresh_meta,
        'oof_recall': rec_meta,
        'oof_pass_rate': pass_meta,
        'oof_precision': prec_meta,
        'oof_auc': meta_oof_auc,
        'test_recall': rec_test_meta,
        'test_pass_rate': pass_test_meta,
        'test_precision': prec_test_meta,
        'test_auc': meta_test_auc,
    })
    
    # 3. 単純な平均
    prob_avg_oof = (prob_lgbm_oof + prob_catboost_oof) / 2
    prob_avg_test = (prob_lgbm_test + prob_catboost_test) / 2
    
    thresh_avg, rec_avg, pass_avg, prec_avg = find_threshold_for_recall_fast(prob_avg_oof, y_oof, target_recall)
    
    pred_test_avg = (prob_avg_test >= thresh_avg).astype(int)
    rec_test_avg = recall_score(y_test, pred_test_avg)
    pass_test_avg = pred_test_avg.mean()
    prec_test_avg = precision_score(y_test, pred_test_avg) if pred_test_avg.sum() > 0 else 0
    
    results.append({
        'method': 'Simple Average',
        'oof_threshold': thresh_avg,
        'oof_recall': rec_avg,
        'oof_pass_rate': pass_avg,
        'oof_precision': prec_avg,
        'oof_auc': roc_auc_score(y_oof, prob_avg_oof),
        'test_recall': rec_test_avg,
        'test_pass_rate': pass_test_avg,
        'test_precision': prec_test_avg,
        'test_auc': roc_auc_score(y_test, prob_avg_test),
    })
    
    # 4. 参考: 旧OR条件
    thresh_lgbm_ind, _, _, _ = find_threshold_for_recall_fast(prob_lgbm_oof, y_oof, target_recall)
    thresh_cat_ind, _, _, _ = find_threshold_for_recall_fast(prob_catboost_oof, y_oof, target_recall)
    
    pred_or_oof = np.maximum(
        (prob_lgbm_oof >= thresh_lgbm_ind).astype(int),
        (prob_catboost_oof >= thresh_cat_ind).astype(int)
    )
    pred_or_test = np.maximum(
        (prob_lgbm_test >= thresh_lgbm_ind).astype(int),
        (prob_catboost_test >= thresh_cat_ind).astype(int)
    )
    
    rec_or_oof = recall_score(y_oof, pred_or_oof)
    pass_or_oof = pred_or_oof.mean()
    prec_or_oof = precision_score(y_oof, pred_or_oof) if pred_or_oof.sum() > 0 else 0
    
    rec_or_test = recall_score(y_test, pred_or_test)
    pass_or_test = pred_or_test.mean()
    prec_or_test = precision_score(y_test, pred_or_test) if pred_or_test.sum() > 0 else 0
    
    results.append({
        'method': 'OR Ensemble (Reference)',
        'oof_threshold': f"LGBM:{thresh_lgbm_ind:.4f}, Cat:{thresh_cat_ind:.4f}",
        'oof_recall': rec_or_oof,
        'oof_pass_rate': pass_or_oof,
        'oof_precision': prec_or_oof,
        'oof_auc': '-',
        'test_recall': rec_or_test,
        'test_pass_rate': pass_or_test,
        'test_precision': prec_or_test,
        'test_auc': '-',
    })
    
    # 結果表示
    print("\n" + "=" * 70)
    print("📈 結果サマリー (OOF)")
    print("=" * 70)
    
    for r in results:
        print(f"\n{r['method']}:")
        thresh_str = r['oof_threshold'] if isinstance(r['oof_threshold'], str) else f"{r['oof_threshold']:.4f}"
        print(f"   閾値: {thresh_str}")
        print(f"   Recall: {r['oof_recall']:.4f}, Pass Rate: {r['oof_pass_rate']:.2%}, Precision: {r['oof_precision']:.4f}")
    
    print("\n" + "=" * 70)
    print("📈 結果サマリー (Test)")
    print("=" * 70)
    
    for r in results:
        print(f"\n{r['method']}:")
        print(f"   Recall: {r['test_recall']:.4f}, Pass Rate: {r['test_pass_rate']:.2%}, Precision: {r['test_precision']:.4f}")
    
    # 最良手法の特定
    valid_results = [r for r in results if r['test_recall'] >= target_recall * 0.99]
    if valid_results:
        best = min(valid_results, key=lambda x: x['test_pass_rate'])
        best_method = best['method']
        best_pass_rate = best['test_pass_rate']
    else:
        best_method = "N/A"
        best_pass_rate = 1.0
    
    baseline_pass_rate = results[0]['test_pass_rate']
    improvement = baseline_pass_rate - best_pass_rate
    
    print(f"\n🎯 最良手法 (Test): {best_method}")
    print(f"🎯 ベースライン(Max)比 Pass Rate改善: {improvement:.2%}")
    
    # ========== 係数分析 ==========
    print("\n📊 メタモデル係数分析...")
    final_model = LogisticRegression(solver='lbfgs', max_iter=1000, C=1.0, class_weight='balanced', random_state=42)
    final_model.fit(X_meta_oof, y_oof)
    
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': final_model.coef_[0]
    }).sort_values('coefficient', ascending=False)
    
    print("\n   係数:")
    for _, row in coef_df.iterrows():
        print(f"      {row['feature']}: {row['coefficient']:.4f}")
    print(f"   Intercept: {final_model.intercept_[0]:.4f}")
    
    # ========== レポート生成 ==========
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "meta_model_report.md")
    
    report_content = f"""# Stage 1 ロジスティックメタモデル実験レポート

**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Target Recall**: {target_recall:.1%}
**ベースライン**: Max Probability

## OOF結果

| 手法 | 閾値 | Recall | Pass Rate | Precision | AUC |
|------|------|--------|-----------|-----------|-----|
"""
    for r in results:
        thresh_str = r['oof_threshold'] if isinstance(r['oof_threshold'], str) else f"{r['oof_threshold']:.4f}"
        auc_str = r['oof_auc'] if r['oof_auc'] == '-' else f"{r['oof_auc']:.4f}"
        report_content += f"| {r['method']} | {thresh_str} | {r['oof_recall']:.4f} | {r['oof_pass_rate']:.2%} | {r['oof_precision']:.4f} | {auc_str} |\n"
    
    report_content += f"""
## Test結果

| 手法 | Recall | Pass Rate | Precision | AUC |
|------|--------|-----------|-----------|-----|
"""
    for r in results:
        auc_str = r['test_auc'] if r['test_auc'] == '-' else f"{r['test_auc']:.4f}"
        report_content += f"| {r['method']} | {r['test_recall']:.4f} | {r['test_pass_rate']:.2%} | {r['test_precision']:.4f} | {auc_str} |\n"
    
    report_content += f"""
## 最良手法

**{best_method}** がTest Pass Rate **{best_pass_rate:.2%}** で最良。
ベースライン(Max Probability)との比較で **{improvement:.2%}** の改善。

## メタモデル係数

| 特徴量 | 係数 |
|--------|------|
"""
    for _, row in coef_df.iterrows():
        report_content += f"| {row['feature']} | {row['coefficient']:.4f} |\n"
    
    report_content += f"""
**Intercept**: {final_model.intercept_[0]:.4f}
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "meta_model_results.csv"), index=False)
    
    print(f"\n📄 レポート保存: {report_path}")
    print("\n" + "=" * 70)
    print("✅ 完了！")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_meta_model_experiment()
