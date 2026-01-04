"""
Stage 1 キャリブレーション実験（Intel最適化版）
==============================================
OOF予測値に対してキャリブレーションを適用し、
閾値を再計算してPass Rateの変化を検証する。

最適化:
- Intel Extension for Scikit-learn (sklearnex) を使用
- joblib.Parallel で 5-Fold CV を並列化

入力:
    data/processed/stage1_oof_predictions.csv
    data/processed/stage1_test_predictions.csv

出力:
    results/stage1_experiments/calibration_report.md

実行方法:
    python scripts/experiment/stage1_calibration_test.py
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import recall_score, precision_score, roc_auc_score, precision_recall_curve
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')


def platt_scaling(proba_train: np.ndarray, y_train: np.ndarray, proba_test: np.ndarray) -> tuple:
    """Platt Scaling (Sigmoid) によるキャリブレーション"""
    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr.fit(proba_train.reshape(-1, 1), y_train)
    return lr.predict_proba(proba_test.reshape(-1, 1))[:, 1], lr


def isotonic_calibration(proba_train: np.ndarray, y_train: np.ndarray, proba_test: np.ndarray) -> tuple:
    """Isotonic Regression によるキャリブレーション"""
    ir = IsotonicRegression(out_of_bounds='clip')
    ir.fit(proba_train, y_train)
    return ir.predict(proba_test), ir


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


def process_fold(fold, train_idx, val_idx, prob_max_oof, y_oof):
    """1つのFoldを処理する関数（並列化用）"""
    # Platt Scaling
    platt_pred, platt_model = platt_scaling(
        prob_max_oof[train_idx], y_oof[train_idx], prob_max_oof[val_idx]
    )
    
    # Isotonic Regression
    isotonic_pred, isotonic_model = isotonic_calibration(
        prob_max_oof[train_idx], y_oof[train_idx], prob_max_oof[val_idx]
    )
    
    return {
        'fold': fold,
        'val_idx': val_idx,
        'platt_pred': platt_pred,
        'platt_model': platt_model,
        'isotonic_pred': isotonic_pred,
        'isotonic_model': isotonic_model,
    }


def run_calibration_experiment(
    oof_path: str = "data/processed/stage1_oof_predictions.csv",
    test_path: str = "data/processed/stage1_test_predictions.csv",
    target_recall: float = 0.995,
    output_dir: str = "results/stage1_experiments",
):
    """キャリブレーション実験を実行"""
    
    print("=" * 70)
    print("Stage 1 キャリブレーション実験 (Intel最適化版)")
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
    
    # Max Probability を計算
    prob_max_oof = np.maximum(prob_lgbm_oof, prob_catboost_oof)
    prob_max_test = np.maximum(prob_lgbm_test, prob_catboost_test)
    
    # 5-Fold CV（並列化）
    n_folds = 5
    random_state = 42
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    print("\n🔧 キャリブレーション中（並列処理）...")
    
    # 全Foldを並列実行
    fold_results = Parallel(n_jobs=-1, verbose=1)(
        delayed(process_fold)(fold, train_idx, val_idx, prob_max_oof, y_oof)
        for fold, (train_idx, val_idx) in enumerate(skf.split(prob_max_oof, y_oof))
    )
    
    # 結果を集約
    calibrated_max_platt_oof = np.zeros(len(y_oof))
    calibrated_max_isotonic_oof = np.zeros(len(y_oof))
    platt_models = []
    isotonic_models = []
    
    for result in sorted(fold_results, key=lambda x: x['fold']):
        val_idx = result['val_idx']
        calibrated_max_platt_oof[val_idx] = result['platt_pred']
        calibrated_max_isotonic_oof[val_idx] = result['isotonic_pred']
        platt_models.append(result['platt_model'])
        isotonic_models.append(result['isotonic_model'])
    
    # テストセットへのキャリブレーション
    calibrated_max_platt_test = np.zeros(len(y_test))
    calibrated_max_isotonic_test = np.zeros(len(y_test))
    
    for lr, ir in zip(platt_models, isotonic_models):
        calibrated_max_platt_test += lr.predict_proba(prob_max_test.reshape(-1, 1))[:, 1] / n_folds
        calibrated_max_isotonic_test += ir.predict(prob_max_test) / n_folds
    
    # ========== 結果集計 ==========
    results = []
    
    print("\n📊 閾値探索とPass Rate計算...")
    
    # 1. オリジナル
    thresh_orig, rec_orig, pass_orig, prec_orig = find_threshold_for_recall_fast(prob_max_oof, y_oof, target_recall)
    pred_test_orig = (prob_max_test >= thresh_orig).astype(int)
    rec_test_orig = recall_score(y_test, pred_test_orig)
    pass_test_orig = pred_test_orig.mean()
    prec_test_orig = precision_score(y_test, pred_test_orig) if pred_test_orig.sum() > 0 else 0
    
    results.append({
        'method': 'Original (Max Prob)',
        'oof_threshold': thresh_orig,
        'oof_recall': rec_orig,
        'oof_pass_rate': pass_orig,
        'oof_precision': prec_orig,
        'test_recall': rec_test_orig,
        'test_pass_rate': pass_test_orig,
        'test_precision': prec_test_orig,
    })
    
    # 2. Platt Scaling
    thresh_platt, rec_platt, pass_platt, prec_platt = find_threshold_for_recall_fast(calibrated_max_platt_oof, y_oof, target_recall)
    pred_test_platt = (calibrated_max_platt_test >= thresh_platt).astype(int)
    rec_test_platt = recall_score(y_test, pred_test_platt)
    pass_test_platt = pred_test_platt.mean()
    prec_test_platt = precision_score(y_test, pred_test_platt) if pred_test_platt.sum() > 0 else 0
    
    results.append({
        'method': 'Platt Scaling',
        'oof_threshold': thresh_platt,
        'oof_recall': rec_platt,
        'oof_pass_rate': pass_platt,
        'oof_precision': prec_platt,
        'test_recall': rec_test_platt,
        'test_pass_rate': pass_test_platt,
        'test_precision': prec_test_platt,
    })
    
    # 3. Isotonic Regression
    thresh_iso, rec_iso, pass_iso, prec_iso = find_threshold_for_recall_fast(calibrated_max_isotonic_oof, y_oof, target_recall)
    pred_test_iso = (calibrated_max_isotonic_test >= thresh_iso).astype(int)
    rec_test_iso = recall_score(y_test, pred_test_iso)
    pass_test_iso = pred_test_iso.mean()
    prec_test_iso = precision_score(y_test, pred_test_iso) if pred_test_iso.sum() > 0 else 0
    
    results.append({
        'method': 'Isotonic Regression',
        'oof_threshold': thresh_iso,
        'oof_recall': rec_iso,
        'oof_pass_rate': pass_iso,
        'oof_precision': prec_iso,
        'test_recall': rec_test_iso,
        'test_pass_rate': pass_test_iso,
        'test_precision': prec_test_iso,
    })
    
    # 結果表示
    print("\n" + "=" * 70)
    print("📈 結果サマリー (OOF)")
    print("=" * 70)
    
    for r in results:
        print(f"\n{r['method']}:")
        print(f"   閾値: {r['oof_threshold']:.4f}")
        print(f"   Recall: {r['oof_recall']:.4f}, Pass Rate: {r['oof_pass_rate']:.2%}, Precision: {r['oof_precision']:.4f}")
    
    print("\n" + "=" * 70)
    print("📈 結果サマリー (Test)")
    print("=" * 70)
    
    for r in results:
        print(f"\n{r['method']}:")
        print(f"   Recall: {r['test_recall']:.4f}, Pass Rate: {r['test_pass_rate']:.2%}, Precision: {r['test_precision']:.4f}")
    
    # Pass Rate改善
    oof_improvement = pass_orig - min(pass_platt, pass_iso)
    test_improvement = pass_test_orig - min(pass_test_platt, pass_test_iso)
    
    print(f"\n🎯 OOF Pass Rate改善幅: {oof_improvement:.2%}")
    print(f"🎯 Test Pass Rate改善幅: {test_improvement:.2%}")
    
    # ========== レポート生成 ==========
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "calibration_report.md")
    
    report_content = f"""# Stage 1 キャリブレーション実験レポート

**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Target Recall**: {target_recall:.1%}
**ベースライン**: Max Probability

## OOF結果

| 手法 | 閾値 | Recall | Pass Rate | Precision |
|------|------|--------|-----------|-----------|
| Original | {results[0]['oof_threshold']:.4f} | {results[0]['oof_recall']:.4f} | {results[0]['oof_pass_rate']:.2%} | {results[0]['oof_precision']:.4f} |
| Platt | {results[1]['oof_threshold']:.4f} | {results[1]['oof_recall']:.4f} | {results[1]['oof_pass_rate']:.2%} | {results[1]['oof_precision']:.4f} |
| Isotonic | {results[2]['oof_threshold']:.4f} | {results[2]['oof_recall']:.4f} | {results[2]['oof_pass_rate']:.2%} | {results[2]['oof_precision']:.4f} |

## Test結果

| 手法 | Recall | Pass Rate | Precision |
|------|--------|-----------|-----------|
| Original | {results[0]['test_recall']:.4f} | {results[0]['test_pass_rate']:.2%} | {results[0]['test_precision']:.4f} |
| Platt | {results[1]['test_recall']:.4f} | {results[1]['test_pass_rate']:.2%} | {results[1]['test_precision']:.4f} |
| Isotonic | {results[2]['test_recall']:.4f} | {results[2]['test_pass_rate']:.2%} | {results[2]['test_precision']:.4f} |

## 改善幅

- **OOF Pass Rate改善**: {oof_improvement:.2%}
- **Test Pass Rate改善**: {test_improvement:.2%}
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, "calibration_results.csv"), index=False)
    
    print(f"\n📄 レポート保存: {report_path}")
    print("\n" + "=" * 70)
    print("✅ 完了！")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    run_calibration_experiment()
