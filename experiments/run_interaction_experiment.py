"""
Interaction Features 実験パイプライン

1. 特徴量生成 (create_interaction_features.py を呼び出し)
2. LightGBM モデル学習 (新特徴量を含む)
3. Feature Importance 分析 (gain/split)
4. SHAP 分析
5. TP vs FP 分布比較 (効果測定)
"""

import pandas as pd
import numpy as np
import os
import sys
import gc
import matplotlib.pyplot as plt
import matplotlib as mpl
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, roc_auc_score, log_loss
import shap
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
mpl.rcParams['font.family'] = 'MS Gothic'
mpl.rcParams['axes.unicode_minus'] = False

# パス設定
# experiments/run_interaction_experiment.py -> traffic-accident/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "experiments", "interaction_features")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===== Step 1: 特徴量生成 =====
def create_interaction_features():
    """既存スクリプトを呼び出して特徴量を生成"""
    print("=" * 60)
    print("Step 1: Creating Interaction Features")
    print("=" * 60)
    
    input_path = os.path.join(DATA_DIR, "honhyo_clean_with_features.csv")
    output_path = os.path.join(DATA_DIR, "honhyo_with_interactions.csv")
    
    # 既に存在する場合はスキップ
    if os.path.exists(output_path):
        print(f"  {output_path} already exists. Loading...")
        return output_path
    
    # スクリプトをsubprocessで実行
    import subprocess
    script_path = os.path.join(BASE_DIR, "scripts", "features", "create_interaction_features.py")
    result = subprocess.run(
        ["python", script_path, "--input", input_path, "--output", output_path],
        capture_output=True, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        raise RuntimeError("Feature creation failed")
    
    return output_path


# ===== Step 2: モデル学習 =====
def train_lightgbm_cv(df: pd.DataFrame, target_col: str = '死者数', n_folds: int = 5):
    """LightGBM CV学習とOOF予測"""
    print("\n" + "=" * 60)
    print("Step 2: Training LightGBM with Interaction Features")
    print("=" * 60)
    
    # 特徴量とターゲット
    drop_cols = [target_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].copy()
    y = (df[target_col] > 0).astype(int)
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target distribution: {y.value_counts().to_dict()}")
    
    # カテゴリカル変数の特定
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category')
    print(f"  Categorical features: {len(cat_cols)}")
    
    # LightGBM パラメータ（解毒剤特徴量が使われるように深い木を許容）
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,  # 学習率を下げてより多くのイテレーションを促す
        'num_leaves': 63,  # 深い木で相互作用を捉える
        'max_depth': -1,
        'min_child_samples': 10,  # より細かいパターンを許容
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1,
        'is_unbalance': True
    }
    
    # CV
    oof_proba = np.zeros(len(df))
    feature_importance_gain = np.zeros(len(feature_cols))
    feature_importance_split = np.zeros(len(feature_cols))
    models = []
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n  Fold {fold + 1}/{n_folds}...")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=cat_cols)
        
        # early_stopping を緩和し、最低100ラウンドは学習させる
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,  # 増加
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, first_metric_only=True),  # 100ラウンド待機
                lgb.log_evaluation(100)
            ]
        )
        
        oof_proba[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        feature_importance_gain += model.feature_importance(importance_type='gain')
        feature_importance_split += model.feature_importance(importance_type='split')
        models.append(model)
        
        gc.collect()
    
    # 平均化
    feature_importance_gain /= n_folds
    feature_importance_split /= n_folds
    
    # OOF評価
    auc = roc_auc_score(y, oof_proba)
    logloss = log_loss(y, oof_proba)
    print(f"\n  OOF AUC: {auc:.4f}")
    print(f"  OOF LogLoss: {logloss:.4f}")
    
    # Recall 99% での閾値と Precision
    precision, recall, thresholds = precision_recall_curve(y, oof_proba)
    idx_99 = np.argmax(recall <= 0.99)
    threshold_99 = thresholds[idx_99] if idx_99 < len(thresholds) else thresholds[-1]
    precision_at_99 = precision[idx_99]
    print(f"  Threshold for 99% Recall: {threshold_99:.4f}")
    print(f"  Precision at 99% Recall: {precision_at_99:.4f}")
    
    results = {
        'oof_proba': oof_proba,
        'y_true': y.values,
        'feature_cols': feature_cols,
        'feature_importance_gain': feature_importance_gain,
        'feature_importance_split': feature_importance_split,
        'models': models,
        'auc': auc,
        'logloss': logloss,
        'threshold_99': threshold_99,
        'precision_at_99': precision_at_99,
        'X': X,
        'cat_cols': cat_cols
    }
    
    return results

# ===== Step 3: Feature Importance 分析 =====
def analyze_feature_importance(results: dict):
    """Feature Importance (gain/split) の分析と可視化"""
    print("\n" + "=" * 60)
    print("Step 3: Feature Importance Analysis")
    print("=" * 60)
    
    feature_cols = results['feature_cols']
    gain = results['feature_importance_gain']
    split = results['feature_importance_split']
    
    # DataFrame作成
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Gain': gain,
        'Split': split
    }).sort_values('Gain', ascending=False)
    
    # Top 20 表示
    print("\n  Top 20 Features by Gain:")
    for i, row in importance_df.head(20).iterrows():
        print(f"    {row['Feature']}: Gain={row['Gain']:.2f}, Split={row['Split']:.0f}")
    
    # 新特徴量のチェック
    new_features = ['stop_sign_interaction', 'speed_reg_diff', 'speed_reg_diff_abs', 
                    'maybe_vulnerable_victim', 'night_terrain', 'road_shape_terrain',
                    'signal_road_shape', 'night_road_condition', 'speed_shape_interaction',
                    'party_type_daytime', 'party_type_road_shape',
                    'is_safe_night_urban', 'midnight_activity_flag', 'intersection_with_signal']
    
    print("\n  New Interaction Features Performance:")
    for feat in new_features:
        if feat in importance_df['Feature'].values:
            row = importance_df[importance_df['Feature'] == feat].iloc[0]
            rank = importance_df[importance_df['Feature'] == feat].index[0] + 1
            print(f"    {feat}: Rank={rank}, Gain={row['Gain']:.2f}, Split={row['Split']:.0f}")
        else:
            print(f"    {feat}: NOT FOUND")
    
    # プロット
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    # Gain
    top_gain = importance_df.head(30)
    colors = ['#e74c3c' if f in new_features else '#3498db' for f in top_gain['Feature']]
    axes[0].barh(range(len(top_gain)), top_gain['Gain'], color=colors)
    axes[0].set_yticks(range(len(top_gain)))
    axes[0].set_yticklabels(top_gain['Feature'])
    axes[0].invert_yaxis()
    axes[0].set_title('Feature Importance (Gain) - Top 30\n赤: 新特徴量')
    axes[0].set_xlabel('Gain')
    
    # Split
    top_split = importance_df.sort_values('Split', ascending=False).head(30)
    colors = ['#e74c3c' if f in new_features else '#3498db' for f in top_split['Feature']]
    axes[1].barh(range(len(top_split)), top_split['Split'], color=colors)
    axes[1].set_yticks(range(len(top_split)))
    axes[1].set_yticklabels(top_split['Feature'])
    axes[1].invert_yaxis()
    axes[1].set_title('Feature Importance (Split) - Top 30\n赤: 新特徴量')
    axes[1].set_xlabel('Split Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_importance.png'), dpi=150)
    plt.close()
    
    # CSV保存
    importance_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance.csv'), index=False)
    print(f"\n  Saved feature importance to {RESULTS_DIR}")
    
    return importance_df

# ===== Step 4: SHAP分析 =====
def analyze_shap(results: dict, sample_size: int = 5000):
    """SHAP分析"""
    print("\n" + "=" * 60)
    print("Step 4: SHAP Analysis")
    print("=" * 60)
    
    model = results['models'][0]  # 最初のfoldのモデルを使用
    X = results['X']
    
    # サンプリング（計算時間短縮）
    if len(X) > sample_size:
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[sample_idx]
    else:
        X_sample = X
    
    print(f"  Computing SHAP values for {len(X_sample)} samples...")
    
    # SHAP計算
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # binaryの場合、正クラスのSHAP値を使用
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Summary Plot
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=30)
    plt.title('SHAP Feature Importance (Top 30)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'shap_importance.png'), dpi=150)
    plt.close()
    
    # Beeswarm Plot (詳細な影響方向)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title('SHAP Summary Plot (Top 20)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'shap_summary.png'), dpi=150)
    plt.close()
    
    # 新特徴量のSHAP値平均
    new_features = ['stop_sign_interaction', 'speed_reg_diff', 'speed_reg_diff_abs', 
                    'maybe_vulnerable_victim', 'night_terrain', 'road_shape_terrain',
                    'signal_road_shape', 'night_road_condition', 'speed_shape_interaction',
                    'party_type_daytime', 'party_type_road_shape',
                    'is_safe_night_urban', 'midnight_activity_flag', 'intersection_with_signal']
    
    print("\n  New Features SHAP Impact (mean |SHAP|):")
    feature_cols = results['feature_cols']
    for feat in new_features:
        if feat in feature_cols:
            idx = feature_cols.index(feat)
            mean_shap = np.abs(shap_values[:, idx]).mean()
            print(f"    {feat}: {mean_shap:.4f}")
    
    print(f"\n  Saved SHAP plots to {RESULTS_DIR}")
    
    return shap_values

# ===== Step 5: TP vs FP 分布比較 =====
def compare_tp_fp_distribution(results: dict, df: pd.DataFrame):
    """TP vs FP の分布比較"""
    print("\n" + "=" * 60)
    print("Step 5: TP vs FP Distribution Comparison")
    print("=" * 60)
    
    oof_proba = results['oof_proba']
    y_true = results['y_true']
    threshold = results['threshold_99']
    
    # 予測ラベル
    pred_label = (oof_proba >= threshold).astype(int)
    
    # TP/FP抽出
    tp_mask = (pred_label == 1) & (y_true == 1)
    fp_mask = (pred_label == 1) & (y_true == 0)
    
    n_tp = tp_mask.sum()
    n_fp = fp_mask.sum()
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
    
    print(f"  Threshold: {threshold:.4f}")
    print(f"  True Positives: {n_tp}")
    print(f"  False Positives: {n_fp}")
    print(f"  Precision: {precision:.4f}")
    
    # 新特徴量の分布比較
    new_features = ['stop_sign_interaction', 'speed_reg_diff', 'speed_reg_diff_abs', 
                    'maybe_vulnerable_victim', 'night_terrain', 'is_safe_night_urban', 
                    'midnight_activity_flag', 'intersection_with_signal']
    
    print("\n  New Features Distribution (TP vs FP):")
    
    comparison_results = []
    for feat in new_features:
        if feat in df.columns:
            tp_vals = df.loc[tp_mask, feat]
            fp_vals = df.loc[fp_mask, feat]
            
            # 数値の場合
            if df[feat].dtype in ['int64', 'float64']:
                tp_mean = tp_vals.mean()
                fp_mean = fp_vals.mean()
                diff = abs(tp_mean - fp_mean)
                print(f"    {feat}: TP mean={tp_mean:.3f}, FP mean={fp_mean:.3f}, Diff={diff:.3f}")
                comparison_results.append({'Feature': feat, 'TP_mean': tp_mean, 'FP_mean': fp_mean, 'Diff': diff})
            else:
                # カテゴリの場合はモード比較
                tp_mode = tp_vals.mode()[0] if len(tp_vals.mode()) > 0 else None
                fp_mode = fp_vals.mode()[0] if len(fp_vals.mode()) > 0 else None
                print(f"    {feat}: TP mode={tp_mode}, FP mode={fp_mode}")
    
    # TP/FP データを保存
    df_with_pred = df.copy()
    df_with_pred['oof_proba'] = oof_proba
    df_with_pred['pred_label'] = pred_label
    df_with_pred['true_label'] = y_true
    
    df_with_pred[tp_mask].to_csv(os.path.join(RESULTS_DIR, 'tp_new_model.csv'), index=False)
    df_with_pred[fp_mask].to_csv(os.path.join(RESULTS_DIR, 'fp_new_model.csv'), index=False)
    
    print(f"\n  Saved TP/FP data to {RESULTS_DIR}")
    
    return {
        'n_tp': n_tp,
        'n_fp': n_fp,
        'precision': precision,
        'threshold': threshold
    }

# ===== メイン実行 =====
def main():
    print("=" * 60)
    print("Interaction Features Experiment Pipeline")
    print("=" * 60)
    
    # Step 1: 特徴量生成
    data_path = create_interaction_features()
    
    # データ読み込み
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # Step 2: モデル学習
    results = train_lightgbm_cv(df)
    
    # Step 3: Feature Importance
    importance_df = analyze_feature_importance(results)
    
    # Step 4: SHAP
    shap_values = analyze_shap(results)
    
    # Step 5: TP vs FP
    tp_fp_results = compare_tp_fp_distribution(results, df)
    
    # 最終レポート
    print("\n" + "=" * 60)
    print("Experiment Summary")
    print("=" * 60)
    print(f"  OOF AUC: {results['auc']:.4f}")
    print(f"  OOF LogLoss: {results['logloss']:.4f}")
    print(f"  Precision @ 99% Recall: {results['precision_at_99']:.4f}")
    print(f"  TP: {tp_fp_results['n_tp']}, FP: {tp_fp_results['n_fp']}")
    print(f"\n  Results saved to: {RESULTS_DIR}")
    
    # レポートファイル生成
    report_path = os.path.join(RESULTS_DIR, 'experiment_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Interaction Features 実験レポート\n\n")
        f.write("## 実験概要\n")
        f.write(f"- **データ**: {data_path}\n")
        f.write(f"- **特徴量数**: {len(results['feature_cols'])}\n\n")
        
        f.write("## モデル性能\n")
        f.write(f"- **OOF AUC**: {results['auc']:.4f}\n")
        f.write(f"- **OOF LogLoss**: {results['logloss']:.4f}\n")
        f.write(f"- **Precision @ 99% Recall**: {results['precision_at_99']:.4f}\n\n")
        
        f.write("## TP/FP 分析\n")
        f.write(f"- **閾値 (99% Recall)**: {tp_fp_results['threshold']:.4f}\n")
        f.write(f"- **True Positives**: {tp_fp_results['n_tp']}\n")
        f.write(f"- **False Positives**: {tp_fp_results['n_fp']}\n")
        f.write(f"- **Precision**: {tp_fp_results['precision']:.4f}\n\n")
        
        f.write("## Feature Importance (Top 10)\n")
        f.write("| Rank | Feature | Gain | Split |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for i, row in importance_df.head(10).iterrows():
            f.write(f"| {i+1} | {row['Feature']} | {row['Gain']:.2f} | {row['Split']:.0f} |\n")
        
        f.write("\n## 可視化\n")
        f.write("![Feature Importance](feature_importance.png)\n")
        f.write("![SHAP Summary](shap_summary.png)\n")
    
    print(f"\n  Report saved to: {report_path}")

if __name__ == "__main__":
    main()
