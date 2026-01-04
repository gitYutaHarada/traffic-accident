"""
Hard Examples Analysis Script
=============================
目的:
1. アンサンブルモデルが大きく予測を外した「予測困難事例 (Hard Examples)」を特定する。
   - 見逃し (Hard FN): 死亡事故なのに予測確率が極端に低いケース。
   - 過剰検知 (Hard FP): 非死亡事故なのに予測確率が極端に高いケース。
2. Hard Examples と正解例 (Easy TP/TN) の特徴量分布を比較し、モデルの弱点を特定する。
3. 最もミスがひどいトップ事例を抽出し、SHAPを用いて要因を解明する。

使用方法:
    python scripts/experiment/analyze_hard_examples.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lightgbm as lgb
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from scipy.stats import ks_2samp

import warnings
warnings.filterwarnings('ignore')

# --- 設定 ---
DATA_PATH = Path("data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv")
STAGE1_OOF_PATH = Path("data/processed/stage1_oof_predictions.csv")
ENSEMBLE_OOF_PATH = Path("results/tabnet_optimized/oof_predictions.csv")
OUTPUT_DIR = Path("results/error_analysis_hard")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
STAGE1_RECALL_TARGET = 0.98

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 分析対象カラム
ANALYSIS_COLS = [
    '都道府県コード', '市区町村コード', '昼夜', '天候', '地形', '路面状態',
    '道路形状', '信号機', 'road_type', '歩車道区分', '中央分離帯施設等',
    '年齢（当事者A）', '当事者種別（当事者A）'
]

# コード辞書 (代表的なもの)
CODE_DICT = {
    '昼夜': {11: '昼-明', 12: '昼-昼', 13: '昼-暮', 21: '夜-暗', 22: '夜-道路照明あり', 23: '夜-道路照明なし'},
    '天候': {1: '晴', 2: '曇', 3: '雨', 4: '霧', 5: '雪', 6: 'その他'},
    '地形': {1: '市街地', 2: '非市街地-DID外', 3: 'その他'},
    '道路形状': {1: '交差点', 11: '単路-トンネル', 21: '踏切', 31: '交差点-環状'},
}


def load_and_align_data():
    """データ読み込みと紐付け"""
    print("📂 データ読み込み・紐付け中...")
    df_full = pd.read_csv(DATA_PATH)
    df_full['fatal'] = df_full['fatal'].astype(int)
    
    stage1_oof = pd.read_csv(STAGE1_OOF_PATH)
    ensemble_oof = pd.read_csv(ENSEMBLE_OOF_PATH)
    
    all_indices = np.arange(len(df_full))
    train_indices, _ = train_test_split(
        all_indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df_full['fatal']
    )
    
    y_train = df_full.iloc[train_indices]['fatal'].values
    stage1_prob = 0.85 * stage1_oof['prob_catboost'].values + 0.15 * stage1_oof['prob_lgbm'].values
    
    precision, recall, thresholds = precision_recall_curve(y_train, stage1_prob)
    valid_idx = np.where(recall[:-1] >= STAGE1_RECALL_TARGET)[0]
    stage1_threshold = thresholds[valid_idx[-1]] if len(valid_idx) > 0 else 0.0
    
    filter_mask = stage1_prob >= stage1_threshold
    filtered_train_indices = train_indices[filter_mask]
    
    if len(filtered_train_indices) != len(ensemble_oof):
        print(f"   ⚠️ 行数不一致: train indices = {len(filtered_train_indices)}, oof = {len(ensemble_oof)}")
        # 最小値に合わせる（継続のため）
        min_len = min(len(filtered_train_indices), len(ensemble_oof))
        filtered_train_indices = filtered_train_indices[:min_len]
        ensemble_oof = ensemble_oof.iloc[:min_len]
        
    df_aligned = df_full.iloc[filtered_train_indices].reset_index(drop=True).copy()
    for col in ensemble_oof.columns:
        df_aligned[f'pred_{col}'] = ensemble_oof[col].values
        
    return df_aligned, stage1_threshold


def identify_hard_examples(df, threshold):
    """Hard Examples (見逃し/過剰検知) を特定する"""
    print("\n🎯 Hard Examples 特定中...")
    
    y_true = df['fatal'].values
    y_prob = df['pred_ensemble'].values
    y_pred = (y_prob >= threshold).astype(int)
    
    # 基本マスク
    tp_mask = (y_true == 1) & (y_pred == 1)  # True Positive
    tn_mask = (y_true == 0) & (y_pred == 0)  # True Negative
    fp_mask = (y_true == 0) & (y_pred == 1)  # False Positive
    fn_mask = (y_true == 1) & (y_pred == 0)  # False Negative (見逃し)
    
    # Hard Examples の抽出
    # Hard FN: 死亡事故の中で予測確率が特に低いもの (下位10%)
    fatal_probs = y_prob[y_true == 1]
    fn_threshold = np.percentile(fatal_probs, 10)  # 下位10%
    hard_fn_mask = fn_mask & (y_prob < fn_threshold)
    
    # Hard FP: 非死亡事故の中で予測確率が特に高いもの (上位1%)
    non_fatal_probs = y_prob[y_true == 0]
    fp_threshold = np.percentile(non_fatal_probs, 99)  # 上位1%
    hard_fp_mask = fp_mask & (y_prob > fp_threshold)
    
    # Easy Examples (比較用)
    # Easy TP: 死亡事故で予測確率が高いもの (上位50%)
    tp_threshold = np.percentile(fatal_probs, 50)
    easy_tp_mask = tp_mask & (y_prob > tp_threshold)
    
    # Easy TN: 非死亡事故で予測確率が低いもの (下位50%)
    tn_threshold = np.percentile(non_fatal_probs, 50)
    easy_tn_mask = tn_mask & (y_prob < tn_threshold)
    
    summary = {
        'Hard FN (見逃し)': hard_fn_mask.sum(),
        'Hard FP (過剰検知)': hard_fp_mask.sum(),
        'Easy TP (正解)': easy_tp_mask.sum(),
        'Easy TN (正解)': easy_tn_mask.sum(),
        'fn_threshold': fn_threshold,
        'fp_threshold': fp_threshold,
    }
    
    print(f"   Hard FN (見逃し): {hard_fn_mask.sum()} 件 (prob < {fn_threshold:.4f})")
    print(f"   Hard FP (過剰検知): {hard_fp_mask.sum()} 件 (prob > {fp_threshold:.4f})")
    print(f"   Easy TP (比較用): {easy_tp_mask.sum()} 件")
    print(f"   Easy TN (比較用): {easy_tn_mask.sum()} 件")
    
    return {
        'hard_fn': hard_fn_mask,
        'hard_fp': hard_fp_mask,
        'easy_tp': easy_tp_mask,
        'easy_tn': easy_tn_mask,
        'fn': fn_mask,
        'fp': fp_mask,
        'summary': summary
    }


def compare_distributions(df, mask1, mask2, label1, label2, output_dir):
    """2群間の特徴量分布をKS検定で比較する"""
    print(f"\n📊 分布比較: {label1} vs {label2}")
    
    available_cols = [c for c in ANALYSIS_COLS if c in df.columns]
    
    results = []
    for col in available_cols:
        data1 = df.loc[mask1, col].dropna()
        data2 = df.loc[mask2, col].dropna()
        
        if len(data1) < 10 or len(data2) < 10:
            continue
            
        # KS検定
        stat, p_value = ks_2samp(data1, data2)
        
        # 代表値の差
        mean_diff = data1.mean() - data2.mean()
        
        results.append({
            'feature': col,
            'ks_stat': stat,
            'ks_pvalue': p_value,
            'mean_diff': mean_diff,
            'n1': len(data1),
            'n2': len(data2)
        })
    
    df_results = pd.DataFrame(results).sort_values('ks_stat', ascending=False)
    
    # 上位5件を表示
    print(f"   分布乖離が大きい特徴量 (Top 5):")
    for i, row in df_results.head(5).iterrows():
        print(f"      - {row['feature']}: KS={row['ks_stat']:.4f}, p={row['ks_pvalue']:.4e}")
    
    # CSVに保存
    output_path = output_dir / f"distribution_comparison_{label1}_vs_{label2}.csv"
    df_results.to_csv(output_path, index=False)
    
    return df_results


def visualize_hard_examples_distributions(df, masks, output_dir):
    """Hard Examples の代表的な特徴量分布を可視化"""
    print("\n📈 Hard Examples の分布可視化...")
    
    # 可視化対象の特徴量
    target_cols = ['昼夜', '地形', '道路形状', '天候']
    target_cols = [c for c in target_cols if c in df.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(target_cols):
        ax = axes[idx]
        
        # 各カテゴリの割合を計算
        categories = []
        for mask_name, mask in [('Hard FN', masks['hard_fn']), 
                                 ('Easy TP', masks['easy_tp']),
                                 ('Hard FP', masks['hard_fp']),
                                 ('Easy TN', masks['easy_tn'])]:
            counts = df.loc[mask, col].value_counts(normalize=True).head(10)
            for val, pct in counts.items():
                # コード辞書があれば変換
                val_label = CODE_DICT.get(col, {}).get(val, str(val))
                categories.append({'group': mask_name, 'value': val_label, 'percentage': pct})
        
        df_cat = pd.DataFrame(categories)
        
        # Grouped bar chart
        if not df_cat.empty:
            pivot = df_cat.pivot_table(index='value', columns='group', values='percentage', aggfunc='first')
            pivot.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title(f'{col} 分布比較')
            ax.set_ylabel('割合')
            ax.legend(title='', fontsize=8)
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "hard_examples_distributions.png", dpi=150)
    plt.close()
    print(f"   保存: hard_examples_distributions.png")


def deep_dive_top_cases(df, masks, output_dir, n_cases=3):
    """Top N の極端な誤分類事例を深掘り (SHAP分析)"""
    print(f"\n🔬 Top {n_cases} 極端事例の深掘り (SHAP分析)...")
    
    y_prob = df['pred_ensemble'].values
    
    # LightGBMプロキシモデルを学習 (SHAP計算用)
    print("   LightGBMプロキシモデルを学習中...")
    
    excluded_cols = ['fatal', 'pred_lgbm', 'pred_catboost', 'pred_mlp', 
                     'pred_tabnet_optimized', 'pred_ensemble', 'target', 'accident_id',
                     '地点　緯度（北緯）', '地点　経度（東経）']
    feature_cols = [c for c in df.columns if c not in excluded_cols and not c.startswith('pred_')]
    
    # カテゴリ変数
    cat_cols = [c for c in ANALYSIS_COLS if c in feature_cols]
    
    X = df[feature_cols].copy()
    y = df['fatal']
    
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype('category')
    
    lgb_train = lgb.Dataset(X, y)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'seed': 42,
        'num_leaves': 31,
        'learning_rate': 0.1
    }
    model = lgb.train(params, lgb_train, num_boost_round=100)
    
    explainer = shap.TreeExplainer(model)
    
    # --- Hard FN Top Cases ---
    hard_fn_indices = np.where(masks['hard_fn'])[0]
    if len(hard_fn_indices) > 0:
        # 予測確率が最も低いものを選択
        fn_probs = y_prob[hard_fn_indices]
        sorted_idx = np.argsort(fn_probs)[:n_cases]
        top_fn_indices = hard_fn_indices[sorted_idx]
        
        print(f"\n   === Hard FN Top {n_cases} Cases (最もひどい見逃し) ===")
        
        for rank, idx in enumerate(top_fn_indices, 1):
            prob = y_prob[idx]
            print(f"\n   Case #{rank}: Index={idx}, Prob={prob:.4f} (死亡事故なのに低確率)")
            
            # 特徴量サマリー
            for col in ['都道府県コード', '市区町村コード', '道路形状', '昼夜', '地形']:
                if col in df.columns:
                    val = df.loc[idx, col]
                    val_label = CODE_DICT.get(col, {}).get(val, val)
                    print(f"      - {col}: {val_label}")
            
            # SHAP Waterfall
            shap_values = explainer.shap_values(X.iloc[[idx]])
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            base_value = explainer.expected_value
            if isinstance(base_value, list):
                base_value = base_value[1]
            
            plt.figure(figsize=(10, 6))
            try:
                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=base_value,
                        data=X.iloc[idx],
                        feature_names=X.columns
                    ),
                    show=False, max_display=10
                )
                plt.title(f"Hard FN Case #{rank}: 見逃し要因分解 (Prob={prob:.4f})")
                plt.tight_layout()
                plt.savefig(output_dir / f"shap_waterfall_hard_fn_{rank}.png", dpi=150)
                plt.close()
            except Exception as e:
                print(f"      ⚠️ SHAP Waterfall生成エラー: {e}")
    
    # --- Hard FP Top Cases ---
    hard_fp_indices = np.where(masks['hard_fp'])[0]
    if len(hard_fp_indices) > 0:
        fp_probs = y_prob[hard_fp_indices]
        sorted_idx = np.argsort(-fp_probs)[:n_cases]  # 降順
        top_fp_indices = hard_fp_indices[sorted_idx]
        
        print(f"\n   === Hard FP Top {n_cases} Cases (最もひどい過剰検知) ===")
        
        for rank, idx in enumerate(top_fp_indices, 1):
            prob = y_prob[idx]
            print(f"\n   Case #{rank}: Index={idx}, Prob={prob:.4f} (非死亡なのに高確率)")
            
            for col in ['都道府県コード', '市区町村コード', '道路形状', '昼夜', '地形']:
                if col in df.columns:
                    val = df.loc[idx, col]
                    val_label = CODE_DICT.get(col, {}).get(val, val)
                    print(f"      - {col}: {val_label}")
            
            shap_values = explainer.shap_values(X.iloc[[idx]])
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            base_value = explainer.expected_value
            if isinstance(base_value, list):
                base_value = base_value[1]
            
            plt.figure(figsize=(10, 6))
            try:
                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_values[0],
                        base_values=base_value,
                        data=X.iloc[idx],
                        feature_names=X.columns
                    ),
                    show=False, max_display=10
                )
                plt.title(f"Hard FP Case #{rank}: 過剰検知要因分解 (Prob={prob:.4f})")
                plt.tight_layout()
                plt.savefig(output_dir / f"shap_waterfall_hard_fp_{rank}.png", dpi=150)
                plt.close()
            except Exception as e:
                print(f"      ⚠️ SHAP Waterfall生成エラー: {e}")
    
    print(f"\n   SHAP Waterfallプロット保存完了")


def generate_summary_report(df, masks, dist_fn, dist_fp, output_dir):
    """サマリーレポート (Markdown) を生成"""
    print("\n📝 サマリーレポート生成中...")
    
    summary = masks['summary']
    
    report = f"""# Hard Examples 分析レポート

## 概要

このレポートは、アンサンブルモデルが予測を大きく外した「Hard Examples (予測困難事例)」の分析結果をまとめたものです。

### 抽出された Hard Examples

| カテゴリ | 件数 | 閾値 |
|---------|------|------|
| **見逃し (Hard FN)** | {summary['Hard FN (見逃し)']} 件 | prob < {summary['fn_threshold']:.4f} |
| **過剰検知 (Hard FP)** | {summary['Hard FP (過剰検知)']} 件 | prob > {summary['fp_threshold']:.4f} |
| 比較用 Easy TP | {summary['Easy TP (正解)']} 件 | - |
| 比較用 Easy TN | {summary['Easy TN (正解)']} 件 | - |

---

## 見逃し (Hard FN) の特徴

Hard FN と Easy TP を比較した結果、以下の特徴量で分布が大きく異なることが判明しました。

"""
    
    if dist_fn is not None and len(dist_fn) > 0:
        report += "| 特徴量 | KS統計量 | p値 | 平均差 |\n|--------|----------|-----|--------|\n"
        for _, row in dist_fn.head(5).iterrows():
            report += f"| {row['feature']} | {row['ks_stat']:.4f} | {row['ks_pvalue']:.2e} | {row['mean_diff']:.2f} |\n"
    else:
        report += "*データ不足のため比較できませんでした。*\n"
    
    report += f"""

---

## 過剰検知 (Hard FP) の特徴

Hard FP と Easy TN を比較した結果、以下の特徴量で分布が大きく異なることが判明しました。

"""
    
    if dist_fp is not None and len(dist_fp) > 0:
        report += "| 特徴量 | KS統計量 | p値 | 平均差 |\n|--------|----------|-----|--------|\n"
        for _, row in dist_fp.head(5).iterrows():
            report += f"| {row['feature']} | {row['ks_stat']:.4f} | {row['ks_pvalue']:.2e} | {row['mean_diff']:.2f} |\n"
    else:
        report += "*データ不足のため比較できませんでした。*\n"
    
    report += f"""

---

## 個別事例の深掘り (SHAP分析)

最も誤分類がひどかったTop 3事例について、SHAPを用いて要因分析を行いました。
詳細は以下のファイルを参照してください:

- `shap_waterfall_hard_fn_1.png`, `shap_waterfall_hard_fn_2.png`, `shap_waterfall_hard_fn_3.png` (見逃し)
- `shap_waterfall_hard_fp_1.png`, `shap_waterfall_hard_fp_2.png`, `shap_waterfall_hard_fp_3.png` (過剰検知)

---

## 結論と改善提案

1. **見逃しの傾向**: Hard FN は特定の条件下で発生しやすい可能性があります。上記の分布比較結果を参考に、モデルへの特徴量追加や重み調整を検討してください。
2. **過剰検知の傾向**: Hard FP は「危険に見えるが実際は死亡に至らなかった事故」を拾っている可能性があります。閾値の調整やフィルタリングロジックの追加を検討してください。
3. **次のステップ**: 特定の条件（例：特定の道路形状、地形）に特化した専門家モデル (Mixture of Experts) の導入を検討する価値があります。

"""
    
    # 保存
    report_path = output_dir / "hard_examples_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # results/md にもコピー
    md_dir = Path("results/md")
    os.makedirs(md_dir, exist_ok=True)
    with open(md_dir / "hard_examples_analysis.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"   保存: {report_path}")
    print(f"   保存: {md_dir / 'hard_examples_analysis.md'}")


def main():
    print("=" * 70)
    print(" 🔍 Hard Examples (予測困難事例) 分析")
    print("=" * 70)
    
    df, threshold = load_and_align_data()
    
    # 1. Hard Examples の特定
    masks = identify_hard_examples(df, threshold)
    
    # 2. 分布比較
    dist_fn = compare_distributions(
        df, masks['hard_fn'], masks['easy_tp'], 
        'Hard_FN', 'Easy_TP', OUTPUT_DIR
    )
    
    dist_fp = compare_distributions(
        df, masks['hard_fp'], masks['easy_tn'], 
        'Hard_FP', 'Easy_TN', OUTPUT_DIR
    )
    
    # 3. 分布可視化
    visualize_hard_examples_distributions(df, masks, OUTPUT_DIR)
    
    # 4. Top事例の深掘り (SHAP)
    deep_dive_top_cases(df, masks, OUTPUT_DIR, n_cases=3)
    
    # 5. レポート生成
    generate_summary_report(df, masks, dist_fn, dist_fp, OUTPUT_DIR)
    
    print("\n✅ Hard Examples 分析完了")
    print(f"   出力先: {OUTPUT_DIR}")
    print(f"   レポート: results/md/hard_examples_analysis.md")


if __name__ == "__main__":
    main()
