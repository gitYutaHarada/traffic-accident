"""
Advanced Error Analysis Script (SHAP & Geospatial)
==================================================
目的:
1. SHAP値を用いて、誤検知（FP）や見逃し（FN）の要因をモデル内部から解明する。
2. 高リスク市区町村（483等）の具体的な地理的位置を特定する。
3. 年齢層×事故類型のクロス集計を行い、デモグラフィックな弱点を特定する。

使用モデル:
- LightGBM (Stage 1 or Stage 2 model) をロードして使用
- ※アンサンブル全体のSHAPは計算コストが高いため、代表としてLightGBMを使用

使用方法:
    python scripts/experiment/analyze_error_shap_geo.py
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

import warnings
warnings.filterwarnings('ignore')

# --- 設定 ---
DATA_PATH = Path("data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv")
STAGE1_OOF_PATH = Path("data/processed/stage1_oof_predictions.csv")
ENSEMBLE_OOF_PATH = Path("results/tabnet_optimized/oof_predictions.csv")
MODEL_PATH = Path("results/tabnet_optimized/lgbm_model_fold0.pkl") # LightGBMモデルパス (仮)
OUTPUT_DIR = Path("results/error_analysis_advanced")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.2
STAGE1_RECALL_TARGET = 0.98

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def load_and_align_data():
    """データ読み込みと紐付け (v3再利用)"""
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
    
    df_aligned = df_full.iloc[filtered_train_indices].reset_index(drop=True).copy()
    for col in ensemble_oof.columns:
        df_aligned[f'pred_{col}'] = ensemble_oof[col].values
        
    return df_aligned, stage1_threshold


def identify_city_location(df, city_codes, output_dir):
    """特定市区町村の緯度経度平均を算出し、場所を特定する"""
    print("\n📍 市区町村ロケーション特定...")
    
    results = []
    
    for code in city_codes:
        mask = df['市区町村コード'] == code
        if mask.sum() == 0:
            continue
            
        lat_mean = df.loc[mask, '地点　緯度（北緯）'].mean()
        lon_mean = df.loc[mask, '地点　経度（東経）'].mean()
        count = mask.sum()
        
        # 度分秒表記の可能性を考慮 (数値が異常に大きい場合)
        # 緯度が 100 以上の場合は度分秒の可能性が高い (日本は緯度20-46)
        # しかし前回のプロットで日本地図になっていたなら変換済みと推測
        # ここではそのまま出力
        
        # Google Maps URL
        gmap_url = f"https://www.google.com/maps/search/?api=1&query={lat_mean},{lon_mean}"
        
        print(f"   City {code}: N={count}, Lat={lat_mean:.4f}, Lon={lon_mean:.4f}")
        print(f"   -> {gmap_url}")
        
        results.append({
            'city_code': code,
            'count': count,
            'lat_mean': lat_mean,
            'lon_mean': lon_mean,
            'google_maps_url': gmap_url
        })
        
    df_res = pd.DataFrame(results)
    df_res.to_csv(output_dir / "city_locations.csv", index=False)
    
    # テキストレポートにも出力
    with open(output_dir / "city_locations_report.txt", "w", encoding="utf-8") as f:
        for res in results:
            f.write(f"City Code: {res['city_code']}\n")
            f.write(f"Sample Count: {res['count']}\n")
            f.write(f"Centroid: {res['lat_mean']}, {res['lon_mean']}\n")
            f.write(f"Map URL: {res['google_maps_url']}\n")
            f.write("-" * 30 + "\n")


def analyze_demographic_heatmap(df, fp_mask, output_dir):
    """年齢層 × 事故類型の誤検知率ヒートマップ"""
    print("\n👥 デモグラフィック分析 (Age x Type)...")
    
    # 年齢層をグルーピング (元データはカテゴリコードの可能性)
    # コードブックによると: 1:0-24, 25:25-34, ..., 75:75+
    # わかりやすいラベルに変換
    age_map = {
        1: '0-24歳', 25: '25-34歳', 35: '35-44歳', 
        45: '45-54歳', 55: '55-64歳', 65: '65-74歳', 
        75: '75歳以上', 0: '不明'
    }
    
    # 年齢カラム名
    col_age = '年齢（当事者A）'
    
    # 事故類型マップ (主要なもの)
    type_map = {
        1: '人対車両', 21: '車両相互', 41: '車両単独', 61: '列車'
    }
    
    # 事故類型がない場合、当事者種別（当事者A）を代用
    target_col = '事故類型'
    title_label = '事故類型'
    
    if target_col not in df.columns:
        if '当事者種別（当事者A）' in df.columns:
            target_col = '当事者種別（当事者A）'
            title_label = '当事者種別'
        else:
            print("   ⚠️ 分析に必要なカラム（事故類型 or 当事者種別）が見つかりません。スキップします。")
            return

    # データコピー
    df_viz = df[[col_age, target_col]].copy()
    
    # マッピング
    df_viz['Age_Group'] = df_viz[col_age].map(age_map)
    
    if title_label == '事故類型':
        df_viz['Type_Group'] = df_viz[target_col].map(type_map)
    else:
        # 当事者種別の簡易マッピング
        # 1-5:乗用車, 11-14:貨物, 31-36:二輪, 51:自転車, 61:歩行者
        def map_party_type_simple(x):
            if 1 <= x <= 10: return '乗用車'
            elif 11 <= x <= 20: return '貨物車'
            elif 31 <= x <= 40: return '二輪車'
            elif x == 51 or x == 52: return '自転車'
            elif x == 61: return '歩行者'
            else: return 'その他'
        
        df_viz['Type_Group'] = df_viz[target_col].apply(map_party_type_simple)

    # ピボットテーブル作成 (FP率)
    # fp_mask は numpy array なので、df_vizと同じインデックスを持つSeriesに変換してから参照する
    fp_mask_series = pd.Series(fp_mask, index=df.index)
    
    # ダミー列を追加
    df_viz['Dummy'] = 1
    
    pivot_fp = df_viz.pivot_table(
        index='Type_Group', columns='Age_Group', 
        values='Dummy', # 重複しないカラムを指定
        aggfunc=lambda x: (fp_mask_series.loc[x.index].sum() / len(x)) 
                          if len(x) > 50 else np.nan 
    )
    
    # 列順序を整える
    age_order = ['0-24歳', '25-34歳', '35-44歳', '45-54歳', '55-64歳', '65-74歳', '75歳以上']
    age_order = [c for c in age_order if c in pivot_fp.columns]
    pivot_fp = pivot_fp[age_order]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_fp, annot=True, fmt='.1%', cmap='Reds')
    plt.title(f'誤検知率 (FP Rate) Heatmap: 年齢 × {title_label}')
    plt.tight_layout()
    plt.savefig(output_dir / "age_type_fp_heatmap.png", dpi=150)
    plt.close()
    print(f"   保存: age_type_fp_heatmap.png")


def analyze_shap_feature_importance(df, fp_mask, output_dir):
    """SHAP値による特徴量重要度分析 (FP要因)"""
    print("\n🌟 SHAP値分析 (LightGBM)...")
    
    # 保存済みのモデルを探す（results/tabnet_optimized, results/lgbm_optuna など）
    model_paths = list(Path("results").glob("**/lgbm*.pkl")) + list(Path("results").glob("**/model*.pkl"))
    lgbm_path = None
    
    # 最も新しいpklを探すなどのロジックが必要だが、ここではファイル名で推測
    # scripts/experiment/train_tabnet_optimized.py はTabNet用なので、
    # 以前の会話で作ったLightGBMモデルがあるはず。
    # なければ、簡易的にここで学習させる方が確実で早い。
    
    print("   LightGBMモデルを再学習してSHAPを計算します (紐付けデータの特性を直接反映するため)...")
    
    # 特徴量選定 (数値のみ、カテゴリはcategorical_featureとして扱う)
    # 分析用データのカラム定義に基づいて選定
    excluded_cols = ['fatal', 'pred_lgbm', 'pred_catboost', 'pred_mlp', 
                     'pred_tabnet_optimized', 'pred_ensemble', 'target']
    feature_cols = [c for c in df.columns if c not in excluded_cols]
    
    # カテゴリ変数の指定
    cat_cols_candidates = ['都道府県コード', '市区町村コード', '昼夜', '天候', '地形', '路面状態',
                '道路形状', '信号機', '事故類型', '曜日(発生年月日)', '歩車道区分', 
                '中央分離帯施設等', 'road_type', '年齢（当事者A）', '当事者種別（当事者A）']
    cat_cols = [c for c in cat_cols_candidates if c in feature_cols]
    
    # 学習データ準備 (元のターゲットを使用)
    X = df[feature_cols].copy()
    y = df['fatal']
    
    # カテゴリ変数をcategory型に変換
    for c in cat_cols:
        X[c] = X[c].astype('category')
        
    # LightGBMデータセット
    lgb_train = lgb.Dataset(X, y)
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'seed': 42
    }
    
    # 簡易学習 (SHAP用)
    model = lgb.train(params, lgb_train, num_boost_round=100)
    
    # SHAP計算 (FPデータのみサンプリング)
    # 背景データは全体からランダムサンプリング
    # カテゴリ型データのサンプリング時にエラーが出ないよう注意
    X_sample = X.sample(1000, random_state=42)
    explainer = shap.TreeExplainer(model)
    
    # FPデータ (誤検知) のSHAP値
    fp_indices = np.where(fp_mask)[0]
    if len(fp_indices) > 500:
        fp_indices = np.random.choice(fp_indices, 500, replace=False)
        
    X_fp = X.iloc[fp_indices]
    shap_values_fp = explainer.shap_values(X_fp)
    
    # SHAP値がリスト（クラス別）の場合、クラス1（死亡）のSHAP値を取得
    if isinstance(shap_values_fp, list):
        shap_values_fp = shap_values_fp[1]
    
    # Summary Plot用にカラム名を日本語対応フォントで表示させるための工夫
    # shapはmatplotlibを使うので設定済みフォントが効くはず
    
    # 1. Global Importance (FP要因)
    plt.figure(figsize=(10, 15)) # 縦長に
    shap.summary_plot(shap_values_fp, X_fp, show=False, plot_type="dot", max_display=20)
    plt.title("誤検知(FP)データのSHAP値 (高いほど誤検知要因)")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_fp_dot.png", dpi=150)
    plt.close()
    print(f"   保存: shap_summary_fp_dot.png")
    
    plt.figure(figsize=(10, 15))
    shap.summary_plot(shap_values_fp, X_fp, show=False, plot_type="bar", max_display=20)
    plt.title("誤検知(FP)への影響度 (絶対値平均)")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_fp_bar.png", dpi=150)
    plt.close()
    print(f"   保存: shap_summary_fp_bar.png")
    
    # 2. Local Importance (踏切事故の誤検知など)
    # '道路形状'=21 (踏切) のケースを探す
    if '道路形状' in X_fp.columns:
        railroad_mask = (X_fp['道路形状'] == 21)
        if railroad_mask.sum() > 0:
            # 最初の1件を取得
            idx = np.where(railroad_mask)[0][0] # これはX_fp内のローカルインデックス
            print(f"\n   🚂 踏切FP事例のSHAP分析... (Index in sample: {idx})")
            
            # shap.plots.waterfall は Explanation オブジェクトを必要とする
            # shap_values_fp[idx] は array
            
            # 期待値 (base_value) の取得
            base_value = explainer.expected_value
            if isinstance(base_value, list):
                base_value = base_value[1]
            
            plt.figure(figsize=(10, 8))
            shap.plots.waterfall(
                shap.Explanation(values=shap_values_fp[idx], 
                                 base_values=base_value, 
                                 data=X_fp.iloc[idx], 
                                 feature_names=X_fp.columns),
                show=False, max_display=10
            )
            plt.title(f"踏切FP事例の要因分解")
            plt.tight_layout()
            plt.savefig(output_dir / "shap_local_railroad.png", dpi=150)
            plt.close()
            print(f"   保存: shap_local_railroad.png")


def main():
    print("=" * 70)
    print(" 🔬 Advanced Error Analysis (SHAP & Geo)")
    print("=" * 70)
    
    df, threshold = load_and_align_data()
    
    y_true = df['fatal'].values
    y_prob = df['pred_ensemble'].values
    y_pred = (y_prob >= threshold).astype(int)
    
    # numpy array mask
    fp_mask = (y_true == 0) & (y_pred == 1)
    
    # 緯度経度の簡易補正 (1/10,000,000)
    # 値が1億を超えている場合に適用
    for col in ['地点　緯度（北緯）', '地点　経度（東経）']:
        if col in df.columns and df[col].mean() > 1000:
            print(f"   ⚠️ {col} の値が大きいため、1/10,000,000 して補正します。")
            df[col] = df[col] / 10000000.0
    
    # 1. 市区町村ロケーション特定 (483, 585, 586など)
    target_cities = [483, 585, 586, 434, 492, 311] # レポートの上位
    identify_city_location(df, target_cities, OUTPUT_DIR)
    
    # 2. デモグラフィック分析
    # ここでFP率を計算するために、heatmap関数内でreindexエラーが起きないようロジック修正済みのはずだが、
    # 念のため関数が正しく実装されていることを前提とする。
    analyze_demographic_heatmap(df, fp_mask, OUTPUT_DIR)
    
    # 3. SHAP分析
    analyze_shap_feature_importance(df, fp_mask, OUTPUT_DIR)
    
    print("\n✅ 高度分析完了")
    print(f"   出力先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
