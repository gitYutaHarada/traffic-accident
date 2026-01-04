"""
Deep Dive Error Analysis Script
===============================
目的: 
1. `analyze_ensemble_errors.py` で特定された高リスク要因（踏切、ワイヤロープ等）を深堀りする。
2. 決定木を用いて、エラー（FP/FN）が発生する複合条件（ルール）を自動抽出する。
3. 地理空間情報をプロットし、地域的な偏りを可視化する。

注意:
- 決定木分析では、カテゴリ変数をOne-Hot Encodingして学習させることで、
  「都道府県コード <= 20」のような無意味な数値分割を防ぎ、
  「踏切の有無」のような明確なルールを抽出する。
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings('ignore')

# --- 設定 ---
DATA_PATH = Path("data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv")
STAGE1_OOF_PATH = Path("data/processed/stage1_oof_predictions.csv")
ENSEMBLE_OOF_PATH = Path("results/tabnet_optimized/oof_predictions.csv")
OUTPUT_DIR = Path("results/error_analysis_deep_dive")
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
    '道路形状', '信号機', '事故類型', '曜日(発生年月日)', '時', '月',
    '歩車道区分', '中央分離帯施設等', 'road_type'
]

# コード定義（判明分）
CODE_DEFINITIONS = {
    '道路形状': {
        21: '踏切-第一種',
        11: '単路-トンネル',
        31: '交差点-環状',
        1: '交差点-その他'
    },
    '中央分離帯施設等': {
        7: '中央線-ワイヤロープ',
        1: '中央分離帯',
        5: '分離なし'
    }
}


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
    
    if len(filtered_train_indices) != len(ensemble_oof):
        raise ValueError("行数不一致")
        
    df_aligned = df_full.iloc[filtered_train_indices].reset_index(drop=True).copy()
    for col in ensemble_oof.columns:
        df_aligned[f'pred_{col}'] = ensemble_oof[col].values
        
    return df_aligned, stage1_threshold


def analyze_high_risk_segments(df, fp_mask, fn_mask, output_dir):
    """高リスクセグメント（踏切、ワイヤロープ等）の詳細分析"""
    print("\n🔍 高リスクセグメント詳細分析...")
    
    segments = [
        {'col': '道路形状', 'val': 21, 'name': '踏切(第一種)'},
        {'col': '中央分離帯施設等', 'val': 7, 'name': 'ワイヤロープ'},
        {'col': '市区町村コード', 'val': 483, 'name': '市区町村483'},
    ]
    
    segment_stats = []
    
    for seg in segments:
        col, val, name = seg['col'], seg['val'], seg['name']
        mask = df[col] == val
        
        if mask.sum() == 0:
            continue
            
        n_total = mask.sum()
        fp_rate = (mask & fp_mask).sum() / mask.sum()
        fn_rate = (mask & fn_mask).sum() / mask.sum()
        fatal_rate = df.loc[mask, 'fatal'].mean()
        
        # 関連する他の特徴量の統計（例：踏切事故は昼が多い？夜が多い？）
        night_rate = (df.loc[mask, '昼夜'] > 20).mean() # 21,22,23は夜
        rain_snow_rate = (df.loc[mask, '天候'].isin([3, 5])).mean() # 3=雨, 5=雪
        
        stats = {
            'segment': name,
            'total_count': n_total,
            'fp_rate': fp_rate,
            'fn_rate': fn_rate,
            'fatal_rate': fatal_rate,
            'night_ratio': night_rate,
            'bad_weather_ratio': rain_snow_rate
        }
        segment_stats.append(stats)
        
    df_stats = pd.DataFrame(segment_stats)
    print(df_stats)
    df_stats.to_csv(output_dir / "high_risk_segment_profiles.csv", index=False)
    return df_stats


def extract_error_rules_with_decision_tree(df, target_mask, target_name, output_dir):
    """
    決定木を用いてエラー発生ルールを抽出する
    One-Hot Encodingを使用することで、「もし踏切なら...」といった明確なルールを生成
    """
    print(f"\n🌲 決定木によるルール抽出 ({target_name})...")
    
    # ターゲット: 指定されたエラータイプか (1) そうでないか (0)
    y = target_mask.astype(int)
    
    # 特徴量: カテゴリ変数をOne-Hot化
    # 存在するカラムのみを使用
    available_cols = [c for c in ANALYSIS_COLS if c in df.columns]
    
    # マッピング: 日本語名がない場合、英語名を試す
    name_mapping = {'時': 'hour', '月': 'month', '曜日(発生年月日)': 'day_of_week'}
    for jp, en in name_mapping.items():
        if jp not in df.columns and en in df.columns:
            available_cols.append(en)
            if jp in available_cols: available_cols.remove(jp) # 重複除去
            
    # 重複除去
    available_cols = list(set(available_cols))
    
    print(f"   使用する特徴量: {len(available_cols)} 個")
    X_raw = df[available_cols].fillna(-1)
    
    # 数値として扱うべきカラム（時、月など）
    # 英語名も含める
    num_cols_candidates = ['時', '月', 'hour', 'month']
    num_cols = [c for c in num_cols_candidates if c in available_cols]
    cat_cols = [c for c in available_cols if c not in num_cols]
    
    # One-Hot Encoding
    if len(cat_cols) > 0:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_cat = encoder.fit_transform(X_raw[cat_cols])
        feature_names_cat = encoder.get_feature_names_out(cat_cols)
    else:
        X_cat = np.empty((len(df), 0))
        feature_names_cat = []
    
    X_num = X_raw[num_cols].values
    feature_names = list(feature_names_cat) + num_cols
    
    X = np.hstack([X_cat, X_num])
    
    # 決定木学習 (深すぎると解釈不能なので深さ3〜4に制限)
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=42, class_weight='balanced')
    clf.fit(X, y)
    
    # 可視化
    plt.figure(figsize=(20, 10))
    plot_tree(clf, feature_names=feature_names, class_names=['Correct', 'Error'], 
              filled=True, fontsize=10, proportion=True)
    plt.title(f"{target_name} 発生ルール (決定木)")
    plt.savefig(output_dir / f"tree_rules_{target_name}.png", dpi=150)
    plt.close()
    
    # テキスト形式でルールを出力
    rules = export_text(clf, feature_names=feature_names)
    with open(output_dir / f"rules_{target_name}.txt", "w", encoding="utf-8") as f:
        f.write(rules)
        
    print(f"   保存: tree_rules_{target_name}.png, rules_{target_name}.txt")


def plot_geospatial_errors(df, fp_mask, fn_mask, output_dir):
    """地理空間プロット (日本地図)"""
    print("\n🗺️ 地理空間エラープロット...")
    
    if '地点　経度（東経）' not in df.columns or '地点　緯度（北緯）' not in df.columns:
        print("   ⚠️ 緯度経度データがないためスキップします")
        return

    # DMS (度分秒) -> 度 (Decimal) 変換が必要な場合があるが、
    # ここではデータが既に変換済みであるか、または簡易的にそのままプロットして確認する
    # ※ 本データセットは通常このままプロット可能
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 全データ（背景）
    # データ量が多すぎるのでサンプリング
    df_sample = df.sample(frac=0.1, random_state=42)
    ax.scatter(df_sample['地点　経度（東経）'], df_sample['地点　緯度（北緯）'], 
               c='lightgray', s=1, alpha=0.5, label='Others')
    
    # FP (赤)
    ax.scatter(df.loc[fp_mask, '地点　経度（東経）'], df.loc[fp_mask, '地点　緯度（北緯）'], 
               c='red', s=5, alpha=0.6, label='False Positive')
    
    # FN (オレンジ)
    ax.scatter(df.loc[fn_mask, '地点　経度（東経）'], df.loc[fn_mask, '地点　緯度（北緯）'], 
               c='orange', s=5, alpha=0.6, label='False Negative')
    
    # 特定の高リスク市区町村 (483) をハイライト
    high_risk_city = df['市区町村コード'] == 483
    if high_risk_city.sum() > 0:
        ax.scatter(df.loc[high_risk_city, '地点　経度（東経）'], df.loc[high_risk_city, '地点　緯度（北緯）'], 
                   c='blue', s=20, marker='x', label='City 483')

    ax.set_title('エラー発生地点の地理分布')
    ax.set_xlabel('経度')
    ax.set_ylabel('緯度')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "geospatial_error_map.png", dpi=150)
    plt.close()
    print(f"   保存: geospatial_error_map.png")


def main():
    print("=" * 70)
    print(" 🌊 Deep Dive Error Analysis")
    print("=" * 70)
    
    df, threshold = load_and_align_data()
    
    # FP/FN マスク作成
    y_true = df['fatal'].values
    y_prob = df['pred_ensemble'].values
    y_pred = (y_prob >= threshold).astype(int)
    
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    
    # 1. 高リスクセグメント詳細分析
    analyze_high_risk_segments(df, fp_mask, fn_mask, OUTPUT_DIR)
    
    # 2. 決定木によるルール抽出 (One-Hot Encoded)
    extract_error_rules_with_decision_tree(df, fp_mask, "False_Positive", OUTPUT_DIR)
    extract_error_rules_with_decision_tree(df, fn_mask, "False_Negative", OUTPUT_DIR)
    
    # 3. 地理空間プロット
    plot_geospatial_errors(df, fp_mask, fn_mask, OUTPUT_DIR)
    
    print("\n✅ 深層分析完了")
    print(f"   出力先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
