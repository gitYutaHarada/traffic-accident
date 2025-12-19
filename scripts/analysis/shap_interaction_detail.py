"""
Phase 1: SHAP Interaction Detail Analysis

相互作用特徴量（party_type_daytime, road_shape_terrain 等）について、
どのカテゴリ組み合わせがリスクを高めているかを SHAP 値を用いて詳細分析する。

Output:
- Risk Matrix Heatmap (各組み合わせの平均SHAP値)
- High Risk Groups CSV (危険度が高い組み合わせのランキング)
"""

import pandas as pd
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import lightgbm as lgb
import shap
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
mpl.rcParams['font.family'] = 'MS Gothic'
mpl.rcParams['axes.unicode_minus'] = False

# パス設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "analysis", "shap_detail")
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_model_for_shap(df: pd.DataFrame, target_col: str = '死者数'):
    """SHAP分析用にモデルを学習（1 Foldのみ）"""
    print("Training model for SHAP analysis...")
    
    drop_cols = [target_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols].copy()
    y = (df[target_col] > 0).astype(int)
    
    # カテゴリカル変数の処理
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category')
    
    # Train/Val split (80/20)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(skf.split(X, y))
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 63,  # 分析用に深めに設定（相互作用を捉えやすくする）
        'min_data_in_leaf': 10,  # より細かいパターンを許容
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1,
        'is_unbalance': True
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=cat_cols)
    
    model = lgb.train(
        params, train_data, num_boost_round=300,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    print(f"Model trained. Best iteration: {model.best_iteration}")
    return model, X, y, feature_cols, cat_cols


def compute_shap_values(model, X: pd.DataFrame, sample_size: int = 10000):
    """SHAP値を計算"""
    print(f"Computing SHAP values for {min(len(X), sample_size)} samples...")
    
    if len(X) > sample_size:
        sample_idx = np.random.choice(len(X), sample_size, replace=False)
        X_sample = X.iloc[sample_idx].copy()
    else:
        X_sample = X.copy()
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # Binary classification の場合、正クラスを使用
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    return shap_values, X_sample


def analyze_interaction_feature(feature_name: str, shap_values: np.ndarray, 
                                 X_sample: pd.DataFrame, feature_cols: list):
    """単一の相互作用特徴量を分析し、リスクマトリックスを作成"""
    print(f"\nAnalyzing: {feature_name}")
    
    if feature_name not in feature_cols:
        print(f"  Feature {feature_name} not found. Skipping.")
        return None
    
    feat_idx = feature_cols.index(feature_name)
    feat_shap = shap_values[:, feat_idx]
    feat_values = X_sample[feature_name].values
    
    # カテゴリごとの平均SHAP値を計算
    df_shap = pd.DataFrame({
        'category': feat_values,
        'shap_value': feat_shap
    })
    
    category_shap = df_shap.groupby('category').agg(
        mean_shap=('shap_value', 'mean'),
        count=('shap_value', 'count'),
        std_shap=('shap_value', 'std')
    ).reset_index()
    
    category_shap = category_shap.sort_values('mean_shap', ascending=False)
    
    print(f"  Top 5 high-risk categories:")
    for _, row in category_shap.head(5).iterrows():
        print(f"    {row['category']}: SHAP={row['mean_shap']:.4f} (n={row['count']})")
    
    return category_shap


def create_risk_matrix(feature_name: str, category_shap: pd.DataFrame):
    """
    相互作用特徴量を元の構成要素に分解し、Risk Matrixを作成する。
    例: 'party_type_daytime' -> 縦軸=party_type, 横軸=daytime
    """
    # 特徴量名から構成要素を推測
    if '_' not in feature_name:
        print(f"  Cannot decompose {feature_name} into matrix.")
        return None
    
    # 各カテゴリ値を分解 (例: "1_2" -> element1=1, element2=2)
    try:
        splits = category_shap['category'].astype(str).str.split('_', expand=True)
        if splits.shape[1] < 2:
            print(f"  Cannot split {feature_name} categories into 2 components.")
            return None
        
        category_shap['elem1'] = splits[0]
        category_shap['elem2'] = splits[1]
        
        # 軸のソート順を正しくするために数値化を試みる
        try:
            category_shap['elem1'] = pd.to_numeric(category_shap['elem1'])
        except (ValueError, TypeError):
            pass  # 数値変換できない場合は文字列のまま
            
        try:
            category_shap['elem2'] = pd.to_numeric(category_shap['elem2'])
        except (ValueError, TypeError):
            pass  # 数値変換できない場合は文字列のまま
            
    except Exception as e:
        print(f"  Error splitting categories: {e}")
        return None
    
    # ピボットテーブルでマトリックス作成
    matrix = category_shap.pivot_table(
        index='elem1', columns='elem2', values='mean_shap', aggfunc='mean'
    )
    
    # ヒートマップ描画
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn_r', center=0,
                linewidths=0.5, cbar_kws={'label': 'Mean SHAP Value (Risk)'})
    
    # ラベル設定（特徴量名から推測）
    if 'party_type_daytime' in feature_name:
        plt.xlabel('昼夜コード')
        plt.ylabel('当事者種別コード')
    elif 'road_shape_terrain' in feature_name:
        plt.xlabel('地形コード')
        plt.ylabel('道路形状コード')
    elif 'night_terrain' in feature_name:
        plt.xlabel('地形コード')
        plt.ylabel('昼夜コード')
    elif 'signal_road_shape' in feature_name:
        plt.xlabel('道路形状コード')
        plt.ylabel('信号機コード')
    else:
        plt.xlabel('Component 2')
        plt.ylabel('Component 1')
    
    plt.title(f'Risk Matrix: {feature_name}\n(赤=危険、緑=安全)')
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, f'risk_matrix_{feature_name}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")
    
    return matrix


def main():
    print("=" * 60)
    print("Phase 1: SHAP Interaction Detail Analysis")
    print("=" * 60)
    
    # データ読み込み
    data_path = os.path.join(DATA_DIR, "honhyo_with_interactions.csv")
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Data shape: {df.shape}")
    
    # モデル学習
    model, X, y, feature_cols, cat_cols = train_model_for_shap(df)
    
    # SHAP計算
    shap_values, X_sample = compute_shap_values(model, X, sample_size=10000)
    
    # 分析対象の相互作用特徴量
    interaction_features = [
        'party_type_daytime',
        'road_shape_terrain',
        'night_terrain',
        'signal_road_shape',
        'night_road_condition',
        'speed_shape_interaction'
    ]
    
    # 結果格納
    all_results = []
    
    for feat in interaction_features:
        category_shap = analyze_interaction_feature(feat, shap_values, X_sample, feature_cols)
        if category_shap is not None:
            category_shap['feature'] = feat
            all_results.append(category_shap)
            
            # Risk Matrix作成
            create_risk_matrix(feat, category_shap)
    
    # 全結果をCSV保存
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined = combined.sort_values('mean_shap', ascending=False)
        save_path = os.path.join(RESULTS_DIR, 'high_risk_groups.csv')
        combined.to_csv(save_path, index=False)
        print(f"\nSaved all results to: {save_path}")
    
    # レポート生成
    report_path = os.path.join(RESULTS_DIR, 'shap_detail_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# SHAP Interaction Detail Report\n\n")
        f.write("## 分析対象特徴量\n")
        for feat in interaction_features:
            f.write(f"- {feat}\n")
        
        f.write("\n## 高リスク組み合わせ (Top 10)\n")
        if all_results:
            top10 = combined.head(10)
            f.write("| Feature | Category | Mean SHAP | Count |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            for _, row in top10.iterrows():
                f.write(f"| {row['feature']} | {row['category']} | {row['mean_shap']:.4f} | {row['count']} |\n")
        
        f.write("\n## Risk Matrix 可視化\n")
        for feat in interaction_features:
            img_path = f"risk_matrix_{feat}.png"
            if os.path.exists(os.path.join(RESULTS_DIR, img_path)):
                f.write(f"### {feat}\n")
                f.write(f"![{feat}]({img_path})\n\n")
    
    print(f"\nReport saved to: {report_path}")
    print("\nPhase 1 Complete!")


if __name__ == "__main__":
    main()
