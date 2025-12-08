
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import gc

# 日本語フォント設定
plt.rcParams['font.family'] = 'Meiryo'

def preprocess_data_with_clustering(df_original, n_clusters):
    """
    データフレームのコピーを受け取り、指定されたn_clustersでエリアIDを付与する
    """
    df = df_original.copy()
    
    # 緯度経度の変換（簡易ロジック）
    # 前回と同様のロジックを使用
    lon_val = df['地点　経度（東経）'].fillna(0).astype(np.int64)
    lat_val = df['地点　緯度（北緯）'].fillna(0).astype(np.int64)
    
    def convert_vectorized(v):
        v = np.where(v == 0, np.nan, v)
        ms = v % 1000
        v = v // 1000
        ss = v % 100
        v = v // 100
        mm = v % 100
        dd = v // 100
        return dd + mm/60 + (ss + ms/1000)/3600

    df['lon_deg'] = convert_vectorized(lon_val)
    df['lat_deg'] = convert_vectorized(lat_val)
    
    # クラスタリング
    print(f"  Clustering with n_clusters={n_clusters}...")
    coords = df[['lat_deg', 'lon_deg']].dropna()
    
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=4096, n_init=10)
    kmeans.fit(coords)
    
    df['area_id'] = -1
    valid_idx = coords.index
    df.loc[valid_idx, 'area_id'] = kmeans.predict(coords)
    df['area_id'] = df['area_id'].astype('category')
    
    # 日時分解
    df['発生日時'] = pd.to_datetime(df['発生日時'])
    df['month'] = df['発生日時'].dt.month.astype('category')
    df['day'] = df['発生日時'].dt.day.astype('category')
    df['hour'] = df['発生日時'].dt.hour.astype('category')
    
    # 不要カラム削除
    drop_cols = ['発生日時', '地点　経度（東経）', '地点　緯度（北緯）', 'lon_deg', 'lat_deg']
    df = df.drop(columns=drop_cols)
    
    # カテゴリカル化
    target_col = '死者数'
    for col in df.columns:
        if col != target_col and df[col].dtype.name != 'category':
             df[col] = df[col].astype('category')
             
    return df

def train_lgbm(df):
    target_col = '死者数'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = neg_count / pos_count
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    recall_scores = []
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'scale_pos_weight': scale_pos_weight,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False)]
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            num_boost_round=1000,
            callbacks=callbacks
        )
        
        y_pred_prob = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred = (y_pred_prob >= 0.5).astype(int)
        
        auc = roc_auc_score(y_val, y_pred_prob)
        rec = recall_score(y_val, y_pred)
        
        auc_scores.append(auc)
        recall_scores.append(rec)
    
    return np.mean(auc_scores), np.mean(recall_scores)

def main():
    input_file = r"c:\Users\socce\software-lab\traffic-accident\data\processed\honhyo_clean_predictable_only.csv"
    output_dir = 'results/experiments/area_granularity_low'
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data: {input_file}")
    df_raw = pd.read_csv(input_file)
    
    n_clusters_list = [10, 50, 100, 200, 300, 400]
    results = []
    
    print(f"Starting experiment for n_clusters: {n_clusters_list}")
    
    for n in n_clusters_list:
        print(f"\nProcessing n_clusters={n}...")
        
        # 前処理
        df_processed = preprocess_data_with_clustering(df_raw, n)
        
        # 学習
        print(f"  Training LightGBM...")
        mean_auc, mean_recall = train_lgbm(df_processed)
        
        print(f"  Result: AUC={mean_auc:.4f}, Recall={mean_recall:.4f}")
        results.append({
            'n_clusters': n,
            'mean_auc': mean_auc,
            'mean_recall': mean_recall
        })
        
        # メモリ解放
        del df_processed
        gc.collect()
        
    # 結果保存
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'summary.csv')
    results_df.to_csv(results_path, index=False)
    
    print("\nExperiment Results:")
    print(results_df)
    
    # プロット
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['n_clusters'], results_df['mean_auc'], marker='o', linewidth=2, label='AUC')
    plt.title('Effect of Area Granularity (n_clusters) on Model Performance')
    plt.xlabel('Number of Clusters (Area IDs)')
    plt.ylabel('Mean AUC Score')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'auc_comparison.png'))
    
    # Recallプロット
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['n_clusters'], results_df['mean_recall'], marker='s', color='orange', linewidth=2, label='Recall')
    plt.title('Effect of Area Granularity (n_clusters) on Recall')
    plt.xlabel('Number of Clusters (Area IDs)')
    plt.ylabel('Mean Recall Score')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'recall_comparison.png'))

    print(f"\nSaved results to {output_dir}")

if __name__ == "__main__":
    main()
