
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 日本語フォント設定
plt.rcParams['font.family'] = 'Meiryo'

def preprocess_no_leakage(df_original, n_clusters=50):
    df = df_original.copy()
    
    # 念のためここでも確認
    if '事故類型' in df.columns:
        print("Warning: '事故類型' still exists! Dropping it.")
        df = df.drop(columns=['事故類型'])
    
    print(f"前処理: エリアID (n={n_clusters}) 作成中...")
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
    
    coords = df[['lat_deg', 'lon_deg']].dropna()
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=4096, n_init=10)
    kmeans.fit(coords)
    
    df['area_id'] = -1
    valid_idx = coords.index
    df.loc[valid_idx, 'area_id'] = kmeans.predict(coords)
    df['area_id'] = df['area_id'].astype('category')
    
    # 日時分解
    print("前処理: 日時分解中...")
    df['発生日時'] = pd.to_datetime(df['発生日時'])
    df['month'] = df['発生日時'].dt.month.astype('category')
    df['day'] = df['発生日時'].dt.day.astype('category')
    df['hour'] = df['発生日時'].dt.hour.astype('category')
    
    drop_cols = ['発生日時', '地点　経度（東経）', '地点　緯度（北緯）', 'lon_deg', 'lat_deg']
    df = df.drop(columns=drop_cols)
    
    # 全変数をカテゴリカル化
    target_col = '死者数'
    for col in df.columns:
        if col != target_col and df[col].dtype.name != 'category':
             df[col] = df[col].astype('category')
             
    # 有望なクロス特徴量のみ追加 (Leakageなし)
    # 車種(A) x 速度規制(A) -> 予測可能
    print("前処理: interaction_type_speed を追加中...")
    df['interaction_type_speed'] = (df['当事者種別（当事者A）'].astype(str) + '_' + df['速度規制（指定のみ）（当事者A）'].astype(str)).astype('category')

    print(f"前処理完了: {df.shape}")
    return df

def train_and_evaluate(df):
    target_col = '死者数'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"Scale Pos Weight: {scale_pos_weight:.2f}")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = []
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = X.columns
    feature_importances['importance'] = 0
    
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
    
    print("学習を開始します (5-Fold CV)...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False)]
        
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
        prec = precision_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        acc = accuracy_score(y_val, y_pred)
        
        metrics.append([fold+1, acc, prec, rec, f1, auc])
        
        gain = model.feature_importance(importance_type='gain')
        feature_importances['importance'] += gain / 5
        
        print(f"  Fold {fold+1}: AUC={auc:.4f}, Recall={rec:.4f}")
        
    metrics_df = pd.DataFrame(metrics, columns=['Fold', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)
    
    return metrics_df, feature_importances

def save_results(metrics_df, feature_importances, output_dir='results/model_no_leakage'):
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_df.to_csv(f'{output_dir}/metrics.csv', index=False)
    metrics_df.mean().to_csv(f'{output_dir}/metrics_mean.csv')
    feature_importances.to_csv(f'{output_dir}/feature_importance.csv', index=False)
    
    print("\n交差検証結果 (平均):")
    print(metrics_df.mean())
    
    print("\nFeature Importance Top 20:")
    print(feature_importances.head(20))
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
    plt.title('Feature Importance (No Leakage)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png')
    print(f"\nSaved results to {output_dir}")

def main():
    # Leakage除去済みデータを使用
    input_file = r"c:\Users\socce\software-lab\traffic-accident\data\processed\honhyo_clean_no_leakage.csv"
    print(f"Loading data: {input_file}")
    df_raw = pd.read_csv(input_file)
    
    df_processed = preprocess_no_leakage(df_raw, n_clusters=50)
    
    metrics_df, feature_importances = train_and_evaluate(df_processed)
    
    save_results(metrics_df, feature_importances)

if __name__ == "__main__":
    main()
