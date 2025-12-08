
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MiniBatchKMeans
import matplotlib.pyplot as plt
import os
import gc

# 日本語フォント設定
plt.rcParams['font.family'] = 'Meiryo'

def preprocess_for_dt(input_file, n_clusters=50):
    print(f"Loading data: {input_file}")
    df = pd.read_csv(input_file)
    
    # 緯度経度の変換 & クラスタリング (n=50)
    print(f"Generating Area ID (n={n_clusters})...")
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
    
    # 日時分解
    print("Processing datetime...")
    df['発生日時'] = pd.to_datetime(df['発生日時'])
    df['month'] = df['発生日時'].dt.month
    df['day'] = df['発生日時'].dt.day
    df['hour'] = df['発生日時'].dt.hour
    
    # 不要カラム削除
    drop_cols = ['発生日時', '地点　経度（東経）', '地点　緯度（北緯）', 'lon_deg', 'lat_deg']
    df = df.drop(columns=drop_cols)
    
    # Label Encoding (決定木用)
    print("Label Encoding...")
    target_col = '死者数'
    encoders = {}
    
    for col in df.columns:
        if col != target_col:
            # fillna with 'Missing' or -1 before encoding
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                df[col] = df[col].astype(str).fillna('Missing')
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                encoders[col] = le
            else:
                df[col] = df[col].fillna(-1)
                
    return df, encoders

def analyze_tree_structure(model, feature_names):
    n_nodes = model.tree_.node_count
    children_left = model.tree_.children_left
    children_right = model.tree_.children_right
    feature = model.tree_.feature
    threshold = model.tree_.threshold
    
    # 親子関係の探索
    # ノードiの親を見つける
    parents = np.zeros(n_nodes, dtype=int) - 1
    for i in range(n_nodes):
        if children_left[i] != -1:
            parents[children_left[i]] = i
        if children_right[i] != -1:
            parents[children_right[i]] = i
            
    interaction_counts = {}
    
    # 全ノード走査（ルート以外）
    for i in range(1, n_nodes):
        parent_idx = parents[i]
        
        # 葉ノードまたは条件分岐
        # このノード自体が分割に使われている場合、親の分割変数とのペアを見る
        if feature[i] != -2: # not leaf? no, feature is -2 for leaf usually in sklearn cython implementation but let's check
            # sklearn: feature[i] == _tree.TREE_UNDEFINED (-2) for leaves
            pass
            
        # 親の分割変数は？
        if feature[parent_idx] != -2:
            parent_feat = feature_names[feature[parent_idx]]
            
            # 自分が分割に使われているなら
            if feature[i] != -2:
                child_feat = feature_names[feature[i]]
                
                # 同じ変数の連続分割は除外（例: 年齢>20 AND 年齢<50）
                if parent_feat != child_feat:
                    pair = tuple(sorted([parent_feat, child_feat]))
                    if pair not in interaction_counts:
                        interaction_counts[pair] = 0
                    
                    # 重要度を加味するなら gain を足すが、簡易的に頻度＋深さで重み付け
                    # 浅い階層（ルートに近い）ほど重要
                    # ここでは単純出現回数として、重要な分岐ほど上位に来ると仮定
                    interaction_counts[pair] += 1

    # 重要度ベース（Feature Importance）で選ぶ方が良いかもしれないが
    # 決定木の構造（Interaction）を見るなら、「Aで分岐した後にBで分岐した」事実が重要
    
    return interaction_counts

def main():
    input_file = r"c:\Users\socce\software-lab\traffic-accident\data\processed\honhyo_clean_predictable_only.csv"
    output_dir = 'results/analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # 前処理
    df_encoded, encoders = preprocess_for_dt(input_file, n_clusters=50)
    
    target_col = '死者数'
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    feature_names = X.columns.tolist()
    
    print("\nTraining Decision Tree (max_depth=4)...")
    # クラス不均衡なので balanced を指定、深さは浅めに
    dt = DecisionTreeClassifier(max_depth=4, class_weight='balanced', random_state=42)
    dt.fit(X, y)
    
    # 木の可視化
    print("Plotting tree...")
    plt.figure(figsize=(20, 10))
    plot_tree(dt, feature_names=feature_names, filled=True, rounded=True, fontsize=10, max_depth=3)
    plt.savefig(os.path.join(output_dir, 'dt_visualization.png'))
    plt.close()
    
    # Interaction抽出
    print("Extracting interactions...")
    interactions = analyze_tree_structure(dt, feature_names)
    
    # 結果整形
    sorted_interactions = sorted(interactions.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*50)
    print("Top Interaction Candidates (Parent-Child Splits)")
    print("="*50)
    
    results = []
    for pair, count in sorted_interactions:
        print(f"{pair[0]} x {pair[1]} (Count: {count})")
        results.append({'Feature_1': pair[0], 'Feature_2': pair[1], 'Count': count})
        
    pd.DataFrame(results).to_csv(os.path.join(output_dir, 'interaction_candidates.csv'), index=False)
    print(f"\nSaved results to {output_dir}")

if __name__ == "__main__":
    main()
