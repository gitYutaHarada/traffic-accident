"""
グラフ構築モジュール
==================
kNNグラフの構築（Haversine距離）
PyTorch Geometric形式でのエッジリスト出力
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
from pathlib import Path
import joblib
from scipy.spatial import cKDTree
from sklearn.neighbors import BallTree
import torch

# ランダムシード固定
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Haversine距離の計算（km）
    """
    R = 6371  # 地球の半径 (km)
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


class GraphBuilder:
    """kNNグラフ構築クラス"""
    
    def __init__(
        self,
        k: int = 8,
        max_distance_km: float = 50.0,
        use_haversine: bool = True,
    ):
        """
        Args:
            k: kNNのk値
            max_distance_km: 最大距離制限（km）
            use_haversine: Haversine距離を使用するか
        """
        self.k = k
        self.max_distance_km = max_distance_km
        self.use_haversine = use_haversine
    
    def build_knn_graph(
        self,
        coords: np.ndarray,
        return_distances: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        kNNグラフの構築
        
        Args:
            coords: 座標配列 [N, 2] (lat, lon)
            return_distances: 距離を返すか
            
        Returns:
            edge_index: [2, num_edges]
            edge_attr: [num_edges, 1] (距離、オプション)
        """
        n_samples = len(coords)
        
        if self.use_haversine:
            # BallTreeを使用してHaversine距離でkNNを計算
            coords_rad = np.radians(coords)
            tree = BallTree(coords_rad, metric='haversine')
            
            # k+1を取得（自分自身を含む）
            distances, indices = tree.query(coords_rad, k=min(self.k + 1, n_samples))
            
            # 距離をkmに変換
            distances = distances * 6371  # 地球の半径
        else:
            # ユークリッド距離でkNN（近似）
            tree = cKDTree(coords)
            distances, indices = tree.query(coords, k=min(self.k + 1, n_samples))
        
        # エッジリストの構築
        edge_list = []
        edge_distances = []
        
        for i in range(n_samples):
            for j_idx in range(1, len(indices[i])):  # 自分自身を除く
                j = indices[i][j_idx]
                dist = distances[i][j_idx]
                
                # 距離制限
                if dist <= self.max_distance_km:
                    edge_list.append([i, j])
                    edge_distances.append(dist)
        
        if not edge_list:
            # エッジがない場合（孤立ノード）
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 1), dtype=torch.float32) if return_distances else None
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_distances, dtype=torch.float32).unsqueeze(1) if return_distances else None
        
        return edge_index, edge_attr
    
    def build_temporal_graph(
        self,
        df: pd.DataFrame,
        time_window: str = 'D',  # 'D': 日, 'W': 週
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        時系列スナップショットグラフの構築
        
        Args:
            df: データフレーム（lat, lon, date列を含む）
            time_window: 時間ウィンドウ
            
        Returns:
            graphs: {時間キー: (edge_index, edge_attr)}
        """
        graphs = {}
        
        df['time_key'] = df['date'].dt.to_period(time_window).astype(str)
        
        for time_key, group in df.groupby('time_key'):
            if len(group) < 2:
                continue
            
            coords = group[['lat', 'lon']].values
            edge_index, edge_attr = self.build_knn_graph(coords)
            
            # グローバルインデックスへのマッピング
            local_to_global = {i: idx for i, idx in enumerate(group.index)}
            
            graphs[time_key] = {
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'node_indices': group.index.tolist(),
            }
        
        return graphs
    
    def build_geohash_graph(
        self,
        df: pd.DataFrame,
        geohash_col: str = 'geohash',
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        ジオハッシュレベルでのグラフ構築
        
        Args:
            df: データフレーム
            geohash_col: ジオハッシュ列名
            
        Returns:
            edge_index: ジオハッシュ間のエッジ
            edge_attr: エッジ特徴量
            geohash_info: ジオハッシュ情報
        """
        # ジオハッシュごとの中心座標を計算
        geohash_coords = df.groupby(geohash_col).agg({
            'lat': 'mean',
            'lon': 'mean',
        }).reset_index()
        
        geohash_to_idx = {gh: i for i, gh in enumerate(geohash_coords[geohash_col])}
        
        coords = geohash_coords[['lat', 'lon']].values
        edge_index, edge_attr = self.build_knn_graph(coords)
        
        geohash_info = {
            'geohash_to_idx': geohash_to_idx,
            'idx_to_geohash': {v: k for k, v in geohash_to_idx.items()},
            'geohash_coords': geohash_coords,
        }
        
        return edge_index, edge_attr, geohash_info


def build_sample_graph(
    df: pd.DataFrame,
    k: int = 8,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    事故サンプルをノードとしたkNNグラフの構築
    
    Args:
        df: 前処理済みデータフレーム
        k: kNNのk値
        output_dir: 出力ディレクトリ
        
    Returns:
        graph_data: グラフデータ辞書
    """
    print(f"\n📊 kNNグラフ構築中 (k={k})...")
    
    builder = GraphBuilder(k=k)
    
    coords = df[['lat', 'lon']].values
    edge_index, edge_attr = builder.build_knn_graph(coords)
    
    print(f"   ノード数: {len(df):,}")
    print(f"   エッジ数: {edge_index.shape[1]:,}")
    print(f"   平均次数: {edge_index.shape[1] / len(df):.2f}")
    
    graph_data = {
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'n_nodes': len(df),
        'k': k,
    }
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(graph_data, output_dir / "graph_data.pt")
        print(f"   保存先: {output_dir / 'graph_data.pt'}")
    
    return graph_data


def build_geohash_level_graph(
    df: pd.DataFrame,
    geohash_col: str = 'geohash',
    k: int = 8,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    ジオハッシュレベルのグラフ構築
    """
    print(f"\n📊 ジオハッシュレベルグラフ構築中 (k={k})...")
    
    builder = GraphBuilder(k=k)
    edge_index, edge_attr, geohash_info = builder.build_geohash_graph(df, geohash_col)
    
    n_geohashes = len(geohash_info['geohash_to_idx'])
    print(f"   ジオハッシュ数: {n_geohashes:,}")
    print(f"   エッジ数: {edge_index.shape[1]:,}")
    
    graph_data = {
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'geohash_info': geohash_info,
        'n_nodes': n_geohashes,
        'k': k,
    }
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(graph_data, output_dir / "geohash_graph_data.pt")
        joblib.dump(geohash_info, output_dir / "geohash_info.joblib")
        print(f"   保存先: {output_dir}")
    
    return graph_data


def build_inductive_graph(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    k: int = 8,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    Inductive学習用のグラフ構築
    
    Train/Val/Testの全データを1つのグラフとして構築し、
    ノードマスクで各セットを区別する
    
    Args:
        train_df: 学習データ
        val_df: 検証データ
        test_df: テストデータ
        k: kNNのk値
        output_dir: 出力ディレクトリ
        
    Returns:
        graph_data: グラフデータ辞書（edge_index, masks含む）
    """
    print(f"\n📊 Inductive kNNグラフ構築中 (k={k})...")
    
    # 全データを結合
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    n_total = n_train + n_val + n_test
    
    print(f"   Train: {n_train:,} / Val: {n_val:,} / Test: {n_test:,}")
    print(f"   Total: {n_total:,}")
    
    # データフレームを結合
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # kNNグラフ構築
    builder = GraphBuilder(k=k)
    coords = combined_df[['lat', 'lon']].values
    edge_index, edge_attr = builder.build_knn_graph(coords)
    
    # ノードマスクを作成
    train_mask = torch.zeros(n_total, dtype=torch.bool)
    val_mask = torch.zeros(n_total, dtype=torch.bool)
    test_mask = torch.zeros(n_total, dtype=torch.bool)
    
    train_mask[:n_train] = True
    val_mask[n_train:n_train + n_val] = True
    test_mask[n_train + n_val:] = True
    
    print(f"   ノード数: {n_total:,}")
    print(f"   エッジ数: {edge_index.shape[1]:,}")
    print(f"   平均次数: {edge_index.shape[1] / n_total:.2f}")
    
    graph_data = {
        'edge_index': edge_index,
        'edge_attr': edge_attr,
        'train_mask': train_mask,
        'val_mask': val_mask,
        'test_mask': test_mask,
        'n_nodes': n_total,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'k': k,
        'combined_df': combined_df,  # 全データを保持
    }
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # combined_dfは保存しない（大きすぎる）
        save_data = {k: v for k, v in graph_data.items() if k != 'combined_df'}
        torch.save(save_data, output_dir / "inductive_graph_data.pt")
        print(f"   保存先: {output_dir / 'inductive_graph_data.pt'}")
    
    return graph_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Graph Builder")
    parser.add_argument('--data-path', type=str, default="data/spatio_temporal/preprocessed_train.parquet")
    parser.add_argument('--output-dir', type=str, default="data/spatio_temporal")
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--graph-type', type=str, choices=['sample', 'geohash'], default='geohash')
    
    args = parser.parse_args()
    
    df = pd.read_parquet(args.data_path)
    
    if args.graph_type == 'sample':
        build_sample_graph(df, k=args.k, output_dir=Path(args.output_dir))
    else:
        build_geohash_level_graph(df, k=args.k, output_dir=Path(args.output_dir))
