"""
kNN-Graph GNN モデル（軽量代替）
================================
事故サンプルを直接ノード化し、kNNで接続するシンプルなGNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np

try:
    from torch_geometric.nn import SAGEConv, GCNConv, GATConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import NeighborLoader, DataLoader
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False


class KNNGraphGNN(nn.Module):
    """
    kNN-GraphベースのGNN分類器
    
    事故サンプルを直接ノードとして扱い、
    空間的近傍関係をkNNグラフで表現
    
    軽量かつ高速な代替実装
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.3,
        conv_type: str = 'sage',  # 'sage', 'gcn', 'gat'
        use_edge_attr: bool = True,
    ):
        super().__init__()
        
        if not HAS_PYGEOMETRIC:
            raise ImportError("torch_geometric is required for KNNGraphGNN")
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_edge_attr = use_edge_attr
        
        # 入力変換
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # GNN層
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        for i in range(num_layers):
            if conv_type == 'sage':
                self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            elif conv_type == 'gcn':
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif conv_type == 'gat':
                self.convs.append(GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True))
            
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        # エッジ特徴量の処理（距離）
        if use_edge_attr:
            self.edge_encoder = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        
        # 分類ヘッド
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        self.dropout = dropout
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: ノード特徴量 [N, input_dim]
            edge_index: エッジリスト [2, E]
            edge_attr: エッジ特徴量 [E, 1] (距離)
            batch: バッチインデックス（ミニバッチ学習用）
            
        Returns:
            logits: [N, 1]
        """
        # 入力変換
        x = self.input_proj(x)
        x = F.relu(x)
        
        # GNN層を適用
        for conv, bn in zip(self.convs, self.bns):
            if self.use_edge_attr and edge_attr is not None:
                # エッジ特徴量を使用（SAGEConvは直接サポートしていないため、
                # ここでは単純化している）
                x_new = conv(x, edge_index)
            else:
                x_new = conv(x, edge_index)
            
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            
            # 残差接続
            x = x + x_new
        
        # ノードレベルの分類
        logits = self.classifier(x)
        
        return logits
    
    def predict_proba(self, x, edge_index, edge_attr=None, batch=None):
        logits = self.forward(x, edge_index, edge_attr, batch)
        return torch.sigmoid(logits)


class KNNGraphSAGE(nn.Module):
    """
    GraphSAGEベースのkNN-GNN
    
    ミニバッチ学習向けに最適化
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.3,
        aggr: str = 'mean',  # 'mean', 'max', 'lstm'
    ):
        super().__init__()
        
        if not HAS_PYGEOMETRIC:
            raise ImportError("torch_geometric is required")
        
        self.num_layers = num_layers
        
        # SAGEConv層
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(input_dim, hidden_dim, aggr=aggr))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim, aggr=aggr))
        
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return self.classifier(x)
    
    def predict_proba(self, x, edge_index):
        return torch.sigmoid(self.forward(x, edge_index))


class KNNGraphDataset(torch.utils.data.Dataset):
    """kNNグラフ用データセット"""
    
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.edge_index = edge_index
        self.edge_attr = edge_attr
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'target': self.targets[idx],
            'idx': idx,
        }
    
    def get_full_graph_data(self):
        """フルグラフデータを取得（PyG Data形式）"""
        return Data(
            x=self.features,
            y=self.targets,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
        )


def create_mini_batch_loader(
    data,  # PyG Data object
    batch_size: int = 1024,
    num_neighbors: list = None,
    shuffle: bool = True,
):
    """
    NeighborLoaderでミニバッチを作成
    
    大規模グラフでもGPUメモリに収まるようにサンプリング
    """
    if not HAS_PYGEOMETRIC:
        raise ImportError("torch_geometric is required")
    
    if num_neighbors is None:
        num_neighbors = [10, 5]
    
    loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=shuffle,
        input_nodes=None,  # 全ノード
    )
    
    return loader


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # バイナリクロスエントロピー
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_weight * bce
        
        return loss.mean()
