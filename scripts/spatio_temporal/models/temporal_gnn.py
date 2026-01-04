"""
Temporal Graph Neural Network モデル
====================================
TGCN (Temporal Graph Convolutional Network)
GAT + Temporal Convolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np

try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    from torch_geometric.data import Data, Batch
    HAS_PYGEOMETRIC = True
except ImportError:
    HAS_PYGEOMETRIC = False
    print("Warning: torch_geometric not available. Using fallback implementation.")


class GRUCell(nn.Module):
    """GRUセル（TGCN用）"""
    
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.W_z = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_r = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.W_h = nn.Linear(input_dim + hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, h], dim=-1)
        
        z = torch.sigmoid(self.W_z(combined))
        r = torch.sigmoid(self.W_r(combined))
        
        combined_r = torch.cat([x, r * h], dim=-1)
        h_tilde = torch.tanh(self.W_h(combined_r))
        
        h_new = (1 - z) * h + z * h_tilde
        
        return h_new


class TemporalGCN(nn.Module):
    """
    Temporal Graph Convolutional Network (TGCN)
    
    GCN + GRU の組み合わせによる時空間モデリング
    
    参考: Zhao et al., "T-GCN: A Temporal Graph Convolutional Network for Traffic Prediction"
    """
    
    def __init__(
        self,
        node_features: int,
        hidden_dim: int = 64,
        num_gcn_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_gcn_layers = num_gcn_layers
        
        if not HAS_PYGEOMETRIC:
            raise ImportError("torch_geometric is required for TemporalGCN")
        
        # GCN層
        self.gcn_layers = nn.ModuleList()
        in_dim = node_features
        for _ in range(num_gcn_layers):
            self.gcn_layers.append(GCNConv(in_dim, hidden_dim))
            in_dim = hidden_dim
        
        # GRU（時間方向）
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # 分類層
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x_seq: List[torch.Tensor],  # [T個の [N, node_features]]
        edge_index_seq: List[torch.Tensor],  # [T個の [2, E]]
        node_mask: Optional[torch.Tensor] = None,  # 予測対象ノード
    ) -> torch.Tensor:
        """
        Args:
            x_seq: 時系列のノード特徴量リスト
            edge_index_seq: 時系列のエッジリスト
            node_mask: 予測対象ノードのマスク
            
        Returns:
            logits: [N, 1] or [masked_N, 1]
        """
        T = len(x_seq)
        N = x_seq[0].size(0)
        
        # 各時点でGCNを適用
        temporal_embeddings = []
        
        for t in range(T):
            x = x_seq[t]
            edge_index = edge_index_seq[t] if t < len(edge_index_seq) else edge_index_seq[-1]
            
            for gcn in self.gcn_layers:
                x = gcn(x, edge_index)
                x = F.relu(x)
                x = self.dropout(x)
            
            temporal_embeddings.append(x)
        
        # [T, N, hidden] -> [N, T, hidden]
        temporal = torch.stack(temporal_embeddings, dim=1)
        
        # GRUで時間方向を処理
        gru_out, _ = self.gru(temporal)
        
        # 最終時点の出力
        final_embedding = gru_out[:, -1, :]  # [N, hidden]
        
        # マスク適用
        if node_mask is not None:
            final_embedding = final_embedding[node_mask]
        
        # 分類
        logits = self.classifier(final_embedding)
        
        return logits
    
    def predict_proba(self, x_seq, edge_index_seq, node_mask=None):
        logits = self.forward(x_seq, edge_index_seq, node_mask)
        return torch.sigmoid(logits)


class GATTemporal(nn.Module):
    """
    Graph Attention Network + Temporal Convolution
    
    GAT でノード間の注意を学習し、時間方向は1D Convで処理
    """
    
    def __init__(
        self,
        node_features: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        num_gat_layers: int = 2,
        temporal_kernel_size: int = 3,
        output_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        if not HAS_PYGEOMETRIC:
            raise ImportError("torch_geometric is required for GATTemporal")
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # GAT層
        self.gat_layers = nn.ModuleList()
        in_dim = node_features
        
        for i in range(num_gat_layers):
            concat = i < num_gat_layers - 1  # 最後の層以外は結合
            out_dim = hidden_dim if not concat else hidden_dim // num_heads
            self.gat_layers.append(
                GATConv(in_dim, out_dim, heads=num_heads, concat=concat, dropout=dropout)
            )
            in_dim = hidden_dim
        
        # Temporal Convolution
        self.temp_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, temporal_kernel_size, padding=temporal_kernel_size//2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, temporal_kernel_size, padding=temporal_kernel_size//2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # Global Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分類層
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x_seq: List[torch.Tensor],
        edge_index_seq: List[torch.Tensor],
        node_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        T = len(x_seq)
        N = x_seq[0].size(0)
        
        # 各時点でGATを適用
        temporal_embeddings = []
        
        for t in range(T):
            x = x_seq[t]
            edge_index = edge_index_seq[t] if t < len(edge_index_seq) else edge_index_seq[-1]
            
            for gat in self.gat_layers:
                x = gat(x, edge_index)
                x = F.elu(x)
                x = self.dropout(x)
            
            temporal_embeddings.append(x)
        
        # [N, T, hidden]
        temporal = torch.stack(temporal_embeddings, dim=1)
        
        # Temporal Convolution: [N, hidden, T]
        temporal = temporal.permute(0, 2, 1)
        temporal = self.temp_conv(temporal)
        
        # Global Pooling
        pooled = self.global_pool(temporal).squeeze(-1)  # [N, hidden]
        
        if node_mask is not None:
            pooled = pooled[node_mask]
        
        logits = self.classifier(pooled)
        
        return logits
    
    def predict_proba(self, x_seq, edge_index_seq, node_mask=None):
        logits = self.forward(x_seq, edge_index_seq, node_mask)
        return torch.sigmoid(logits)


class SimpleTGCN(nn.Module):
    """
    シンプルなTGCN実装（単一時点グラフ用）
    
    静的グラフ + ノード特徴量（時系列埋め込み含む）
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        if not HAS_PYGEOMETRIC:
            raise ImportError("torch_geometric is required")
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )
        
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        logits = self.classifier(x)
        return logits
    
    def predict_proba(self, x, edge_index):
        logits = self.forward(x, edge_index)
        return torch.sigmoid(logits)
