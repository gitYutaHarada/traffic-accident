"""
ジオハッシュ単位 LSTM 時系列モデル
=================================
各ジオハッシュセルの時系列を学習し、事故サンプルに結合して分類
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class GeoHashLSTM(nn.Module):
    """
    ジオハッシュ単位の事故時系列をLSTMで学習するモデル
    
    アーキテクチャ:
    1. 各ジオハッシュの時系列特徴量をLSTMでエンコード
    2. セル特徴量を事故サンプルの静的特徴量と結合
    3. 分類層で予測
    """
    
    def __init__(
        self,
        temporal_input_dim: int,  # 時系列特徴量の次元
        static_input_dim: int,     # 静的特徴量の次元
        lstm_hidden_dim: int = 128,
        lstm_num_layers: int = 2,
        fc_hidden_dim: int = 64,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        
        self.temporal_input_dim = temporal_input_dim
        self.static_input_dim = static_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.bidirectional = bidirectional
        
        # LSTM層
        self.lstm = nn.LSTM(
            input_size=temporal_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional,
        )
        
        # 特徴量結合後の次元
        lstm_output_dim = lstm_hidden_dim * (2 if bidirectional else 1)
        combined_dim = lstm_output_dim + static_input_dim
        
        # 分類層
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, fc_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim // 2, 1),
        )
        
        # 初期化
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        temporal_features: torch.Tensor,  # [batch, seq_len, temporal_dim]
        static_features: torch.Tensor,     # [batch, static_dim]
        lengths: Optional[torch.Tensor] = None,  # 可変長系列用
    ) -> torch.Tensor:
        """
        Args:
            temporal_features: 時系列特徴量 [batch, seq_len, temporal_dim]
            static_features: 静的特徴量 [batch, static_dim]
            lengths: 各系列の実際の長さ（パディング用）
            
        Returns:
            logits: [batch, 1]
        """
        batch_size = temporal_features.size(0)
        
        if lengths is not None:
            # 可変長系列の処理
            packed = nn.utils.rnn.pack_padded_sequence(
                temporal_features, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(packed)
        else:
            lstm_out, (h_n, c_n) = self.lstm(temporal_features)
        
        # 最終隠れ状態を取得
        if self.bidirectional:
            # 順方向と逆方向の最終状態を結合
            h_forward = h_n[-2, :, :]  # [batch, hidden]
            h_backward = h_n[-1, :, :]
            temporal_encoding = torch.cat([h_forward, h_backward], dim=1)
        else:
            temporal_encoding = h_n[-1, :, :]
        
        # 静的特徴量と結合
        combined = torch.cat([temporal_encoding, static_features], dim=1)
        
        # 分類
        logits = self.classifier(combined)
        
        return logits
    
    def predict_proba(self, temporal_features, static_features, lengths=None):
        """確率を出力"""
        logits = self.forward(temporal_features, static_features, lengths)
        return torch.sigmoid(logits)


class GeoHash1DCNN(nn.Module):
    """
    ジオハッシュ単位の事故時系列を1D CNNで学習するモデル
    
    LSTMよりも計算が軽く、並列処理向き
    """
    
    def __init__(
        self,
        temporal_input_dim: int,
        static_input_dim: int,
        seq_len: int = 30,  # 時系列の長さ
        cnn_channels: list = [64, 128, 256],
        kernel_size: int = 3,
        fc_hidden_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.temporal_input_dim = temporal_input_dim
        self.static_input_dim = static_input_dim
        
        # 1D CNN層
        cnn_layers = []
        in_channels = temporal_input_dim
        
        for out_channels in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Dropout(dropout),
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # CNN出力サイズを計算
        cnn_output_len = seq_len
        for _ in cnn_channels:
            cnn_output_len = cnn_output_len // 2
        cnn_output_dim = cnn_channels[-1] * max(1, cnn_output_len)
        
        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        combined_dim = cnn_channels[-1] + static_input_dim
        
        # 分類層
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, fc_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden_dim, 1),
        )
    
    def forward(
        self,
        temporal_features: torch.Tensor,  # [batch, seq_len, temporal_dim]
        static_features: torch.Tensor,     # [batch, static_dim]
    ) -> torch.Tensor:
        # CNNは [batch, channels, seq_len] を期待
        x = temporal_features.permute(0, 2, 1)  # [batch, temporal_dim, seq_len]
        
        x = self.cnn(x)
        x = self.global_pool(x).squeeze(-1)  # [batch, channels]
        
        combined = torch.cat([x, static_features], dim=1)
        logits = self.classifier(combined)
        
        return logits
    
    def predict_proba(self, temporal_features, static_features):
        logits = self.forward(temporal_features, static_features)
        return torch.sigmoid(logits)


class GeoHashSequenceDataset(torch.utils.data.Dataset):
    """ジオハッシュ時系列データセット"""
    
    def __init__(
        self,
        df,
        geohash_sequences: dict,  # {geohash: [daily_features]}
        static_feature_cols: list,
        target_col: str = 'fatal',
        seq_len: int = 30,
    ):
        self.df = df.reset_index(drop=True)
        self.geohash_sequences = geohash_sequences
        self.static_feature_cols = static_feature_cols
        self.target_col = target_col
        self.seq_len = seq_len
        
        # 静的特徴量
        self.static_features = self.df[static_feature_cols].values.astype(np.float32)
        self.targets = self.df[target_col].values.astype(np.float32)
        self.geohashes = self.df['geohash'].values
        self.dates = pd.to_datetime(self.df['date'])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        geohash = self.geohashes[idx]
        date = self.dates.iloc[idx]
        
        # ジオハッシュの過去時系列を取得
        if geohash in self.geohash_sequences:
            seq = self.geohash_sequences[geohash]
            # 日付より前のデータを取得
            temporal = self._get_past_sequence(seq, date)
        else:
            temporal = np.zeros((self.seq_len, self._get_temporal_dim()))
        
        return {
            'temporal': torch.tensor(temporal, dtype=torch.float32),
            'static': torch.tensor(self.static_features[idx], dtype=torch.float32),
            'target': torch.tensor(self.targets[idx], dtype=torch.float32),
        }
    
    def _get_past_sequence(self, seq, date):
        # 簡略化: seq_len分の過去データを取得
        # 実装では日付でフィルタリングが必要
        if len(seq) >= self.seq_len:
            return np.array(seq[-self.seq_len:])
        else:
            # パディング
            pad_len = self.seq_len - len(seq)
            padded = np.zeros((self.seq_len, seq[0].shape[0] if len(seq) > 0 else 1))
            if len(seq) > 0:
                padded[pad_len:] = np.array(seq)
            return padded
    
    def _get_temporal_dim(self):
        # 時系列特徴量の次元を取得
        for gh, seq in self.geohash_sequences.items():
            if len(seq) > 0:
                return seq[0].shape[0]
        return 1
