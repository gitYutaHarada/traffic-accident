"""
Denoising Autoencoder (DAE) 特徴量抽出器
==========================================
Porto Seguroコンペ優勝手法を参考に、テーブルデータ向けのDAEを実装。
入力データにSwap Noiseを加え、それを復元する学習を通じて、
「隠れた関係性」を捉えた特徴量（ボトルネック層の出力）を生成します。

特徴:
- RankGauss: 数値変数を正規分布に変換
- Swap Noise: 入力直後にランダムに値を入れ替える
- Embedding: カテゴリ変数を低次元ベクトルに埋め込む
- Loss: MSE(数値) + CrossEntropy(カテゴリ)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class RankGaussTransformer:
    """数値変数をRankGauss（正規分布）に変換"""
    
    def __init__(self):
        self.transformers = {}
    
    def fit(self, X: pd.DataFrame, numeric_cols: List[str]):
        for col in numeric_cols:
            qt = QuantileTransformer(output_distribution='normal', random_state=42)
            qt.fit(X[[col]].values)
            self.transformers[col] = qt
        return self
    
    def transform(self, X: pd.DataFrame, numeric_cols: List[str]) -> np.ndarray:
        result = []
        for col in numeric_cols:
            if col in self.transformers:
                result.append(self.transformers[col].transform(X[[col]].values))
            else:
                result.append(X[[col]].values)
        return np.hstack(result).astype(np.float32)
    
    def fit_transform(self, X: pd.DataFrame, numeric_cols: List[str]) -> np.ndarray:
        self.fit(X, numeric_cols)
        return self.transform(X, numeric_cols)


class CategoryEncoder:
    """カテゴリ変数をLabel Encodingし、Embedding用のインデックスを生成"""
    
    def __init__(self):
        self.encoders = {}
        self.n_classes = {}
    
    def fit(self, X: pd.DataFrame, cat_cols: List[str]):
        for col in cat_cols:
            le = LabelEncoder()
            # カテゴリ型の場合は先にstr変換してからfillna
            col_values = X[col].astype(str).fillna('__missing__').tolist()
            # 未知のカテゴリに対応するため、fit時に'__unknown__'を追加
            le.fit(col_values + ['__unknown__', '__missing__'])
            self.encoders[col] = le
            self.n_classes[col] = len(le.classes_)
        return self
    
    def transform(self, X: pd.DataFrame, cat_cols: List[str]) -> np.ndarray:
        result = []
        for col in cat_cols:
            le = self.encoders[col]
            # カテゴリ型の場合は先にstr変換してからfillna
            values = X[col].astype(str).fillna('__missing__')
            # 未知のカテゴリを'__unknown__'に変換
            encoded = []
            for v in values:
                if v in le.classes_:
                    encoded.append(le.transform([v])[0])
                else:
                    encoded.append(le.transform(['__unknown__'])[0])
            result.append(np.array(encoded).reshape(-1, 1))
        return np.hstack(result).astype(np.int64)
    
    def fit_transform(self, X: pd.DataFrame, cat_cols: List[str]) -> np.ndarray:
        self.fit(X, cat_cols)
        return self.transform(X, cat_cols)


class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder モデル
    
    構造:
    Input -> [Swap Noise] -> [Embeddings + Numeric] -> Dense(1500) -> Dense(128) [Bottleneck] -> Dense(1500) -> Output
    """
    
    def __init__(
        self,
        n_numeric: int,
        cat_cardinalities: List[int],
        embedding_dim: int = 8,
        hidden_dim: int = 1500,
        bottleneck_dim: int = 128,
        swap_noise_rate: float = 0.15,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.n_numeric = n_numeric
        self.cat_cardinalities = cat_cardinalities
        self.swap_noise_rate = swap_noise_rate
        
        # Embedding層（各カテゴリ変数）
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_classes, embedding_dim) for n_classes in cat_cardinalities
        ])
        
        total_cat_dim = len(cat_cardinalities) * embedding_dim
        input_dim = n_numeric + total_cat_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),  # Bottleneck (Linear activation for features)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 出力ヘッド（数値: 回帰, カテゴリ: 分類）
        self.numeric_head = nn.Linear(hidden_dim, n_numeric)
        self.cat_heads = nn.ModuleList([
            nn.Linear(hidden_dim, n_classes) for n_classes in cat_cardinalities
        ])
    
    def swap_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Swap Noise: ランダムに他の行の値と入れ替える"""
        if not self.training or self.swap_noise_rate == 0:
            return x
        
        noise_mask = torch.rand_like(x) < self.swap_noise_rate
        shuffle_idx = torch.randperm(x.size(0))
        noisy_x = torch.where(noise_mask, x[shuffle_idx], x)
        return noisy_x
    
    def forward(self, numeric: torch.Tensor, categories: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # Swap Noise適用（入力直後）
        numeric_noisy = self.swap_noise(numeric)
        categories_noisy = self.swap_noise(categories.float()).long()
        
        # Embedding
        cat_embedded = [emb(categories_noisy[:, i]) for i, emb in enumerate(self.embeddings)]
        cat_embedded = torch.cat(cat_embedded, dim=1) if cat_embedded else torch.zeros(numeric.size(0), 0)
        
        # 結合
        x = torch.cat([numeric_noisy, cat_embedded], dim=1)
        
        # Encode -> Bottleneck
        bottleneck = self.encoder(x)
        
        # Decode
        decoded = self.decoder(bottleneck)
        
        # 出力
        numeric_out = self.numeric_head(decoded)
        cat_outs = [head(decoded) for head in self.cat_heads]
        
        return bottleneck, numeric_out, cat_outs
    
    def get_features(self, numeric: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
        """ボトルネック特徴量を取得（推論用）"""
        self.eval()
        with torch.no_grad():
            cat_embedded = [emb(categories[:, i]) for i, emb in enumerate(self.embeddings)]
            cat_embedded = torch.cat(cat_embedded, dim=1) if cat_embedded else torch.zeros(numeric.size(0), 0)
            x = torch.cat([numeric, cat_embedded], dim=1)
            bottleneck = self.encoder(x)
        return bottleneck


class DAEFeatureExtractor:
    """
    DAE特徴量抽出器 (高レベルAPI)
    
    使い方:
        extractor = DAEFeatureExtractor(numeric_cols, cat_cols)
        extractor.fit(X_train)
        features = extractor.transform(X_test)
    """
    
    def __init__(
        self,
        numeric_cols: List[str],
        cat_cols: List[str],
        bottleneck_dim: int = 128,
        hidden_dim: int = 1500,
        embedding_dim: int = 8,
        swap_noise_rate: float = 0.15,
        batch_size: int = 512,
        epochs: int = 50,
        lr: float = 1e-3,
        patience: int = 5,
        verbose: bool = True,
        n_workers: int = 0,  # DataLoader用ワーカー数 (Windowsは0推奨)
    ):
        self.numeric_cols = numeric_cols
        self.cat_cols = cat_cols
        self.bottleneck_dim = bottleneck_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.swap_noise_rate = swap_noise_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.verbose = verbose
        self.n_workers = n_workers
        
        self.rank_gauss = RankGaussTransformer()
        self.cat_encoder = CategoryEncoder()
        self.model = None
        # GPU自動検出: CUDAが利用可能ならGPUを使用
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.verbose:
            print(f"🖥️ DAE Device: {self.device}")
    
    def fit(self, X: pd.DataFrame):
        """DAEを学習"""
        import time
        start_time = time.time()
        
        if self.verbose:
            print(f"📦 DAE学習開始 (Bottleneck={self.bottleneck_dim}, epochs={self.epochs}, device={self.device})")
        
        # 前処理
        print(f"   ⏳ [Preproc] RankGauss & LabelEncoding starting...")
        X_num = self.rank_gauss.fit_transform(X, self.numeric_cols)
        X_cat = self.cat_encoder.fit_transform(X, self.cat_cols)
        print(f"   ✅ [Preproc] Done in {time.time() - start_time:.1f}s")
        
        # Train/Val分割
        X_num_train, X_num_val, X_cat_train, X_cat_val = train_test_split(
            X_num, X_cat, test_size=0.1, random_state=42
        )
        
        train_dataset = TensorDataset(
            torch.tensor(X_num_train, dtype=torch.float32),
            torch.tensor(X_cat_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_num_val, dtype=torch.float32),
            torch.tensor(X_cat_val, dtype=torch.long)
        )
        
        use_cuda = self.device.type == 'cuda'
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.n_workers, pin_memory=use_cuda
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.n_workers, pin_memory=use_cuda
        )
        
        # モデル初期化
        cat_cardinalities = [self.cat_encoder.n_classes[col] for col in self.cat_cols]
        self.model = DenoisingAutoencoder(
            n_numeric=len(self.numeric_cols),
            cat_cardinalities=cat_cardinalities,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            bottleneck_dim=self.bottleneck_dim,
            swap_noise_rate=self.swap_noise_rate,
        ).to(self.device)
        
        # 損失関数
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss()
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=self.lr, epochs=self.epochs, steps_per_epoch=len(train_loader)
        )
        scaler = torch.amp.GradScaler('cuda') if use_cuda else None
        
        # 学習ループ
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        train_start_time = time.time()
        print(f"   🚀 [Train] Start training loop...")
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # Train
            self.model.train()
            train_loss = 0.0
            
            for numeric, categories in train_loader:
                numeric = numeric.to(self.device)
                categories = categories.to(self.device)
                
                optimizer.zero_grad()
                
                if use_cuda:
                    with torch.amp.autocast('cuda'):
                        _, numeric_out, cat_outs = self.model(numeric, categories)
                        loss = mse_loss(numeric_out, numeric)
                        for i, cat_out in enumerate(cat_outs):
                            loss += ce_loss(cat_out, categories[:, i])
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    _, numeric_out, cat_outs = self.model(numeric, categories)
                    loss = mse_loss(numeric_out, numeric)
                    for i, cat_out in enumerate(cat_outs):
                        loss += ce_loss(cat_out, categories[:, i])
                    loss.backward()
                    optimizer.step()
                
                scheduler.step()
                train_loss += loss.item()
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for numeric, categories in val_loader:
                    numeric = numeric.to(self.device)
                    categories = categories.to(self.device)
                    
                    if use_cuda:
                        with torch.amp.autocast('cuda'):
                            _, numeric_out, cat_outs = self.model(numeric, categories)
                            loss = mse_loss(numeric_out, numeric)
                            for i, cat_out in enumerate(cat_outs):
                                loss += ce_loss(cat_out, categories[:, i])
                    else:
                        _, numeric_out, cat_outs = self.model(numeric, categories)
                        loss = mse_loss(numeric_out, numeric)
                        for i, cat_out in enumerate(cat_outs):
                            loss += ce_loss(cat_out, categories[:, i])
                    
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # ログ表示 (毎回表示して速度感を確認)
            elapsed = time.time() - epoch_start
            print(f"      Epoch {epoch+1}/{self.epochs}: T-Loss={train_loss:.4f}, V-Loss={val_loss:.4f} ({elapsed:.2f}s/ep)")
            
            # Early Stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    if self.verbose:
                        print(f"   ⏹️ Early Stopping at epoch {epoch+1}")
                    break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
        
        total_time = time.time() - train_start_time
        if self.verbose:
            print(f"✅ DAE学習完了 (Best Val Loss={best_val_loss:.4f}, Total Train Time={total_time:.1f}s)")
        
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """ボトルネック特徴量を抽出"""
        if self.model is None:
            raise ValueError("DAEがまだ学習されていません。fit()を先に呼んでください。")
        
        X_num = self.rank_gauss.transform(X, self.numeric_cols)
        X_cat = self.cat_encoder.transform(X, self.cat_cols)
        
        dataset = TensorDataset(
            torch.tensor(X_num, dtype=torch.float32),
            torch.tensor(X_cat, dtype=torch.long)
        )
        use_cuda = self.device.type == 'cuda'
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.n_workers, pin_memory=use_cuda
        )
        
        features = []
        self.model.eval()
        with torch.no_grad():
            for numeric, categories in loader:
                numeric = numeric.to(self.device)
                categories = categories.to(self.device)
                bottleneck = self.model.get_features(numeric, categories)
                features.append(bottleneck.cpu().numpy())
        
        return np.vstack(features)
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """学習してから特徴量抽出"""
        self.fit(X)
        return self.transform(X)


if __name__ == "__main__":
    # テスト用
    print("DAE Feature Extractor - Test Run")
    
    # ダミーデータ生成
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'num1': np.random.randn(n_samples),
        'num2': np.random.randn(n_samples) * 10,
        'cat1': np.random.choice(['A', 'B', 'C'], n_samples),
        'cat2': np.random.choice(['X', 'Y'], n_samples),
    })
    
    extractor = DAEFeatureExtractor(
        numeric_cols=['num1', 'num2'],
        cat_cols=['cat1', 'cat2'],
        bottleneck_dim=16,
        epochs=20,
        verbose=True
    )
    
    features = extractor.fit_transform(df)
    print(f"Generated features shape: {features.shape}")  # (1000, 16)
