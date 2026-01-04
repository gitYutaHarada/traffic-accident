"""
Spatio-Temporal Stage2 学習パイプライン
======================================
全モデルの学習・評価・比較
チェックポイント機能付き
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')

# 自作モジュール
from utils.checkpoint import CheckpointManager, EarlyStopping, set_seed
from evaluate import evaluate_model, ModelEvaluator
from visualize import Visualizer

# ランダムシード
RANDOM_SEED = 42


class SpatioTemporalTrainer:
    """
    Spatio-Temporal モデル学習クラス
    
    - 複数モデルの学習・比較
    - チェックポイント管理
    - TensorBoardログ
    """
    
    def __init__(
        self,
        data_dir: str = "data/spatio_temporal",
        output_dir: str = "results/spatio_temporal",
        model_type: str = "knn_gnn",  # 'lstm', 'tgcn', 'gat', 'knn_gnn'
        config: Optional[Dict] = None,
        device: str = "auto",
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_type = model_type
        self.config = config or self._default_config()
        
        # デバイス設定
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"   🖥️ デバイス: {self.device}")
        
        # チェックポイントマネージャー
        self.ckpt_manager = CheckpointManager(
            self.output_dir / "checkpoints" / model_type
        )
        
        # TensorBoardライター
        self.writer = SummaryWriter(self.output_dir / "logs" / model_type)
        
        # シード固定
        set_seed(RANDOM_SEED)
    
    def _default_config(self) -> Dict:
        """デフォルト設定"""
        return {
            'hidden_dim': 128,
            'num_layers': 2,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 1024,
            'epochs': 100,
            'patience': 15,
            'focal_alpha': 0.75,
            'focal_gamma': 2.0,
            'k_neighbors': 8,
        }
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """前処理済みデータの読み込み"""
        print("\n📂 データ読み込み中...")
        
        train_df = pd.read_parquet(self.data_dir / "preprocessed_train.parquet")
        val_df = pd.read_parquet(self.data_dir / "preprocessed_val.parquet")
        test_df = pd.read_parquet(self.data_dir / "preprocessed_test.parquet")
        
        print(f"   Train: {len(train_df):,} 行")
        print(f"   Val:   {len(val_df):,} 行")
        print(f"   Test:  {len(test_df):,} 行")
        
        return train_df, val_df, test_df
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = 'fatal',
    ) -> Tuple[np.ndarray, np.ndarray]:
        """特徴量とターゲットの準備"""
        
        # 除外列
        exclude_cols = [target_col, 'date', 'lat', 'lon', 'geohash', 'geohash_fine', 'year']
        
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols].values.astype(np.float32)
        y = df[target_col].values.astype(np.float32)
        
        # NaN処理
        X = np.nan_to_num(X, nan=0.0)
        
        return X, y
    
    def create_model(self, input_dim: int):
        """モデルの作成"""
        
        if self.model_type == 'knn_gnn':
            from models.knn_gnn import KNNGraphGNN
            model = KNNGraphGNN(
                input_dim=input_dim,
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout'],
            )
        elif self.model_type == 'mlp':
            # シンプルなMLPベースライン
            model = nn.Sequential(
                nn.Linear(input_dim, self.config['hidden_dim']),
                nn.ReLU(),
                nn.BatchNorm1d(self.config['hidden_dim']),
                nn.Dropout(self.config['dropout']),
                nn.Linear(self.config['hidden_dim'], self.config['hidden_dim'] // 2),
                nn.ReLU(),
                nn.BatchNorm1d(self.config['hidden_dim'] // 2),
                nn.Dropout(self.config['dropout']),
                nn.Linear(self.config['hidden_dim'] // 2, 1),
            )
        elif self.model_type == 'lstm':
            from models.lstm_geohash import GeoHashLSTM
            # LSTMの場合は時系列データが必要
            # ここでは簡略化してシンプルなMLPを使用
            model = nn.Sequential(
                nn.Linear(input_dim, self.config['hidden_dim']),
                nn.ReLU(),
                nn.Dropout(self.config['dropout']),
                nn.Linear(self.config['hidden_dim'], self.config['hidden_dim'] // 2),
                nn.ReLU(),
                nn.Dropout(self.config['dropout']),
                nn.Linear(self.config['hidden_dim'] // 2, 1),
            )
        elif self.model_type == 'tgcn':
            from models.temporal_gnn import SimpleTGCN
            model = SimpleTGCN(
                input_dim=input_dim,
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout'],
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return model.to(self.device)
    
    def train_simple_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[nn.Module, Dict]:
        """
        シンプルなNN/MLPモデルの学習
        （GNNを使わないベースライン）
        """
        print(f"\n🌿 {self.model_type} モデル学習中...")
        
        # データローダー作成
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=0,
        )
        
        # モデル作成
        model = self.create_model(X_train.shape[1])
        
        # オプティマイザ
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # 損失関数（Focal Loss）
        from models.knn_gnn import FocalLoss
        criterion = FocalLoss(
            alpha=self.config['focal_alpha'],
            gamma=self.config['focal_gamma']
        )
        
        # Early Stopping
        early_stopping = EarlyStopping(patience=self.config['patience'], mode='max')
        
        # チェックポイントからの再開
        resume_info = self.ckpt_manager.get_resume_info()
        start_epoch = 0
        
        if resume_info['should_resume']:
            checkpoint = self.ckpt_manager.load_checkpoint(model, optimizer, scheduler)
            start_epoch = checkpoint['epoch'] + 1
            print(f"   再開: epoch {start_epoch} から")
        
        # 学習ループ
        best_val_auc = 0.0
        
        for epoch in range(start_epoch, self.config['epochs']):
            # 学習
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # 検証
            model.eval()
            val_preds = []
            val_targets = []
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    probs = torch.sigmoid(outputs)
                    val_preds.extend(probs.cpu().numpy().flatten())
                    val_targets.extend(batch_y.cpu().numpy().flatten())
            
            val_loss /= len(val_loader)
            val_preds = np.array(val_preds)
            val_targets = np.array(val_targets)
            
            # 評価指標
            from sklearn.metrics import roc_auc_score, average_precision_score
            val_auc = roc_auc_score(val_targets, val_preds)
            val_pr_auc = average_precision_score(val_targets, val_preds)
            
            # スケジューラ更新
            scheduler.step(val_loss)
            
            # ログ
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('AUC/val', val_auc, epoch)
            self.writer.add_scalar('PR-AUC/val', val_pr_auc, epoch)
            
            # 改善チェック
            is_best = early_stopping(val_auc)
            
            if is_best:
                best_val_auc = val_auc
            
            # チェックポイント保存
            if epoch % 5 == 0 or is_best:
                self.ckpt_manager.save_checkpoint(
                    model, optimizer, epoch, 0,
                    {'val_auc': val_auc, 'val_pr_auc': val_pr_auc, 'val_loss': val_loss},
                    self.config, scheduler, is_best
                )
            
            # 進捗表示
            if epoch % 5 == 0:
                print(f"   Epoch {epoch:3d}: Loss={train_loss:.4f}/{val_loss:.4f}, "
                      f"AUC={val_auc:.4f}, PR-AUC={val_pr_auc:.4f}")
            
            # Early Stopping
            if early_stopping.should_stop:
                print(f"   ⏹️ Early Stopping at epoch {epoch}")
                break
        
        # ベストモデルを読み込み
        self.ckpt_manager.load_best_model(model)
        
        return model, {'best_val_auc': best_val_auc}
    
    def train_gnn_model(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        edge_index: torch.Tensor,
    ) -> Tuple[nn.Module, Dict]:
        """
        GNNモデルの学習（Inductive: 全データ統合グラフ + ノードマスク方式）
        """
        print(f"\n🌿 GNN ({self.model_type}) Inductive学習中...")
        
        # 特徴量準備 (train + val を結合した combined_df を使う)
        # ここでは combined_df として渡された train_df を使用
        X_all, y_all = self.prepare_features(train_df)  # train_df は実際には combined_df
        
        # モデル作成
        if self.model_type == 'knn_gnn':
            from models.knn_gnn import KNNGraphGNN, FocalLoss
            model = KNNGraphGNN(
                input_dim=X_all.shape[1],
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout'],
            ).to(self.device)
        elif self.model_type == 'tgcn':
            from models.temporal_gnn import SimpleTGCN
            model = SimpleTGCN(
                input_dim=X_all.shape[1],
                hidden_dim=self.config['hidden_dim'],
                num_layers=self.config['num_layers'],
                dropout=self.config['dropout'],
            ).to(self.device)
        else:
            raise ValueError(f"Unknown GNN model type: {self.model_type}")
        
        # データをGPUに転送
        X_all_t = torch.tensor(X_all, dtype=torch.float32).to(self.device)
        y_all_t = torch.tensor(y_all, dtype=torch.float32).unsqueeze(1).to(self.device)
        edge_index = edge_index.to(self.device)
        
        # マスクもGPUに転送（val_dfにマスクが含まれている想定）
        train_mask = val_df['train_mask'].to(self.device)
        val_mask = val_df['val_mask'].to(self.device)
        
        # オプティマイザ
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        
        # 損失関数
        from models.knn_gnn import FocalLoss
        criterion = FocalLoss(
            alpha=self.config['focal_alpha'],
            gamma=self.config['focal_gamma']
        )
        
        # Early Stopping
        early_stopping = EarlyStopping(patience=self.config['patience'], mode='max')
        
        best_val_auc = 0.0
        
        for epoch in range(self.config['epochs']):
            # 学習
            model.train()
            optimizer.zero_grad()
            
            # 全ノードに対して forward
            outputs = model(X_all_t, edge_index)
            
            # Train マスクのノードのみで損失計算
            train_outputs = outputs[train_mask]
            train_targets = y_all_t[train_mask]
            loss = criterion(train_outputs, train_targets)
            
            loss.backward()
            optimizer.step()
            
            # 検証 (Val マスクのノードで評価)
            model.eval()
            with torch.no_grad():
                val_outputs = outputs[val_mask]
                val_preds = torch.sigmoid(val_outputs).cpu().numpy().flatten()
                val_targets_np = y_all_t[val_mask].cpu().numpy().flatten()
            
            from sklearn.metrics import roc_auc_score, average_precision_score
            val_auc = roc_auc_score(val_targets_np, val_preds)
            val_pr_auc = average_precision_score(val_targets_np, val_preds)
            
            # ログ
            self.writer.add_scalar('Loss/train', loss.item(), epoch)
            self.writer.add_scalar('AUC/val', val_auc, epoch)
            self.writer.add_scalar('PR-AUC/val', val_pr_auc, epoch)
            
            is_best = early_stopping(val_auc)
            if is_best:
                best_val_auc = val_auc
            
            # チェックポイント
            if epoch % 10 == 0 or is_best:
                self.ckpt_manager.save_checkpoint(
                    model, optimizer, epoch, 0,
                    {'val_auc': val_auc, 'val_pr_auc': val_pr_auc},
                    self.config, is_best=is_best
                )
            
            if epoch % 10 == 0:
                print(f"   Epoch {epoch:3d}: Loss={loss.item():.4f}, "
                      f"AUC={val_auc:.4f}, PR-AUC={val_pr_auc:.4f}")
            
            if early_stopping.should_stop:
                print(f"   ⏹️ Early Stopping at epoch {epoch}")
                break
        
        self.ckpt_manager.load_best_model(model)
        
        return model, {'best_val_auc': best_val_auc}
    
    def evaluate_on_test(
        self,
        model: nn.Module,
        test_df: pd.DataFrame,
        edge_index: Optional[torch.Tensor] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """テストセットでの評価"""
        print("\n📊 テストセット評価中...")
        
        X_test, y_test = self.prepare_features(test_df)
        
        model.eval()
        
        if edge_index is not None and self.model_type in ['knn_gnn', 'tgcn']:
            # GNNモデル
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            edge_index = edge_index.to(self.device)
            
            with torch.no_grad():
                outputs = model(X_test_t, edge_index)
                predictions = torch.sigmoid(outputs).cpu().numpy().flatten()
        else:
            # 通常のNNモデル
            X_test_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                outputs = model(X_test_t)
                predictions = torch.sigmoid(outputs).cpu().numpy().flatten()
        
        # 評価
        metrics = evaluate_model(y_test, predictions)
        
        print(f"   PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"   ECE: {metrics['ece']:.4f}")
        
        return predictions, metrics
    
    def run(self):
        """学習パイプラインの実行"""
        start_time = datetime.now()
        
        # データ読み込み
        train_df, val_df, test_df = self.load_data()
        
        # 特徴量準備
        X_train, y_train = self.prepare_features(train_df)
        X_val, y_val = self.prepare_features(val_df)
        
        print(f"   特徴量次元: {X_train.shape[1]}")
        
        # モデル学習
        if self.model_type in ['lstm', 'mlp']:
            model, train_info = self.train_simple_model(X_train, y_train, X_val, y_val)
            edge_index = None
            graph_data = None
        else:
            # Inductive グラフ構築（全データ統合）
            from graph_builder import build_inductive_graph
            
            graph_data = build_inductive_graph(
                train_df, 
                val_df,
                test_df,
                k=self.config['k_neighbors'],
                output_dir=self.data_dir
            )
            edge_index = graph_data['edge_index']
            
            # マスク情報を辞書として渡す
            mask_info = {
                'train_mask': graph_data['train_mask'],
                'val_mask': graph_data['val_mask'],
                'test_mask': graph_data['test_mask'],
            }
            
            # 結合されたデータで学習
            combined_df = graph_data['combined_df']
            
            model, train_info = self.train_gnn_model(combined_df, mask_info, edge_index)
        
        # テスト評価
        if graph_data is not None:
            # GNN の場合は combined_df と test_mask で評価
            predictions, test_metrics = self.evaluate_on_test_gnn(
                model, 
                graph_data['combined_df'], 
                edge_index,
                graph_data['test_mask']
            )
            # テスト予測のみ抽出
            test_indices = graph_data['test_mask'].numpy()
            test_df_with_pred = test_df.copy()
            test_df_with_pred['prediction'] = predictions
        else:
            predictions, test_metrics = self.evaluate_on_test(model, test_df, edge_index)
            test_df_with_pred = test_df.copy()
            test_df_with_pred['prediction'] = predictions
        
        # 結果保存
        test_df_with_pred.to_parquet(self.output_dir / "test_predictions.parquet")
        
        # 結果サマリ
        results = {
            'model_type': self.model_type,
            'config': self.config,
            'train_info': train_info,
            'test_metrics': test_metrics,
            'elapsed_seconds': (datetime.now() - start_time).total_seconds(),
        }
        
        with open(self.output_dir / f"results_{self.model_type}.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # TensorBoardクローズ
        self.writer.close()
        
        print("\n" + "=" * 70)
        print(f"✅ {self.model_type} 学習完了！")
        print(f"   PR-AUC: {test_metrics['pr_auc']:.4f}")
        print(f"   所要時間: {results['elapsed_seconds']:.1f}秒")
        print("=" * 70)
        
        return results
    
    def evaluate_on_test_gnn(
        self,
        model: nn.Module,
        combined_df: pd.DataFrame,
        edge_index: torch.Tensor,
        test_mask: torch.Tensor,
    ) -> Tuple[np.ndarray, Dict]:
        """GNNモデルのテストセット評価（マスク方式）"""
        print("\n📊 テストセット評価中（Inductive GNN）...")
        
        X_all, y_all = self.prepare_features(combined_df)
        
        model.eval()
        
        X_all_t = torch.tensor(X_all, dtype=torch.float32).to(self.device)
        edge_index = edge_index.to(self.device)
        test_mask = test_mask.to(self.device)
        
        with torch.no_grad():
            outputs = model(X_all_t, edge_index)
            test_outputs = outputs[test_mask]
            predictions = torch.sigmoid(test_outputs).cpu().numpy().flatten()
        
        # テストセットのターゲット
        y_test = y_all[test_mask.cpu().numpy()]
        
        # 評価
        metrics = evaluate_model(y_test, predictions)
        
        print(f"   PR-AUC: {metrics['pr_auc']:.4f}")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"   ECE: {metrics['ece']:.4f}")
        
        return predictions, metrics



def main():
    parser = argparse.ArgumentParser(description="Spatio-Temporal Model Training")
    parser.add_argument('--data-dir', type=str, default="data/spatio_temporal")
    parser.add_argument('--output-dir', type=str, default="results/spatio_temporal")
    parser.add_argument('--model', type=str, default="knn_gnn",
                        choices=['lstm', 'tgcn', 'gat', 'knn_gnn', 'mlp', 'all'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--k', type=int, default=8, help="k for kNN graph")
    parser.add_argument('--debug', action='store_true', help="デバッグモード（少ないエポック）")
    
    args = parser.parse_args()
    
    config = {
        'hidden_dim': args.hidden_dim,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'epochs': 2 if args.debug else args.epochs,
        'patience': 15,
        'focal_alpha': 0.75,
        'focal_gamma': 2.0,
        'k_neighbors': args.k,
    }
    
    if args.model == 'all':
        models = ['mlp', 'knn_gnn']
        all_results = {}
        
        for model_type in models:
            print(f"\n{'='*70}")
            print(f"モデル: {model_type}")
            print(f"{'='*70}")
            
            trainer = SpatioTemporalTrainer(
                data_dir=args.data_dir,
                output_dir=args.output_dir,
                model_type=model_type,
                config=config,
            )
            
            results = trainer.run()
            all_results[model_type] = results
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 比較レポート生成
        print("\n📊 モデル比較:")
        for model_type, results in all_results.items():
            print(f"   {model_type}: PR-AUC={results['test_metrics']['pr_auc']:.4f}")
    else:
        trainer = SpatioTemporalTrainer(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            model_type=args.model,
            config=config,
        )
        
        trainer.run()


if __name__ == "__main__":
    main()
