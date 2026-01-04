"""
Optuna ハイパーパラメータ探索
============================
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import warnings
import gc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.trial import Trial

warnings.filterwarnings('ignore')

# 自作モジュール
from utils.checkpoint import set_seed
from models.knn_gnn import KNNGraphGNN, FocalLoss
from graph_builder import GraphBuilder

# ランダムシード
RANDOM_SEED = 42


class OptunaObjective:
    """Optuna最適化の目的関数クラス"""
    
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        coords_train: np.ndarray,
        model_type: str = 'knn_gnn',
        n_epochs: int = 50,
        device: str = 'auto',
    ):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.coords_train = coords_train
        self.model_type = model_type
        self.n_epochs = n_epochs
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
    
    def __call__(self, trial: Trial) -> float:
        """1回のトライアル"""
        
        # ハイパーパラメータのサンプリング
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [64, 128, 256]),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'focal_alpha': trial.suggest_float('focal_alpha', 0.5, 0.9),
            'focal_gamma': trial.suggest_float('focal_gamma', 1.0, 3.0),
            'batch_size': trial.suggest_categorical('batch_size', [512, 1024, 2048]),
        }
        
        if self.model_type == 'knn_gnn':
            params['k_neighbors'] = trial.suggest_int('k_neighbors', 4, 16)
        
        try:
            val_auc = self._train_and_evaluate(params)
        except Exception as e:
            print(f"   ⚠️ Trial failed: {e}")
            return 0.0
        
        return val_auc
    
    def _train_and_evaluate(self, params: Dict) -> float:
        """モデルの学習と評価"""
        
        set_seed(RANDOM_SEED)
        
        if self.model_type == 'knn_gnn':
            # グラフ構築
            builder = GraphBuilder(k=params.get('k_neighbors', 8))
            edge_index, _ = builder.build_knn_graph(self.coords_train)
            edge_index = edge_index.to(self.device)
            
            # モデル作成
            model = KNNGraphGNN(
                input_dim=self.X_train.shape[1],
                hidden_dim=params['hidden_dim'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
            ).to(self.device)
        else:
            # MLPベースライン
            edge_index = None
            model = nn.Sequential(
                nn.Linear(self.X_train.shape[1], params['hidden_dim']),
                nn.ReLU(),
                nn.Dropout(params['dropout']),
                nn.Linear(params['hidden_dim'], params['hidden_dim'] // 2),
                nn.ReLU(),
                nn.Dropout(params['dropout']),
                nn.Linear(params['hidden_dim'] // 2, 1),
            ).to(self.device)
        
        # データ準備
        X_train_t = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        y_train_t = torch.tensor(self.y_train, dtype=torch.float32).unsqueeze(1).to(self.device)
        X_val_t = torch.tensor(self.X_val, dtype=torch.float32).to(self.device)
        
        # オプティマイザ・損失関数
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = FocalLoss(alpha=params['focal_alpha'], gamma=params['focal_gamma'])
        
        # 学習
        best_val_auc = 0.0
        patience_counter = 0
        patience = 10
        
        for epoch in range(self.n_epochs):
            model.train()
            optimizer.zero_grad()
            
            if edge_index is not None:
                outputs = model(X_train_t, edge_index)
            else:
                outputs = model(X_train_t)
            
            loss = criterion(outputs, y_train_t)
            loss.backward()
            optimizer.step()
            
            # 検証
            model.eval()
            with torch.no_grad():
                if edge_index is not None:
                    val_outputs = model(X_val_t, edge_index)
                else:
                    val_outputs = model(X_val_t)
                val_preds = torch.sigmoid(val_outputs).cpu().numpy().flatten()
            
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(self.y_val, val_preds)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            # Pruning
            trial = optuna.trial.FixedTrial(params)
        
        # メモリ解放
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return best_val_auc


def run_optuna_search(
    data_dir: str = "data/spatio_temporal",
    output_dir: str = "results/spatio_temporal/optuna",
    model_type: str = "knn_gnn",
    n_trials: int = 50,
    n_epochs: int = 50,
):
    """Optuna探索の実行"""
    
    print("=" * 70)
    print("Optuna ハイパーパラメータ探索")
    print(f"モデル: {model_type}")
    print(f"試行数: {n_trials}")
    print("=" * 70)
    
    # 出力ディレクトリ
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # データ読み込み
    print("\n📂 データ読み込み中...")
    train_df = pd.read_parquet(Path(data_dir) / "preprocessed_train.parquet")
    val_df = pd.read_parquet(Path(data_dir) / "preprocessed_val.parquet")
    
    # 特徴量準備
    exclude_cols = ['fatal', 'date', 'lat', 'lon', 'geohash', 'geohash_fine', 'year']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols]
    
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df['fatal'].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df['fatal'].values.astype(np.float32)
    
    # 座標
    coords_train = train_df[['lat', 'lon']].values
    
    # NaN処理
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0)
    
    print(f"   Train: {len(X_train):,}, Val: {len(X_val):,}")
    print(f"   特徴量: {X_train.shape[1]}")
    
    # 目的関数
    objective = OptunaObjective(
        X_train, y_train, X_val, y_val,
        coords_train, model_type, n_epochs
    )
    
    # Study作成
    study = optuna.create_study(
        direction='maximize',
        study_name=f'spatio_temporal_{model_type}',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    
    # 最適化実行
    print("\n🔍 探索開始...")
    study.optimize(
        objective,
        n_trials=n_trials,
        show_progress_bar=True,
        gc_after_trial=True,
    )
    
    # 結果出力
    print("\n" + "=" * 70)
    print("📊 探索結果")
    print("=" * 70)
    
    print(f"\n🏆 Best Trial:")
    print(f"   Value (Val AUC): {study.best_trial.value:.4f}")
    print(f"   Params: {study.best_trial.params}")
    
    # 結果保存
    results = {
        'best_value': study.best_trial.value,
        'best_params': study.best_trial.params,
        'n_trials': n_trials,
        'model_type': model_type,
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_dir / f"optuna_results_{model_type}.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # 全トライアル結果
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output_dir / f"optuna_trials_{model_type}.csv", index=False)
    
    print(f"\n📄 結果保存: {output_dir}")
    
    return study.best_trial.params


def main():
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Search")
    parser.add_argument('--data-dir', type=str, default="data/spatio_temporal")
    parser.add_argument('--output-dir', type=str, default="results/spatio_temporal/optuna")
    parser.add_argument('--model', type=str, default="knn_gnn",
                        choices=['knn_gnn', 'mlp'])
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--n-epochs', type=int, default=50)
    
    args = parser.parse_args()
    
    run_optuna_search(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model,
        n_trials=args.n_trials,
        n_epochs=args.n_epochs,
    )


if __name__ == "__main__":
    main()
