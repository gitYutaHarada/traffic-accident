"""
チェックポイント・再開機能
=======================
"""

import os
import json
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class CheckpointManager:
    """
    学習チェックポイントの管理
    
    - モデルの状態保存
    - オプティマイザの状態保存
    - 学習ステータスの保存
    - 中断からの再開
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.status_file = self.checkpoint_dir / "training_status.json"
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        step: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        scheduler: Optional[Any] = None,
        is_best: bool = False,
    ):
        """チェックポイントの保存"""
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': config,
            'timestamp': datetime.now().isoformat(),
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # 通常のチェックポイント
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # ベストモデル
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
        
        # 最新のチェックポイント
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
        
        # ステータスファイル更新
        self._update_status(epoch, step, metrics)
        
        # 古いチェックポイントを削除
        self._cleanup_old_checkpoints()
        
        print(f"   💾 チェックポイント保存: epoch={epoch}, step={step}")
    
    def load_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """チェックポイントの読み込み"""
        
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_dir / "latest_checkpoint.pt"
        else:
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            print("   ℹ️ チェックポイントが見つかりません。新規開始します。")
            return {'epoch': 0, 'step': 0, 'metrics': {}}
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # モデル状態の復元
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # オプティマイザ状態の復元
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # スケジューラ状態の復元
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"   📂 チェックポイント読み込み: epoch={checkpoint['epoch']}")
        
        return checkpoint
    
    def load_best_model(self, model: torch.nn.Module) -> Dict[str, Any]:
        """ベストモデルの読み込み"""
        best_path = self.checkpoint_dir / "best_model.pt"
        
        if not best_path.exists():
            print("   ⚠️ ベストモデルが見つかりません")
            return {}
        
        checkpoint = torch.load(best_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"   📂 ベストモデル読み込み: epoch={checkpoint['epoch']}")
        
        return checkpoint
    
    def get_resume_info(self) -> Dict[str, Any]:
        """再開情報の取得"""
        if not self.status_file.exists():
            return {'should_resume': False, 'epoch': 0, 'step': 0}
        
        with open(self.status_file, 'r') as f:
            status = json.load(f)
        
        return {
            'should_resume': True,
            'epoch': status.get('epoch', 0),
            'step': status.get('step', 0),
            'metrics': status.get('metrics', {}),
        }
    
    def _update_status(self, epoch: int, step: int, metrics: Dict):
        """ステータスファイルの更新"""
        status = {
            'epoch': epoch,
            'step': step,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(self.status_file, 'w') as f:
            json.dump(status, f, indent=2)
    
    def _cleanup_old_checkpoints(self):
        """古いチェックポイントの削除"""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for old_ckpt in checkpoints[self.max_checkpoints:]:
            old_ckpt.unlink()
    
    def clear_checkpoints(self):
        """全チェックポイントの削除"""
        for f in self.checkpoint_dir.glob("*.pt"):
            f.unlink()
        if self.status_file.exists():
            self.status_file.unlink()
        print("   🗑️ チェックポイントをクリア")


class EarlyStopping:
    """
    Early Stopping
    
    検証損失が改善しない場合に学習を停止
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0001,
        mode: str = 'min',  # 'min' or 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        """
        値を評価してストップするか判定
        
        Returns:
            True: 改善あり（is_best）
            False: 改善なし
        """
        if self.best_value is None:
            self.best_value = value
            return True
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False
    
    def reset(self):
        self.counter = 0
        self.best_value = None
        self.should_stop = False


def set_seed(seed: int = 42):
    """ランダムシードの固定"""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"   🎲 ランダムシード設定: {seed}")
