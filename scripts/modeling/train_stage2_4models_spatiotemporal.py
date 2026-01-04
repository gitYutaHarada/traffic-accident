"""
Spatio-Temporal 4-Model Ensemble Training Script
=================================================
時空間特徴量（Geohash履歴 + 時間サイクル）を使用した4モデルアンサンブル学習。

特徴:
- LightGBM, CatBoost, MLP, TabNet の4モデルをアンサンブル
- チェックポイント対応（途中から再開可能）
- Intel Core Ultra 9 285K / 64GB RAM 最大活用
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.optimize import minimize
import joblib
import gc

warnings.filterwarnings('ignore')

# ========================================
# 環境最適化設定 (Intel Core Ultra 9 285K - P-core優先)
# ========================================
# Arrow LakeはP-core/E-coreのハイブリッド構成。
# E-coreに重い計算が割り当てられると遅くなるため、
# P-core数(8)に合わせてスレッド数を設定。
N_CORES = 24  # Intel Core Ultra 9 285K (全コア)
N_JOBS = 8    # P-coreのみ使用（E-coreへのオーバーフローを防止）
os.environ['OMP_NUM_THREADS'] = str(N_JOBS)
os.environ['MKL_NUM_THREADS'] = str(N_JOBS)
os.environ['OPENBLAS_NUM_THREADS'] = str(N_JOBS)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(N_JOBS)
os.environ['NUMEXPR_NUM_THREADS'] = str(N_JOBS)

# PyTorch設定
import torch
torch.set_num_threads(N_JOBS)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ランダムシード
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ========================================
# ライブラリのインポート (オプショナル)
# ========================================
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("⚠️ LightGBM not available")

try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available")

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("⚠️ TabNet not available")

print(f"🚀 Device: {DEVICE}")
print(f"🧵 Threads: {N_JOBS}")


class SpatioTemporalEnsemble:
    """時空間特徴量を用いた4モデルアンサンブル"""
    
    def __init__(
        self,
        data_dir: str = "data/spatio_temporal",
        output_dir: str = "results/spatio_temporal_ensemble",
        n_folds: int = 5,
        random_state: int = RANDOM_SEED,
        force_retrain: bool = False,  # Trueの場合、チェックポイントを無視して再学習
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.n_folds = n_folds
        self.random_state = random_state
        self.force_retrain = force_retrain
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # データ格納
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.feature_cols = None
        self.cat_cols = None
        self.num_cols = None
        self.target_col = "fatal"
        
        # 予測格納
        self.oof_predictions = {}
        self.test_predictions = {}
        self.model_aucs = {}
        
        print("=" * 70)
        print("🚀 Spatio-Temporal 4-Model Ensemble")
        print(f"   Data: {self.data_dir}")
        print(f"   Output: {self.output_dir}")
        print(f"   Checkpoints: {self.checkpoint_dir}")
        print(f"   Folds: {n_folds}, Seed: {random_state}")
        print("=" * 70)
    
    # ========================================
    # チェックポイント管理
    # ========================================
    def _ckpt_path(self, name: str) -> Path:
        return self.checkpoint_dir / f"{name}.npy"
    
    def _model_ckpt_path(self, model_name: str, fold: int) -> Path:
        return self.checkpoint_dir / f"{model_name}_fold{fold}.joblib"
    
    def _load_oof_checkpoint(self, model_name: str) -> Optional[np.ndarray]:
        if self.force_retrain:
            return None
        path = self._ckpt_path(f"{model_name}_oof")
        if path.exists():
            return np.load(path)
        return None
    
    def _load_test_checkpoint(self, model_name: str) -> Optional[np.ndarray]:
        if self.force_retrain:
            return None
        path = self._ckpt_path(f"{model_name}_test")
        if path.exists():
            return np.load(path)
        return None
    
    def _save_checkpoint(self, model_name: str, oof: np.ndarray, test: np.ndarray):
        np.save(self._ckpt_path(f"{model_name}_oof"), oof)
        np.save(self._ckpt_path(f"{model_name}_test"), test)
        print(f"   💾 {model_name} チェックポイント保存完了")
    
    def _load_fold_checkpoint(self, model_name: str, fold: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if self.force_retrain:
            return None, None
        oof_path = self.checkpoint_dir / f"{model_name}_fold{fold}_oof.npy"
        test_path = self.checkpoint_dir / f"{model_name}_fold{fold}_test.npy"
        if oof_path.exists() and test_path.exists():
            return np.load(oof_path), np.load(test_path)
        return None, None
    
    def _save_fold_checkpoint(self, model_name: str, fold: int, oof: np.ndarray, test: np.ndarray):
        np.save(self.checkpoint_dir / f"{model_name}_fold{fold}_oof.npy", oof)
        np.save(self.checkpoint_dir / f"{model_name}_fold{fold}_test.npy", test)
    
    # ========================================
    # データ読み込み
    # ========================================
    def load_data(self):
        """データ読み込み"""
        print("\n📂 データ読み込み中...")
        
        # 生データ（GBDT用）を読み込み
        train_path = self.data_dir / "raw_train.parquet"
        val_path = self.data_dir / "raw_val.parquet"
        test_path = self.data_dir / "raw_test.parquet"
        
        if not train_path.exists():
            raise FileNotFoundError(
                f"データが見つかりません: {train_path}\n"
                "先に preprocess_spatio_temporal.py を実行してください。"
            )
        
        self.train_df = pd.read_parquet(train_path)
        self.val_df = pd.read_parquet(val_path)
        self.test_df = pd.read_parquet(test_path)
        
        # 【ID Propagation】元のインデックスを保持
        self.train_df['original_index'] = self.train_df.index
        self.val_df['original_index'] = self.val_df.index
        self.test_df['original_index'] = self.test_df.index
        
        # Train + Val を学習用に統合（ignore_index=Trueでも、original_indexカラムは保持される）
        self.full_train_df = pd.concat([self.train_df, self.val_df], ignore_index=True)
        
        print(f"   Train: {len(self.train_df):,} 行")
        print(f"   Val:   {len(self.val_df):,} 行")
        print(f"   Test:  {len(self.test_df):,} 行")
        print(f"   Train+Val (学習用): {len(self.full_train_df):,} 行")
        
        # 特徴量列の特定
        self._identify_columns()
    
    def _identify_columns(self):
        """特徴量列の特定"""
        exclude_cols = [
            self.target_col, 'lat', 'lon', 'geohash', 'geohash_fine', 
            'date', 'year', 'accident_id'
        ]
        
        self.feature_cols = [c for c in self.full_train_df.columns if c not in exclude_cols]
        
        # カテゴリ列と数値列を分類
        self.cat_cols = []
        self.num_cols = []
        
        for col in self.feature_cols:
            if self.full_train_df[col].dtype == 'object' or self.full_train_df[col].nunique() < 50:
                self.cat_cols.append(col)
            else:
                self.num_cols.append(col)
        
        print(f"   特徴量数: {len(self.feature_cols)}")
        print(f"   カテゴリ: {len(self.cat_cols)}, 数値: {len(self.num_cols)}")
        
        # 時空間特徴量の確認
        spatiotemporal_cols = [c for c in self.feature_cols if 'past_' in c or '_sin' in c or '_cos' in c]
        print(f"   時空間特徴量: {len(spatiotemporal_cols)}")
    
    # ========================================
    # LightGBM
    # ========================================
    def train_lightgbm(self):
        """LightGBM 学習"""
        model_name = "lgbm"
        
        # チェックポイント確認
        oof = self._load_oof_checkpoint(model_name)
        test = self._load_test_checkpoint(model_name)
        if oof is not None and test is not None:
            print(f"\n✅ {model_name} チェックポイントから復元")
            self.oof_predictions[model_name] = oof
            self.test_predictions[model_name] = test
            self.model_aucs[model_name] = roc_auc_score(self.full_train_df[self.target_col], oof)
            print(f"   OOF AUC: {self.model_aucs[model_name]:.4f}")
            return
        
        if not LGBM_AVAILABLE:
            print(f"⚠️ {model_name} スキップ (ライブラリなし)")
            return
        
        print(f"\n🌲 {model_name} 学習中...")
        
        X = self.full_train_df[self.feature_cols].copy()
        y = self.full_train_df[self.target_col].values
        X_test = self.test_df[self.feature_cols].copy()
        
        # カテゴリ変数をcategoryに変換
        for col in self.cat_cols:
            X[col] = X[col].astype('category')
            X_test[col] = X_test[col].astype('category')
        
        # クラス重み計算
        pos_weight = (len(y) - y.sum()) / y.sum()
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 127,
            'max_depth': 8,
            'min_child_samples': 100,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'scale_pos_weight': pos_weight,
            'n_jobs': N_JOBS,
            'seed': self.random_state,
            'verbose': -1,
        }
        
        oof = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            # Foldチェックポイント確認
            fold_oof, fold_test = self._load_fold_checkpoint(model_name, fold)
            if fold_oof is not None:
                oof[val_idx] = fold_oof
                test_preds += fold_test / self.n_folds
                print(f"   Fold {fold+1}/{self.n_folds} ✅ チェックポイントから復元")
                continue
            
            print(f"   Fold {fold+1}/{self.n_folds}...")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=self.cat_cols)
            val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=self.cat_cols, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=2000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
            )
            
            fold_oof_pred = model.predict(X_val)
            fold_test_pred = model.predict(X_test)
            
            oof[val_idx] = fold_oof_pred
            test_preds += fold_test_pred / self.n_folds
            
            # Foldチェックポイント保存
            self._save_fold_checkpoint(model_name, fold, fold_oof_pred, fold_test_pred)
            
            auc = roc_auc_score(y_val, fold_oof_pred)
            print(f"      Fold {fold+1} AUC: {auc:.4f}")
            
            del model, train_data, val_data
            gc.collect()
        
        self.oof_predictions[model_name] = oof
        self.test_predictions[model_name] = test_preds
        self.model_aucs[model_name] = roc_auc_score(y, oof)
        
        self._save_checkpoint(model_name, oof, test_preds)
        print(f"   {model_name} OOF AUC: {self.model_aucs[model_name]:.4f}")
    
    # ========================================
    # CatBoost
    # ========================================
    def train_catboost(self):
        """CatBoost 学習"""
        model_name = "catboost"
        
        # チェックポイント確認
        oof = self._load_oof_checkpoint(model_name)
        test = self._load_test_checkpoint(model_name)
        if oof is not None and test is not None:
            print(f"\n✅ {model_name} チェックポイントから復元")
            self.oof_predictions[model_name] = oof
            self.test_predictions[model_name] = test
            self.model_aucs[model_name] = roc_auc_score(self.full_train_df[self.target_col], oof)
            print(f"   OOF AUC: {self.model_aucs[model_name]:.4f}")
            return
        
        if not CATBOOST_AVAILABLE:
            print(f"⚠️ {model_name} スキップ (ライブラリなし)")
            return
        
        print(f"\n🐱 {model_name} 学習中...")
        
        X = self.full_train_df[self.feature_cols].copy()
        y = self.full_train_df[self.target_col].values
        X_test = self.test_df[self.feature_cols].copy()
        
        # カテゴリ変数を文字列に変換
        for col in self.cat_cols:
            X[col] = X[col].astype(str).fillna('_missing')
            X_test[col] = X_test[col].astype(str).fillna('_missing')
        
        # カテゴリインデックス
        cat_features_idx = [X.columns.get_loc(c) for c in self.cat_cols]
        
        oof = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            # Foldチェックポイント確認
            fold_oof, fold_test = self._load_fold_checkpoint(model_name, fold)
            if fold_oof is not None:
                oof[val_idx] = fold_oof
                test_preds += fold_test / self.n_folds
                print(f"   Fold {fold+1}/{self.n_folds} ✅ チェックポイントから復元")
                continue
            
            print(f"   Fold {fold+1}/{self.n_folds}...")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = CatBoostClassifier(
                iterations=2000,
                learning_rate=0.05,
                depth=8,
                l2_leaf_reg=3,
                loss_function='Logloss',
                eval_metric='AUC',
                random_seed=self.random_state,
                early_stopping_rounds=50,
                verbose=100,
                thread_count=N_JOBS,
                auto_class_weights='Balanced',
            )
            
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                cat_features=cat_features_idx,
                use_best_model=True,
            )
            
            fold_oof_pred = model.predict_proba(X_val)[:, 1]
            fold_test_pred = model.predict_proba(X_test)[:, 1]
            
            oof[val_idx] = fold_oof_pred
            test_preds += fold_test_pred / self.n_folds
            
            # Foldチェックポイント保存
            self._save_fold_checkpoint(model_name, fold, fold_oof_pred, fold_test_pred)
            
            auc = roc_auc_score(y_val, fold_oof_pred)
            print(f"      Fold {fold+1} AUC: {auc:.4f}")
            
            del model
            gc.collect()
        
        self.oof_predictions[model_name] = oof
        self.test_predictions[model_name] = test_preds
        self.model_aucs[model_name] = roc_auc_score(y, oof)
        
        self._save_checkpoint(model_name, oof, test_preds)
        print(f"   {model_name} OOF AUC: {self.model_aucs[model_name]:.4f}")
    
    # ========================================
    # MLP (データリーク対策済み)
    # ========================================
    def train_mlp(self):
        """MLP 学習 (Fold内で前処理を行いリーク防止)"""
        model_name = "mlp"
        
        # チェックポイント確認
        oof = self._load_oof_checkpoint(model_name)
        test = self._load_test_checkpoint(model_name)
        if oof is not None and test is not None:
            print(f"\n✅ {model_name} チェックポイントから復元")
            self.oof_predictions[model_name] = oof
            self.test_predictions[model_name] = test
            self.model_aucs[model_name] = roc_auc_score(self.full_train_df[self.target_col], oof)
            print(f"   OOF AUC: {self.model_aucs[model_name]:.4f}")
            return
        
        print(f"\n🧠 {model_name} 学習中...")
        
        # 生データを保持（前処理はFold内で行う）
        X_raw = self.full_train_df[self.feature_cols].copy()
        y = self.full_train_df[self.target_col].values
        X_test_raw = self.test_df[self.feature_cols].copy()
        
        oof = np.zeros(len(X_raw))
        test_preds = np.zeros(len(X_test_raw))
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
            # Foldチェックポイント確認
            fold_oof, fold_test = self._load_fold_checkpoint(model_name, fold)
            if fold_oof is not None:
                oof[val_idx] = fold_oof
                test_preds += fold_test / self.n_folds
                print(f"   Fold {fold+1}/{self.n_folds} ✅ チェックポイントから復元")
                continue
            
            print(f"   Fold {fold+1}/{self.n_folds}...")
            
            # ========================================
            # Fold内で前処理（高速化・適正化版）
            # ========================================
            from sklearn.preprocessing import OrdinalEncoder
            from sklearn.impute import SimpleImputer
            
            X_train_fold = X_raw.iloc[train_idx].copy()
            X_val_fold = X_raw.iloc[val_idx].copy()
            X_test_fold = X_test_raw.copy()
            y_train, y_val = y[train_idx], y[val_idx]
            
            # 1. カテゴリ変数の処理 (OrdinalEncoderで高速化)
            # 文字列化と欠損埋め
            X_train_fold[self.cat_cols] = X_train_fold[self.cat_cols].astype(str).fillna('_missing')
            X_val_fold[self.cat_cols] = X_val_fold[self.cat_cols].astype(str).fillna('_missing')
            X_test_fold[self.cat_cols] = X_test_fold[self.cat_cols].astype(str).fillna('_missing')
            
            # OrdinalEncoder: 未知カテゴリは -1 に設定
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_train_fold[self.cat_cols] = oe.fit_transform(X_train_fold[self.cat_cols])
            X_val_fold[self.cat_cols] = oe.transform(X_val_fold[self.cat_cols])
            X_test_fold[self.cat_cols] = oe.transform(X_test_fold[self.cat_cols])
            
            # -1 (unknown) を 0 に置換し、他を +1 シフト
            for col in self.cat_cols:
                X_train_fold[col] = X_train_fold[col] + 1
                X_val_fold[col] = X_val_fold[col] + 1
                X_test_fold[col] = X_test_fold[col] + 1
            
            # 2. 数値変数の欠損埋め (平均値) & Scaling
            imputer = SimpleImputer(strategy='mean')
            scaler = StandardScaler()
            
            # Impute (Trainでfit) -> Scale (Trainでfit)
            X_train_num = imputer.fit_transform(X_train_fold[self.num_cols])
            X_val_num = imputer.transform(X_val_fold[self.num_cols])
            X_test_num = imputer.transform(X_test_fold[self.num_cols])
            
            X_train_fold[self.num_cols] = scaler.fit_transform(X_train_num)
            X_val_fold[self.num_cols] = scaler.transform(X_val_num)
            X_test_fold[self.num_cols] = scaler.transform(X_test_num)
            
            X_train_np = X_train_fold.values.astype(np.float32)
            X_val_np = X_val_fold.values.astype(np.float32)
            X_test_np = X_test_fold.values.astype(np.float32)
            
            # PyTorch MLP
            fold_oof_pred, fold_test_pred = self._train_mlp_fold(
                X_train_np, y_train, X_val_np, y_val, X_test_np, fold
            )
            
            oof[val_idx] = fold_oof_pred
            test_preds += fold_test_pred / self.n_folds
            
            # Foldチェックポイント保存
            self._save_fold_checkpoint(model_name, fold, fold_oof_pred, fold_test_pred)
            
            auc = roc_auc_score(y_val, fold_oof_pred)
            print(f"      Fold {fold+1} AUC: {auc:.4f}")
        
        self.oof_predictions[model_name] = oof
        self.test_predictions[model_name] = test_preds
        self.model_aucs[model_name] = roc_auc_score(y, oof)
        
        self._save_checkpoint(model_name, oof, test_preds)
        print(f"   {model_name} OOF AUC: {self.model_aucs[model_name]:.4f}")
    
    def _train_mlp_fold(self, X_train, y_train, X_val, y_val, X_test, fold):
        """MLP 1 Foldの学習"""
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        # データをTensorに変換
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        
        train_ds = TensorDataset(X_train_t, y_train_t)
        val_ds = TensorDataset(X_val_t, y_val_t)
        
        # バッチサイズ: 8192→1024に縮小（汎化性能向上）
        batch_size = 1024
        # 【高速化】E-core (16個) 活用でデータ読み込み並列化
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
        
        # モデル定義
        input_dim = X_train.shape[1]
        
        class MLP(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(256, 64),
                    nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.Linear(64, 1)  # BCEWithLogitsLossがSigmoidを適用するため、ここにSigmoidは不要
                )
            
            def forward(self, x):
                return self.layers(x)
        
        model = MLP(input_dim).to(DEVICE)
        
        # クラス重み
        pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()]).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
        
        best_auc = 0
        patience = 15
        wait = 0
        best_state = None
        
        for epoch in range(100):
            model.train()
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            val_preds = []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(DEVICE)
                    logits = model(batch_x)
                    probs = torch.sigmoid(logits)  # Sigmoid適用
                    val_preds.extend(probs.cpu().numpy().flatten())
            
            val_auc = roc_auc_score(y_val, val_preds)
            scheduler.step(val_auc)
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict().copy()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break
        
        # ベストモデルで予測
        model.load_state_dict(best_state)
        model.eval()
        
        with torch.no_grad():
            # Sigmoid適用（モデルはLogitsを出力するため）
            logits = model(X_val_t.to(DEVICE))
            val_preds = torch.sigmoid(logits).cpu().numpy().flatten()
            
            # テスト予測
            test_preds = []
            test_loader = DataLoader(TensorDataset(X_test_t), batch_size=batch_size*2, shuffle=False)
            for batch in test_loader:
                batch_x = batch[0].to(DEVICE)
                logits = model(batch_x)
                probs = torch.sigmoid(logits)
                test_preds.extend(probs.cpu().numpy().flatten())
            test_preds = np.array(test_preds)
        
        return val_preds, test_preds
    
    # ========================================
    # TabNet (データリーク対策済み)
    # ========================================
    def train_tabnet(self):
        """TabNet 学習 (Fold内で前処理を行いリーク防止)"""
        model_name = "tabnet"
        
        # チェックポイント確認
        oof = self._load_oof_checkpoint(model_name)
        test = self._load_test_checkpoint(model_name)
        if oof is not None and test is not None:
            print(f"\n✅ {model_name} チェックポイントから復元")
            self.oof_predictions[model_name] = oof
            self.test_predictions[model_name] = test
            self.model_aucs[model_name] = roc_auc_score(self.full_train_df[self.target_col], oof)
            print(f"   OOF AUC: {self.model_aucs[model_name]:.4f}")
            return
        
        if not TABNET_AVAILABLE:
            print(f"⚠️ {model_name} スキップ (ライブラリなし)")
            return
        
        print(f"\n📊 {model_name} 学習中...")
        
        # 生データを保持（前処理はFold内で行う）
        X_raw = self.full_train_df[self.feature_cols].copy()
        y = self.full_train_df[self.target_col].values
        X_test_raw = self.test_df[self.feature_cols].copy()
        
        oof = np.zeros(len(X_raw))
        test_preds = np.zeros(len(X_test_raw))
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_raw, y)):
            # Foldチェックポイント確認
            fold_oof, fold_test = self._load_fold_checkpoint(model_name, fold)
            if fold_oof is not None:
                oof[val_idx] = fold_oof
                test_preds += fold_test / self.n_folds
                print(f"   Fold {fold+1}/{self.n_folds} ✅ チェックポイントから復元")
                continue
            
            print(f"   Fold {fold+1}/{self.n_folds}...")
            
            # ========================================
            # Fold内で前処理（リーク防止）
            # ========================================
            X_train_fold = X_raw.iloc[train_idx].copy()
            X_val_fold = X_raw.iloc[val_idx].copy()
            X_test_fold = X_test_raw.copy()
            y_train, y_val = y[train_idx], y[val_idx]
            
            # LabelEncoding + cat_idxs/cat_dims計算 (Train基準)
            # 【高速化】辞書マッピングによるベクトル化エンコーディング
            cat_idxs = []
            cat_dims = []
            
            for i, col in enumerate(self.feature_cols):
                if col in self.cat_cols:
                    # 文字列変換
                    train_vals = X_train_fold[col].astype(str).fillna('_missing')
                    val_vals = X_val_fold[col].astype(str).fillna('_missing')
                    test_vals = X_test_fold[col].astype(str).fillna('_missing')
                    
                    # マッパー作成 (1-based index, 0=unknown)
                    unique_vals = sorted(set(train_vals))
                    mapper = {v: idx + 1 for idx, v in enumerate(unique_vals)}
                    
                    # mapで一括変換 (存在しないキーはNaNになる→0で埋める)
                    X_train_fold[col] = train_vals.map(mapper).fillna(0).astype(int)
                    X_val_fold[col] = val_vals.map(mapper).fillna(0).astype(int)
                    X_test_fold[col] = test_vals.map(mapper).fillna(0).astype(int)
                    
                    cat_idxs.append(i)
                    cat_dims.append(len(unique_vals) + 1)  # +1 for unknown (0)
            
            # StandardScaler (Train基準)
            scaler = StandardScaler()
            X_train_fold[self.num_cols] = scaler.fit_transform(X_train_fold[self.num_cols].fillna(0))
            X_val_fold[self.num_cols] = scaler.transform(X_val_fold[self.num_cols].fillna(0))
            X_test_fold[self.num_cols] = scaler.transform(X_test_fold[self.num_cols].fillna(0))
            
            X_train_np = X_train_fold.values.astype(np.float32)
            X_val_np = X_val_fold.values.astype(np.float32)
            X_test_np = X_test_fold.values.astype(np.float32)
            
            model = TabNetClassifier(
                cat_idxs=cat_idxs,
                cat_dims=cat_dims,
                cat_emb_dim=2,
                n_d=32,
                n_a=32,
                n_steps=5,
                gamma=1.5,
                lambda_sparse=1e-4,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=2e-2),
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                scheduler_params=dict(step_size=10, gamma=0.9),
                mask_type='entmax',
                seed=self.random_state,
                device_name=DEVICE,
                verbose=0,
            )
            
            model.fit(
                X_train_np, y_train,
                eval_set=[(X_val_np, y_val)],
                eval_metric=['auc'],
                max_epochs=200,
                patience=20,
                batch_size=1024,  # 縮小（汎化性能向上）
                virtual_batch_size=256,
            )
            
            fold_oof_pred = model.predict_proba(X_val_np)[:, 1]
            fold_test_pred = model.predict_proba(X_test_np)[:, 1]
            
            # 【安定性対策】NaNチェックとフォールバック
            if np.isnan(fold_oof_pred).any():
                print(f"      ⚠️ Warning: TabNet produced NaN in Fold {fold+1}. Fallback to mean.")
                fold_oof_pred = np.nan_to_num(fold_oof_pred, nan=np.mean(y_train))
            if np.isnan(fold_test_pred).any():
                fold_test_pred = np.nan_to_num(fold_test_pred, nan=np.mean(y_train))
            
            oof[val_idx] = fold_oof_pred
            test_preds += fold_test_pred / self.n_folds
            
            # Foldチェックポイント保存
            self._save_fold_checkpoint(model_name, fold, fold_oof_pred, fold_test_pred)
            
            auc = roc_auc_score(y_val, fold_oof_pred)
            print(f"      Fold {fold+1} AUC: {auc:.4f}")
            
            del model
            gc.collect()
            torch.cuda.empty_cache() if DEVICE == 'cuda' else None
        
        self.oof_predictions[model_name] = oof
        self.test_predictions[model_name] = test_preds
        self.model_aucs[model_name] = roc_auc_score(y, oof)
        
        self._save_checkpoint(model_name, oof, test_preds)
        print(f"   {model_name} OOF AUC: {self.model_aucs[model_name]:.4f}")
    
    # ========================================
    # アンサンブル
    # ========================================
    def optimize_ensemble(self):
        """アンサンブル重みの最適化"""
        print("\n⚖️ アンサンブル重み最適化中...")
        
        if len(self.oof_predictions) < 2:
            print("   ⚠️ モデル数が不足しています")
            return
        
        y = self.full_train_df[self.target_col].values
        model_names = list(self.oof_predictions.keys())
        oof_matrix = np.column_stack([self.oof_predictions[m] for m in model_names])
        
        def neg_auc(weights):
            weights = weights / weights.sum()
            pred = oof_matrix @ weights
            return -roc_auc_score(y, pred)
        
        n_models = len(model_names)
        initial_weights = np.ones(n_models) / n_models
        bounds = [(0, 1)] * n_models
        constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
        
        result = minimize(neg_auc, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        optimal_weights = result.x / result.x.sum()
        
        print("   最適化された重み:")
        for name, weight in zip(model_names, optimal_weights):
            print(f"     {name}: {weight:.4f}")
        
        # アンサンブル予測
        oof_ensemble = oof_matrix @ optimal_weights
        test_matrix = np.column_stack([self.test_predictions[m] for m in model_names])
        test_ensemble = test_matrix @ optimal_weights
        
        self.oof_predictions['ensemble'] = oof_ensemble
        self.test_predictions['ensemble'] = test_ensemble
        self.model_aucs['ensemble'] = roc_auc_score(y, oof_ensemble)
        
        print(f"   Ensemble OOF AUC: {self.model_aucs['ensemble']:.4f}")
        
        # 結果保存
        self._save_results(model_names, optimal_weights)
    
    def _save_results(self, model_names: List[str], weights: np.ndarray):
        """結果の保存"""
        print("\n📈 結果保存中...")
        
        y = self.full_train_df[self.target_col].values
        
        # スコアCSV
        results = []
        for model in list(self.oof_predictions.keys()):
            oof_auc = roc_auc_score(y, self.oof_predictions[model])
            oof_prauc = average_precision_score(y, self.oof_predictions[model])
            test_auc = roc_auc_score(self.test_df[self.target_col], self.test_predictions[model])
            test_prauc = average_precision_score(self.test_df[self.target_col], self.test_predictions[model])
            
            results.append({
                'model': model,
                'oof_auc': oof_auc,
                'oof_prauc': oof_prauc,
                'test_auc': test_auc,
                'test_prauc': test_prauc,
            })
            print(f"   {model}: OOF AUC={oof_auc:.4f}, Test AUC={test_auc:.4f}")
        
        pd.DataFrame(results).to_csv(self.output_dir / "final_scores.csv", index=False)
        
        # OOF予測CSV（original_indexを含む）
        oof_df = pd.DataFrame(self.oof_predictions)
        oof_df['original_index'] = self.full_train_df['original_index'].values
        oof_df['target'] = y
        oof_df.to_csv(self.output_dir / "oof_predictions.csv", index=False)
        
        # Test予測CSV（original_indexを含む）
        test_df = pd.DataFrame(self.test_predictions)
        test_df['original_index'] = self.test_df['original_index'].values
        test_df.to_csv(self.output_dir / "test_predictions.csv", index=False)
        
        # 重みJSON
        weight_dict = {name: float(w) for name, w in zip(model_names, weights)}
        with open(self.output_dir / "ensemble_weights.json", 'w') as f:
            json.dump(weight_dict, f, indent=2)
        
        print(f"   ✅ 完了: {self.output_dir}")
    
    # ========================================
    # 実行
    # ========================================
    def run(self):
        """全工程実行"""
        start_time = datetime.now()
        
        # 1. データ読み込み
        self.load_data()
        
        # 2. 各モデル学習
        self.train_lightgbm()
        self.train_catboost()
        self.train_mlp()
        self.train_tabnet()
        
        # 3. アンサンブル
        self.optimize_ensemble()
        
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        
        print("\n" + "=" * 70)
        print(f"✅ 全工程完了！ 実行時間: {elapsed:.1f}分")
        print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Spatio-Temporal 4-Model Ensemble Training")
    parser.add_argument(
        '--force-retrain',
        action='store_true',
        help='チェックポイントを無視して全モデルを再学習'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default="data/spatio_temporal",
        help='前処理済みデータのディレクトリ'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="results/spatio_temporal_ensemble",
        help='結果出力ディレクトリ'
    )
    args = parser.parse_args()
    
    ensemble = SpatioTemporalEnsemble(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_folds=5,
        random_state=42,
        force_retrain=args.force_retrain,
    )
    ensemble.run()


if __name__ == "__main__":
    main()
