"""
Two-Stage Spatio-Temporal 4-Model Ensemble Training Script
==========================================================
Stage 1 フィルタリング + 時空間特徴量を組み合わせた最強のモデル。

特徴:
- Stage 1 OOF予測によるフィルタリング（高難易度データに特化）
- 時空間特徴量（Geohash履歴 + 時間サイクル）
- LightGBM, CatBoost, MLP, TabNet の4モデルアンサンブル
- 堅牢なチェックポイント機能（Fold単位・モデル単位で再開可能）
- Intel Core Ultra 9 285K / 64GB RAM 最大活用

使用法:
    python scripts/modeling/train_stage2_4models_spatiotemporal_twostage.py
    
    # チェックポイントから再開
    python scripts/modeling/train_stage2_4models_spatiotemporal_twostage.py --resume
    
    # 強制的に最初から再学習
    python scripts/modeling/train_stage2_4models_spatiotemporal_twostage.py --force-retrain
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.optimize import minimize
import joblib
import gc

warnings.filterwarnings('ignore')

# ========================================
# 環境最適化設定 (Intel Core Ultra 9 285K)
# ========================================
# Arrow Lake: 8 P-cores + 16 E-cores = 24 cores / 24 threads
# P-coreのみを使用してパフォーマンスを最大化
# E-coreはOSやバックグラウンドタスクに任せる
N_CORES = 24  # Intel Core Ultra 9 285K (全コア)
N_JOBS = 8    # P-coreのみ使用（E-coreへのオーバーフローを防止）

# スレッド設定を最初に行う（ライブラリインポート前に必須）
os.environ['OMP_NUM_THREADS'] = str(N_JOBS)
os.environ['MKL_NUM_THREADS'] = str(N_JOBS)
os.environ['OPENBLAS_NUM_THREADS'] = str(N_JOBS)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(N_JOBS)
os.environ['NUMEXPR_NUM_THREADS'] = str(N_JOBS)
os.environ['NUMBA_NUM_THREADS'] = str(N_JOBS)

# PyTorch設定
import torch
torch.set_num_threads(N_JOBS)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ランダムシード
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
print(f"🧵 Threads: {N_JOBS} (P-cores only)")
print(f"💾 Available RAM: 64GB")


class TwoStageSpatioTemporalEnsemble:
    """Two-Stage構成 + 時空間特徴量を用いた4モデルアンサンブル"""
    
    def __init__(
        self,
        spatio_temporal_dir: str = "data/spatio_temporal",
        stage1_oof_path: str = "data/processed/stage1_oof_predictions.csv",
        stage1_test_path: str = "data/processed/stage1_test_predictions.csv",
        output_dir: str = "results/twostage_spatiotemporal_ensemble",
        stage1_recall_target: float = 0.98,
        stage1_weights: Tuple[float, float] = (0.85, 0.15),  # (catboost, lgbm)
        n_folds: int = 5,
        random_state: int = RANDOM_SEED,
        force_retrain: bool = False,
    ):
        self.spatio_temporal_dir = Path(spatio_temporal_dir)
        self.stage1_oof_path = Path(stage1_oof_path)
        self.stage1_test_path = Path(stage1_test_path)
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        
        self.stage1_recall_target = stage1_recall_target
        self.stage1_weights = stage1_weights
        self.n_folds = n_folds
        self.random_state = random_state
        self.force_retrain = force_retrain
        
        # ディレクトリ作成
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
        
        # Stage 1 フィルタリング用
        self.stage1_threshold = None
        self.train_mask = None
        self.test_mask = None
        
        # 予測格納
        self.oof_predictions = {}
        self.test_predictions = {}
        self.model_aucs = {}
        
        print("=" * 70)
        print("🚀 Two-Stage Spatio-Temporal 4-Model Ensemble")
        print(f"   Spatio-Temporal Data: {self.spatio_temporal_dir}")
        print(f"   Stage 1 OOF: {self.stage1_oof_path}")
        print(f"   Output: {self.output_dir}")
        print(f"   Checkpoints: {self.checkpoint_dir}")
        print(f"   Stage 1 Recall Target: {stage1_recall_target}")
        print(f"   Folds: {n_folds}, Seed: {random_state}")
        print("=" * 70)
    
    # ========================================
    # チェックポイント管理（堅牢版）
    # ========================================
    def _ckpt_path(self, name: str) -> Path:
        return self.checkpoint_dir / f"{name}.npy"
    
    def _model_ckpt_path(self, model_name: str, fold: int) -> Path:
        return self.checkpoint_dir / f"{model_name}_fold{fold}.joblib"
    
    def _state_path(self) -> Path:
        return self.checkpoint_dir / "training_state.json"
    
    def _save_state(self, state: Dict):
        """学習状態の保存"""
        with open(self._state_path(), 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load_state(self) -> Optional[Dict]:
        """学習状態の読み込み"""
        if self.force_retrain:
            return None
        path = self._state_path()
        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)
        return None
    
    def _load_oof_checkpoint(self, model_name: str) -> Optional[np.ndarray]:
        """OOF予測チェックポイントの読み込み"""
        if self.force_retrain:
            return None
        path = self._ckpt_path(f"{model_name}_oof")
        if path.exists():
            return np.load(path)
        return None
    
    def _load_test_checkpoint(self, model_name: str) -> Optional[np.ndarray]:
        """テスト予測チェックポイントの読み込み"""
        if self.force_retrain:
            return None
        path = self._ckpt_path(f"{model_name}_test")
        if path.exists():
            return np.load(path)
        return None
    
    def _save_checkpoint(self, model_name: str, oof: np.ndarray, test: np.ndarray):
        """チェックポイントの保存"""
        np.save(self._ckpt_path(f"{model_name}_oof"), oof)
        np.save(self._ckpt_path(f"{model_name}_test"), test)
        print(f"   💾 {model_name} チェックポイント保存完了")
    
    def _load_fold_checkpoint(self, model_name: str, fold: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Fold単位のチェックポイント読み込み"""
        if self.force_retrain:
            return None, None
        oof_path = self.checkpoint_dir / f"{model_name}_fold{fold}_oof.npy"
        test_path = self.checkpoint_dir / f"{model_name}_fold{fold}_test.npy"
        if oof_path.exists() and test_path.exists():
            return np.load(oof_path), np.load(test_path)
        return None, None
    
    def _save_fold_checkpoint(self, model_name: str, fold: int, oof: np.ndarray, test: np.ndarray):
        """Fold単位のチェックポイント保存"""
        np.save(self.checkpoint_dir / f"{model_name}_fold{fold}_oof.npy", oof)
        np.save(self.checkpoint_dir / f"{model_name}_fold{fold}_test.npy", test)
    
    # ========================================
    # データ読み込み & Stage 1 フィルタリング
    # ========================================
    def load_data(self):
        """データ読み込みとStage 1フィルタリング"""
        print("\n📂 データ読み込み中...")
        
        # Spatio-Temporal 生データ（GBDT用）を読み込み
        train_path = self.spatio_temporal_dir / "raw_train.parquet"
        val_path = self.spatio_temporal_dir / "raw_val.parquet"
        test_path = self.spatio_temporal_dir / "raw_test.parquet"
        
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
        
        # Stage 1 フィルタリングの適用
        self._apply_stage1_filtering()
    
    def _identify_columns(self):
        """特徴量列の識別"""
        # 除外する列
        exclude_cols = [
            self.target_col, '死者数', '負傷者数', '重傷者数', '軽傷者数',
            '当事者A_死傷状況', '当事者B_死傷状況', '本票番号', '発生日時',
            'lat', 'lon', 'geohash', 'geohash_fine', 'date', 'year',
            'accident_id', 'original_index'
        ]
        
        # 利用可能な列を抽出
        available_cols = [c for c in self.full_train_df.columns if c not in exclude_cols]
        
        # カテゴリカル列と数値列を識別
        self.cat_cols = []
        self.num_cols = []
        
        for col in available_cols:
            if self.full_train_df[col].dtype == 'object':
                self.cat_cols.append(col)
            elif self.full_train_df[col].nunique() < 50 and self.full_train_df[col].dtype in ['int64', 'int32']:
                self.cat_cols.append(col)
            else:
                self.num_cols.append(col)
        
        self.feature_cols = self.num_cols + self.cat_cols
        
        print(f"\n📊 特徴量:")
        print(f"   数値: {len(self.num_cols)} 列")
        print(f"   カテゴリ: {len(self.cat_cols)} 列")
        print(f"   合計: {len(self.feature_cols)} 列")
    
    def _apply_stage1_filtering(self):
        """Stage 1 OOF予測によるフィルタリング"""
        print("\n🔍 Stage 1 フィルタリング適用中...")
        
        # Stage 1 OOF予測を読み込み
        if not self.stage1_oof_path.exists():
            raise FileNotFoundError(
                f"Stage 1 OOF予測が見つかりません: {self.stage1_oof_path}\n"
                "先に save_stage1_oof.py を実行してください。"
            )
        
        df_oof = pd.read_csv(self.stage1_oof_path)
        df_test_pred = pd.read_csv(self.stage1_test_path)
        
        # 重み付きアンサンブル確率
        cat_w, lgb_w = self.stage1_weights
        oof_prob = cat_w * df_oof['prob_catboost'].values + lgb_w * df_oof['prob_lgbm'].values
        test_prob = cat_w * df_test_pred['prob_catboost'].values + lgb_w * df_test_pred['prob_lgbm'].values
        
        # Recall target の閾値を見つける
        y_train_oof = df_oof['target'].values if 'target' in df_oof.columns else None
        
        if y_train_oof is not None:
            precision, recall, thresholds = precision_recall_curve(y_train_oof, oof_prob)
            valid_idx = np.where(recall[:-1] >= self.stage1_recall_target)[0]
            if len(valid_idx) > 0:
                best_idx = valid_idx[-1]
                self.stage1_threshold = thresholds[best_idx]
            else:
                self.stage1_threshold = 0.0
        else:
            # 既知の閾値を使用
            self.stage1_threshold = 0.0645
        
        print(f"   Stage 1 閾値: {self.stage1_threshold:.4f}")
        
        # フィルタリング用のoriginal_indexを取得
        train_original_indices = df_oof[oof_prob >= self.stage1_threshold]['original_index'].values
        test_original_indices = df_test_pred[test_prob >= self.stage1_threshold]['original_index'].values
        
        print(f"   Train OOF: {len(oof_prob):,} → {len(train_original_indices):,} (通過率: {len(train_original_indices)/len(oof_prob)*100:.1f}%)")
        print(f"   Test:      {len(test_prob):,} → {len(test_original_indices):,} (通過率: {len(test_original_indices)/len(test_prob)*100:.1f}%)")
        
        # 【Fix #4】インデックス整合性チェック
        # Stage 1 OOF作成時とデータの並びが同一であることを確認
        train_idx_set = set(self.full_train_df.index)
        test_idx_set = set(self.test_df.index)
        train_match = len(train_idx_set.intersection(train_original_indices))
        test_match = len(test_idx_set.intersection(test_original_indices))
        print(f"   インデックス一致確認: Train {train_match:,}, Test {test_match:,}")
        
        if train_match == 0:
            print("   ⚠️ 警告: Trainインデックスが一致しません。データの並び順を確認してください。")
        if test_match == 0:
            print("   ⚠️ 警告: Testインデックスが一致しません。データの並び順を確認してください。")
        
        # 【追加Fix】全テストデータのインデックスを保存（後で全件復元に使用）
        self.original_test_indices = self.test_df.index.tolist()
        self.filtered_test_indices = test_original_indices.tolist()
        # 【ID Propagation】フィルタ済みTrainインデックスも保存
        self.filtered_train_indices = train_original_indices.tolist()
        
        # Spatio-Temporal データをフィルタリング
        # full_train_dfとtest_dfのインデックスで絞り込み
        self.full_train_df = self.full_train_df[self.full_train_df.index.isin(train_original_indices)].reset_index(drop=True)
        self.test_df = self.test_df[self.test_df.index.isin(test_original_indices)].reset_index(drop=True)
        
        print(f"   フィルタ後 Train: {len(self.full_train_df):,} 行")
        print(f"   フィルタ後 Test:  {len(self.test_df):,} 行")
        
        # ターゲット確認
        if self.target_col not in self.full_train_df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in data")
        
        train_fatal = self.full_train_df[self.target_col].sum()
        test_fatal = self.test_df[self.target_col].sum()
        print(f"   Train Fatal: {train_fatal:,} ({train_fatal/len(self.full_train_df)*100:.2f}%)")
        print(f"   Test Fatal:  {test_fatal:,} ({test_fatal/len(self.test_df)*100:.2f}%)")
    
    # ========================================
    # LightGBM
    # ========================================
    def train_lgbm(self) -> Tuple[np.ndarray, np.ndarray]:
        """LightGBM学習"""
        print("\n📈 lightgbm 学習中...")
        
        # チェックポイント確認
        oof_ckpt = self._load_oof_checkpoint("lgbm")
        test_ckpt = self._load_test_checkpoint("lgbm")
        if oof_ckpt is not None and test_ckpt is not None:
            print("   📂 チェックポイントから復元")
            self.oof_predictions["lgbm"] = oof_ckpt
            self.test_predictions["lgbm"] = test_ckpt
            auc = roc_auc_score(self.full_train_df[self.target_col].values, oof_ckpt)
            self.model_aucs["lgbm"] = auc
            print(f"   lgbm OOF AUC: {auc:.4f}")
            return oof_ckpt, test_ckpt
        
        X = self.full_train_df[self.feature_cols].copy()
        y = self.full_train_df[self.target_col].values
        X_test = self.test_df[self.feature_cols].copy()
        
        # カテゴリカル列の処理
        for col in self.cat_cols:
            X[col] = X[col].astype('category')
            X_test[col] = X_test[col].astype('category')
        
        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))
        
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'max_depth': 8,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 100,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,
            'n_jobs': N_JOBS,
            'seed': self.random_state,
        }
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            
            # Fold単位チェックポイント確認
            fold_oof, fold_test = self._load_fold_checkpoint("lgbm", fold)
            if fold_oof is not None and fold_test is not None:
                oof_preds[val_idx] = fold_oof
                test_preds += fold_test / self.n_folds
                print(f"      Fold {fold+1} チェックポイントから復元")
                continue
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=self.cat_cols)
            val_data = lgb.Dataset(X_val, label=y_val, categorical_feature=self.cat_cols, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                num_boost_round=2000,
                valid_sets=[val_data],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
            )
            
            val_pred = model.predict(X_val)
            test_pred = model.predict(X_test)
            
            oof_preds[val_idx] = val_pred
            test_preds += test_pred / self.n_folds
            
            fold_auc = roc_auc_score(y_val, val_pred)
            print(f"      Fold {fold+1} AUC: {fold_auc:.4f}")
            
            # Fold単位チェックポイント保存
            self._save_fold_checkpoint("lgbm", fold, val_pred, test_pred)
            
            # モデル保存
            joblib.dump(model, self._model_ckpt_path("lgbm", fold))
            
            del model, train_data, val_data
            gc.collect()
        
        # 最終チェックポイント保存
        self._save_checkpoint("lgbm", oof_preds, test_preds)
        
        auc = roc_auc_score(y, oof_preds)
        self.oof_predictions["lgbm"] = oof_preds
        self.test_predictions["lgbm"] = test_preds
        self.model_aucs["lgbm"] = auc
        print(f"   lgbm OOF AUC: {auc:.4f}")
        
        return oof_preds, test_preds
    
    # ========================================
    # CatBoost
    # ========================================
    def train_catboost(self) -> Tuple[np.ndarray, np.ndarray]:
        """CatBoost学習"""
        print("\n🐱 catboost 学習中...")
        
        # チェックポイント確認
        oof_ckpt = self._load_oof_checkpoint("catboost")
        test_ckpt = self._load_test_checkpoint("catboost")
        if oof_ckpt is not None and test_ckpt is not None:
            print("   📂 チェックポイントから復元")
            self.oof_predictions["catboost"] = oof_ckpt
            self.test_predictions["catboost"] = test_ckpt
            auc = roc_auc_score(self.full_train_df[self.target_col].values, oof_ckpt)
            self.model_aucs["catboost"] = auc
            print(f"   catboost OOF AUC: {auc:.4f}")
            return oof_ckpt, test_ckpt
        
        X = self.full_train_df[self.feature_cols].copy()
        y = self.full_train_df[self.target_col].values
        X_test = self.test_df[self.feature_cols].copy()
        
        # カテゴリカル列のインデックス
        cat_features = [self.feature_cols.index(c) for c in self.cat_cols if c in self.feature_cols]
        
        # 文字列型に変換（CatBoost用）
        for col in self.cat_cols:
            X[col] = X[col].astype(str).fillna('missing')
            X_test[col] = X_test[col].astype(str).fillna('missing')
        
        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            
            # Fold単位チェックポイント確認
            fold_oof, fold_test = self._load_fold_checkpoint("catboost", fold)
            if fold_oof is not None and fold_test is not None:
                oof_preds[val_idx] = fold_oof
                test_preds += fold_test / self.n_folds
                print(f"      Fold {fold+1} チェックポイントから復元")
                continue
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = CatBoostClassifier(
                iterations=2000,
                learning_rate=0.05,
                depth=8,
                l2_leaf_reg=3,
                loss_function='Logloss',
                eval_metric='AUC',
                cat_features=cat_features,
                random_seed=self.random_state,
                task_type='CPU',  # CPUモードに変更（GPUは大規模データでハングする場合あり）
                thread_count=N_JOBS,
                early_stopping_rounds=50,
                verbose=100,
            )
            
            model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
            )
            
            val_pred = model.predict_proba(X_val)[:, 1]
            test_pred = model.predict_proba(X_test)[:, 1]
            
            oof_preds[val_idx] = val_pred
            test_preds += test_pred / self.n_folds
            
            fold_auc = roc_auc_score(y_val, val_pred)
            print(f"      Fold {fold+1} AUC: {fold_auc:.4f}")
            
            # Fold単位チェックポイント保存
            self._save_fold_checkpoint("catboost", fold, val_pred, test_pred)
            
            # モデル保存
            model.save_model(str(self._model_ckpt_path("catboost", fold)))
            
            del model
            gc.collect()
        
        # 最終チェックポイント保存
        self._save_checkpoint("catboost", oof_preds, test_preds)
        
        auc = roc_auc_score(y, oof_preds)
        self.oof_predictions["catboost"] = oof_preds
        self.test_predictions["catboost"] = test_preds
        self.model_aucs["catboost"] = auc
        print(f"   catboost OOF AUC: {auc:.4f}")
        
        return oof_preds, test_preds
    
    # ========================================
    # MLP (PyTorch)
    # ========================================
    def train_mlp(self) -> Tuple[np.ndarray, np.ndarray]:
        """MLP学習（データリーク対策済み）"""
        print("\n🧠 mlp 学習中...")
        
        # チェックポイント確認
        oof_ckpt = self._load_oof_checkpoint("mlp")
        test_ckpt = self._load_test_checkpoint("mlp")
        if oof_ckpt is not None and test_ckpt is not None:
            print("   📂 チェックポイントから復元")
            self.oof_predictions["mlp"] = oof_ckpt
            self.test_predictions["mlp"] = test_ckpt
            auc = roc_auc_score(self.full_train_df[self.target_col].values, oof_ckpt)
            self.model_aucs["mlp"] = auc
            print(f"   mlp OOF AUC: {auc:.4f}")
            return oof_ckpt, test_ckpt
        
        X = self.full_train_df[self.feature_cols].copy()
        y = self.full_train_df[self.target_col].values
        X_test = self.test_df[self.feature_cols].copy()
        
        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            
            # Fold単位チェックポイント確認
            fold_oof, fold_test = self._load_fold_checkpoint("mlp", fold)
            if fold_oof is not None and fold_test is not None:
                oof_preds[val_idx] = fold_oof
                test_preds += fold_test / self.n_folds
                print(f"      Fold {fold+1} チェックポイントから復元")
                continue
            
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y[train_idx], y[val_idx]
            X_test_fold = X_test.copy()
            
            # ===【データリーク対策】Fold内でエンコーダを訓練データのみでfit===
            from sklearn.impute import SimpleImputer
            
            scaler = StandardScaler()
            imputer = SimpleImputer(strategy='mean')
            label_encoders = {}
            
            # 数値列の欠損埋め (平均値) & スケーリング
            X_train_num = scaler.fit_transform(imputer.fit_transform(X_train[self.num_cols]))
            X_val_num = scaler.transform(imputer.transform(X_val[self.num_cols]))
            X_test_num = scaler.transform(imputer.transform(X_test_fold[self.num_cols]))
            
            # 【追加Fix #4】カテゴリ列のエンコーディング（辞書マップで高速化・安全性向上）
            X_train_cat = np.zeros((len(X_train), len(self.cat_cols)), dtype=np.int64)
            X_val_cat = np.zeros((len(X_val), len(self.cat_cols)), dtype=np.int64)
            X_test_cat = np.zeros((len(X_test_fold), len(self.cat_cols)), dtype=np.int64)
            
            cat_mappers = {}
            for i, col in enumerate(self.cat_cols):
                train_vals = X_train[col].astype(str).fillna('missing')
                # 辞書マップ作成（+1オフセット: 0=未知, 1以上=既知）
                unique_vals = sorted(set(train_vals))
                mapper = {v: idx + 1 for idx, v in enumerate(unique_vals)}
                cat_mappers[col] = mapper
                
                # 高速変換: map関数で一括変換（存在しないキーはNaN→fillna(0)）
                X_train_cat[:, i] = train_vals.map(mapper).fillna(0).astype(np.int64).values
                
                val_vals = X_val[col].astype(str).fillna('missing')
                test_vals = X_test_fold[col].astype(str).fillna('missing')
                
                X_val_cat[:, i] = val_vals.map(mapper).fillna(0).astype(np.int64).values
                X_test_cat[:, i] = test_vals.map(mapper).fillna(0).astype(np.int64).values
            
            # PyTorchテンソル
            # 【Fix】Train用テンソルはCPUに保持（DataLoader + pin_memory用）
            X_train_t = torch.FloatTensor(np.hstack([X_train_num, X_train_cat]))  # CPU
            y_train_t = torch.FloatTensor(y_train)  # CPU
            # Val/TestはGPUに直接配置
            X_val_t = torch.FloatTensor(np.hstack([X_val_num, X_val_cat])).to(DEVICE)
            X_test_t = torch.FloatTensor(np.hstack([X_test_num, X_test_cat])).to(DEVICE)
            y_val_t = torch.FloatTensor(y_val).to(DEVICE)
            
            # モデル定義（Sigmoid層なし - BCEWithLogitsLoss使用）
            input_dim = X_train_t.shape[1]
            model = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 512),
                torch.nn.BatchNorm1d(512),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(512, 256),
                torch.nn.BatchNorm1d(256),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(256, 64),
                torch.nn.BatchNorm1d(64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(64, 1),
                # NO Sigmoid here - BCEWithLogitsLoss applies it internally
            ).to(DEVICE)
            
            # 損失関数と最適化
            pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / sum(y_train)]).to(DEVICE)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            # 【Fix #3】mode='max' でAUCをそのまま渡す（直感的で安全）
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
            
            # データローダー
            # 【Fix】num_workers=0でマルチプロセス問題を回避、pin_memory=Trueで高速GPU転送
            train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0, pin_memory=True)
            
            # 学習ループ
            best_val_auc = 0
            patience = 10
            no_improve = 0
            
            for epoch in range(100):
                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)  # CPU -> GPU
                    optimizer.zero_grad()
                    outputs = model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                
                # 検証
                model.eval()
                with torch.no_grad():
                    val_logits = model(X_val_t).squeeze()
                    val_prob = torch.sigmoid(val_logits).cpu().numpy()  # Explicit sigmoid for inference
                    val_auc = roc_auc_score(y_val, val_prob)
                
                scheduler.step(val_auc)  # 【Fix #3】AUCをそのまま渡す
                
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_model_state = model.state_dict().copy()
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
            
            # ベストモデルで推論
            model.load_state_dict(best_model_state)
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t).squeeze()
                test_logits = model(X_test_t).squeeze()
                val_pred = torch.sigmoid(val_logits).cpu().numpy()  # Explicit sigmoid
                test_pred = torch.sigmoid(test_logits).cpu().numpy()  # Explicit sigmoid
            
            oof_preds[val_idx] = val_pred
            test_preds += test_pred / self.n_folds
            
            fold_auc = roc_auc_score(y_val, val_pred)
            print(f"      Fold {fold+1} AUC: {fold_auc:.4f}")
            
            # Fold単位チェックポイント保存
            self._save_fold_checkpoint("mlp", fold, val_pred, test_pred)
            
            # モデル保存
            torch.save(model.state_dict(), self._model_ckpt_path("mlp", fold))
            
            del model, X_train_t, X_val_t, X_test_t
            torch.cuda.empty_cache() if DEVICE == 'cuda' else None
            gc.collect()
        
        # 最終チェックポイント保存
        self._save_checkpoint("mlp", oof_preds, test_preds)
        
        auc = roc_auc_score(y, oof_preds)
        self.oof_predictions["mlp"] = oof_preds
        self.test_predictions["mlp"] = test_preds
        self.model_aucs["mlp"] = auc
        print(f"   mlp OOF AUC: {auc:.4f}")
        
        return oof_preds, test_preds
    
    # ========================================
    # TabNet
    # ========================================
    def train_tabnet(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """TabNet学習（データリーク対策済み）"""
        print("\n📊 tabnet 学習中...")
        
        # 【Fix #1】TabNet未利用時はNoneを返し、アンサンブルから除外
        if not TABNET_AVAILABLE:
            print("   ⚠️ TabNet not available, skipping...")
            return None, None
        
        # チェックポイント確認
        oof_ckpt = self._load_oof_checkpoint("tabnet")
        test_ckpt = self._load_test_checkpoint("tabnet")
        if oof_ckpt is not None and test_ckpt is not None:
            print("   📂 チェックポイントから復元")
            self.oof_predictions["tabnet"] = oof_ckpt
            self.test_predictions["tabnet"] = test_ckpt
            auc = roc_auc_score(self.full_train_df[self.target_col].values, oof_ckpt)
            self.model_aucs["tabnet"] = auc
            print(f"   tabnet OOF AUC: {auc:.4f}")
            return oof_ckpt, test_ckpt
        
        X = self.full_train_df[self.feature_cols].copy()
        y = self.full_train_df[self.target_col].values
        X_test = self.test_df[self.feature_cols].copy()
        
        oof_preds = np.zeros(len(X))
        test_preds = np.zeros(len(X_test))
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            
            # Fold単位チェックポイント確認
            fold_oof, fold_test = self._load_fold_checkpoint("tabnet", fold)
            if fold_oof is not None and fold_test is not None:
                oof_preds[val_idx] = fold_oof
                test_preds += fold_test / self.n_folds
                print(f"      Fold {fold+1} チェックポイントから復元")
                continue
            
            X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
            y_train, y_val = y[train_idx], y[val_idx]
            X_test_fold = X_test.copy()
            
            # ===【データリーク対策】Fold内でエンコーダを訓練データのみでfit===
            scaler = StandardScaler()
            label_encoders = {}
            
            # 数値列のスケーリング
            X_train[self.num_cols] = scaler.fit_transform(X_train[self.num_cols].fillna(0))
            X_val[self.num_cols] = scaler.transform(X_val[self.num_cols].fillna(0))
            X_test_fold[self.num_cols] = scaler.transform(X_test_fold[self.num_cols].fillna(0))
            
            # 【Fix #2】カテゴリ列のエンコーディング（+1オフセットで未知カテゴリ=0と既存カテゴリを分離）
            # 【高速化】辞書マッピングによるベクトル化エンコーディング
            cat_idxs = []
            cat_dims = []
            
            for i, col in enumerate(self.feature_cols):
                if col in self.cat_cols:
                    # 文字列変換
                    train_vals = X_train[col].astype(str).fillna('missing')
                    val_vals = X_val[col].astype(str).fillna('missing')
                    test_vals = X_test_fold[col].astype(str).fillna('missing')
                    
                    # マッパー作成 (1-based index, 0=unknown)
                    unique_vals = sorted(set(train_vals))
                    mapper = {v: idx + 1 for idx, v in enumerate(unique_vals)}
                    
                    # mapで一括変換 (存在しないキーはNaNになる→0で埋める)
                    X_train[col] = train_vals.map(mapper).fillna(0).astype(int)
                    X_val[col] = val_vals.map(mapper).fillna(0).astype(int)
                    X_test_fold[col] = test_vals.map(mapper).fillna(0).astype(int)
                    
                    cat_idxs.append(i)
                    cat_dims.append(len(unique_vals) + 1)  # +1 for unknown category (0)
            
            # NumPy配列に変換
            X_train_np = X_train[self.feature_cols].values.astype(np.float32)
            X_val_np = X_val[self.feature_cols].values.astype(np.float32)
            X_test_np = X_test_fold[self.feature_cols].values.astype(np.float32)
            
            # TabNet
            model = TabNetClassifier(
                n_d=32,
                n_a=32,
                n_steps=5,
                gamma=1.5,
                lambda_sparse=1e-4,
                cat_idxs=cat_idxs,
                cat_dims=cat_dims,
                cat_emb_dim=8,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=0.02),
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                scheduler_params=dict(step_size=10, gamma=0.9),
                seed=self.random_state,
                verbose=0,
                device_name=DEVICE,
            )
            
            model.fit(
                X_train_np, y_train,
                eval_set=[(X_val_np, y_val)],
                eval_metric=['auc'],
                max_epochs=100,
                patience=20,
                batch_size=512,  # 縮小（フィルタ後データ用）
                virtual_batch_size=128,
            )
            
            val_pred = model.predict_proba(X_val_np)[:, 1]
            test_pred = model.predict_proba(X_test_np)[:, 1]
            
            # 【安定性対策】NaNチェックとフォールバック
            if np.isnan(val_pred).any():
                print(f"      ⚠️ Warning: TabNet produced NaN in Fold {fold+1}. Fallback to mean.")
                val_pred = np.nan_to_num(val_pred, nan=np.mean(y_train))
            if np.isnan(test_pred).any():
                test_pred = np.nan_to_num(test_pred, nan=np.mean(y_train))
            
            oof_preds[val_idx] = val_pred
            test_preds += test_pred / self.n_folds
            
            fold_auc = roc_auc_score(y_val, val_pred)
            print(f"      Fold {fold+1} AUC: {fold_auc:.4f}")
            
            # Fold単位チェックポイント保存
            self._save_fold_checkpoint("tabnet", fold, val_pred, test_pred)
            
            # モデル保存
            model.save_model(str(self._model_ckpt_path("tabnet", fold)))
            
            del model
            gc.collect()
        
        # 最終チェックポイント保存
        self._save_checkpoint("tabnet", oof_preds, test_preds)
        
        auc = roc_auc_score(y, oof_preds)
        # 【Fix #1】TabNetが正常に学習した場合のみ辞書に追加
        self.oof_predictions["tabnet"] = oof_preds
        self.test_predictions["tabnet"] = test_preds
        self.model_aucs["tabnet"] = auc
        print(f"   tabnet OOF AUC: {auc:.4f}")
        
        return oof_preds, test_preds
    
    # ========================================
    # アンサンブル重み最適化
    # ========================================
    def optimize_weights(self) -> Dict[str, float]:
        """アンサンブル重みの最適化（【Fix #1, #5】改良版）"""
        print("\n⚖️ アンサンブル重み最適化中...")
        
        y = self.full_train_df[self.target_col].values
        
        # 【Fix #1】Noneのモデル（学習失敗やスキップ）を除外
        # 【追加Fix #5】AUCが低いモデル（学習失敗）も除外
        valid_models = {}
        for k, v in self.oof_predictions.items():
            if v is None or k == "ensemble":
                continue
            auc = roc_auc_score(y, v)
            if auc < 0.55:
                print(f"   ⚠️ {k} を除外 (AUCが低すぎる: {auc:.4f})")
                continue
            valid_models[k] = v
        model_names = list(valid_models.keys())
        
        if len(model_names) == 0:
            print("   ⚠️ 有効なモデルがありません")
            return {}
        
        if len(model_names) < 2:
            print("   モデルが1つしかないため最適化をスキップ")
            self.oof_predictions["ensemble"] = valid_models[model_names[0]]
            self.test_predictions["ensemble"] = self.test_predictions[model_names[0]]
            return {model_names[0]: 1.0}
        
        oof_matrix = np.column_stack([valid_models[name] for name in model_names])
        
        def objective(weights):
            ensemble_pred = np.dot(oof_matrix, weights)
            return -roc_auc_score(y, ensemble_pred)
        
        init_weights = np.ones(len(model_names)) / len(model_names)
        
        # 【Fix #5】SLSQP: 制約付き最適化（重みは0〜1、合計1）
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = [(0.0, 1.0) for _ in range(len(model_names))]
        result = minimize(objective, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        optimal_weights = result.x
        weights_dict = {name: float(w) for name, w in zip(model_names, optimal_weights)}
        
        print("   最適化された重み:")
        for name, w in weights_dict.items():
            print(f"     {name}: {w:.4f}")
        
        # アンサンブル予測
        ensemble_oof = np.dot(oof_matrix, optimal_weights)
        ensemble_test = np.dot(
            np.column_stack([self.test_predictions[name] for name in model_names]),
            optimal_weights
        )
        
        ensemble_auc = roc_auc_score(y, ensemble_oof)
        print(f"   Ensemble OOF AUC: {ensemble_auc:.4f}")
        
        self.oof_predictions["ensemble"] = ensemble_oof
        self.test_predictions["ensemble"] = ensemble_test
        self.model_aucs["ensemble"] = ensemble_auc
        
        # 重み保存
        with open(self.output_dir / "ensemble_weights.json", 'w') as f:
            json.dump(weights_dict, f, indent=2)
        
        return weights_dict
    
    # ========================================
    # 結果保存
    # ========================================
    def save_results(self):
        """結果の保存"""
        print("\n📈 結果保存中...")
        
        y_train = self.full_train_df[self.target_col].values
        y_test = self.test_df[self.target_col].values
        
        # スコア計算
        scores = []
        for model_name in self.oof_predictions.keys():
            oof_auc = roc_auc_score(y_train, self.oof_predictions[model_name])
            oof_prauc = average_precision_score(y_train, self.oof_predictions[model_name])
            test_auc = roc_auc_score(y_test, self.test_predictions[model_name])
            test_prauc = average_precision_score(y_test, self.test_predictions[model_name])
            
            scores.append({
                'model': model_name,
                'oof_auc': oof_auc,
                'oof_prauc': oof_prauc,
                'test_auc': test_auc,
                'test_prauc': test_prauc,
            })
            
            print(f"   {model_name}: OOF AUC={oof_auc:.4f}, Test AUC={test_auc:.4f}")
        
        # スコア保存
        scores_df = pd.DataFrame(scores)
        scores_df.to_csv(self.output_dir / "final_scores.csv", index=False)
        
        # OOF予測保存（original_indexを含む）
        oof_df = pd.DataFrame(self.oof_predictions)
        # 【修正】フィルタ後のfull_train_dfからoriginal_indexを取得（正しい長さ）
        oof_df['original_index'] = self.full_train_df['original_index'].values
        oof_df['target'] = y_train
        oof_df.to_csv(self.output_dir / "oof_predictions.csv", index=False)
        
        # テスト予測保存（フィルタ済みのみ、original_indexを含む）
        test_df = pd.DataFrame(self.test_predictions)
        # 【修正】フィルタ後のtest_dfからoriginal_indexを取得（正しい長さ）
        test_df['original_index'] = self.test_df['original_index'].values
        test_df['target'] = y_test
        test_df.to_csv(self.output_dir / "test_predictions.csv", index=False)
        
        # 【追加Fix #1】全テストデータへの復元処理
        print("\n   🔄 全テストデータへの予測復元中...")
        try:
            # オリジナルの全テストデータを読み込み
            raw_test_df = pd.read_parquet(self.spatio_temporal_dir / "raw_test.parquet")
            
            # 全行を含むDataFrameを作成
            final_submission = pd.DataFrame({
                'original_index': raw_test_df.index
            })
            
            # Stage 2 予測結果（フィルタ済み）をマージ用に準備
            # フィルタ後のテストデータの元のインデックスを使用
            if hasattr(self, 'filtered_test_indices') and 'ensemble' in self.test_predictions:
                stage2_preds = pd.DataFrame({
                    'original_index': self.filtered_test_indices[:len(self.test_predictions['ensemble'])],
                    'prob_ensemble': self.test_predictions['ensemble']
                })
                
                # 全データにマージ（Stage 2にないデータは欠損になる）
                final_submission = final_submission.merge(stage2_preds, on='original_index', how='left')
                
                # Stage 2に含まれなかったデータ（Easy sample）は0.0で埋める
                # （閾値以下の低確率として扱う）
                final_submission['prob_ensemble'] = final_submission['prob_ensemble'].fillna(0.0)
                
                # 保存
                final_submission.to_csv(self.output_dir / "final_submission_full.csv", index=False)
                print(f"   💾 全件復元済み予測を保存: final_submission_full.csv ({len(final_submission):,} 行)")
                print(f"      Stage 2 対象: {(final_submission['prob_ensemble'] > 0).sum():,} 行")
                print(f"      フィルタ除外: {(final_submission['prob_ensemble'] == 0).sum():,} 行")
            else:
                print("   ⚠️ 全件復元に必要な情報が不足しています")
        except Exception as e:
            print(f"   ⚠️ 全件復元に失敗: {e}")
        
        print(f"   ✅ 完了: {self.output_dir}")
    
    # ========================================
    # メイン実行
    # ========================================
    def run(self):
        """学習パイプラインの実行"""
        start_time = datetime.now()
        
        # データ読み込み
        self.load_data()
        
        # 各モデルの学習
        if LGBM_AVAILABLE:
            self.train_lgbm()
        
        if CATBOOST_AVAILABLE:
            self.train_catboost()
        
        self.train_mlp()
        
        if TABNET_AVAILABLE:
            self.train_tabnet()
        
        # アンサンブル
        self.optimize_weights()
        
        # 結果保存
        self.save_results()
        
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        
        print("\n" + "=" * 70)
        print(f"✅ 全工程完了！ 実行時間: {elapsed:.1f}分")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Two-Stage Spatio-Temporal 4-Model Ensemble")
    parser.add_argument('--spatio-temporal-dir', type=str, default="data/spatio_temporal",
                        help="Spatio-Temporal前処理済みデータのディレクトリ")
    parser.add_argument('--stage1-oof', type=str, default="data/processed/stage1_oof_predictions.csv",
                        help="Stage 1 OOF予測ファイル")
    parser.add_argument('--stage1-test', type=str, default="data/processed/stage1_test_predictions.csv",
                        help="Stage 1 テスト予測ファイル")
    parser.add_argument('--output-dir', type=str, default="results/twostage_spatiotemporal_ensemble",
                        help="結果出力ディレクトリ")
    parser.add_argument('--recall-target', type=float, default=0.98,
                        help="Stage 1 Recall Target (default: 0.98)")
    parser.add_argument('--n-folds', type=int, default=5, help="交差検証のフォールド数")
    parser.add_argument('--force-retrain', action='store_true',
                        help="チェックポイントを無視して最初から学習")
    parser.add_argument('--resume', action='store_true',
                        help="チェックポイントから再開（デフォルトの動作）")
    
    args = parser.parse_args()
    
    ensemble = TwoStageSpatioTemporalEnsemble(
        spatio_temporal_dir=args.spatio_temporal_dir,
        stage1_oof_path=args.stage1_oof,
        stage1_test_path=args.stage1_test,
        output_dir=args.output_dir,
        stage1_recall_target=args.recall_target,
        n_folds=args.n_folds,
        force_retrain=args.force_retrain,
    )
    
    ensemble.run()


if __name__ == "__main__":
    main()
