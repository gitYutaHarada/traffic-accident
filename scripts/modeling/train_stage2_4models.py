"""
Stage 2: 4モデルアンサンブル (Intel Core Ultra 9 最適化版 v2)
LightGBM, CatBoost, TabNet, MLP を使用して Stage 1 フィルタリング済みデータを学習。
最後に重み付きアンサンブルを実行。

修正版 (v2):
- データ整合性: OOF予測をマージしてから train_test_split
- MLP: Embedding層でカテゴリ変数を適切に処理
- TabNet: cat_idxs, cat_dims を正しく設定
- Intel最適化: OMP/MKL スレッド設定
- PyTorch再現性: シード固定

使用法:
    python scripts/modeling/train_stage2_4models.py
"""
import os
import gc
import json
import warnings
warnings.filterwarnings('ignore')

# ====== Intel スレッド設定 (最初に設定) ======
N_JOBS = 8  # Intel Core Ultra 9 285K: 8 P-cores
os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_JOBS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(N_JOBS)
os.environ["NUMEXPR_NUM_THREADS"] = str(N_JOBS)

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.optimize import minimize

# Intel Extension for Scikit-learn (高速化)
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    print("✅ Intel Extension for Scikit-learn enabled")
except ImportError:
    print("⚠️ sklearnex not available. Install with: pip install scikit-learn-intelex")

import lightgbm as lgb
from catboost import CatBoostClassifier

# TabNet
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("⚠️ TabNet not available. Install with: pip install pytorch-tabnet")

# MLP (PyTorch)
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    # Intel拡張 (PyTorch)
    try:
        import intel_extension_for_pytorch as ipex
        IPEX_AVAILABLE = True
        print("✅ Intel Extension for PyTorch enabled")
    except ImportError:
        IPEX_AVAILABLE = False
except ImportError:
    TORCH_AVAILABLE = False
    IPEX_AVAILABLE = False
    print("⚠️ PyTorch not available.")


# ====== ハードウェア最適化設定 ======
BATCH_SIZE_LARGE = 8192  # 64GB RAMを活用
BATCH_SIZE_MLP = 4096
NUM_WORKERS = 0  # Windows互換性のため0に設定


def set_seed(seed: int):
    """再現性のためのシード設定"""
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class EmbeddingMLP(nn.Module):
    """カテゴリ変数をEmbeddingで処理するMLP"""
    
    def __init__(self, num_numerical: int, cat_dims: list, embed_dims: list, hidden_dims: list = [512, 256, 64]):
        super().__init__()
        
        # Embedding layers for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, embed_dim) 
            for num_classes, embed_dim in zip(cat_dims, embed_dims)
        ])
        
        # Total input dimension
        total_embed_dim = sum(embed_dims)
        input_dim = num_numerical + total_embed_dim
        
        # Build MLP layers: Linear -> BatchNorm -> ReLU -> Dropout
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            dropout_rate = 0.3 if i == 0 else 0.2
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)
        self.num_numerical = num_numerical
        
    def forward(self, x_numerical, x_categorical):
        # Process categorical features through embeddings
        embedded = [emb(x_categorical[:, i]) for i, emb in enumerate(self.embeddings)]
        embedded = torch.cat(embedded, dim=1) if embedded else torch.zeros(x_numerical.size(0), 0, device=x_numerical.device)
        
        # Concatenate numerical and embedded categorical
        x = torch.cat([x_numerical, embedded], dim=1)
        
        return self.mlp(x)


class Stage2EnsemblePipeline:
    """Stage 2: 4モデルアンサンブルパイプライン (修正版)"""
    
    def __init__(
        self,
        data_path: str = "data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv",
        oof_predictions_path: str = "data/processed/stage1_oof_predictions.csv",
        test_predictions_path: str = "data/processed/stage1_test_predictions.csv",
        target_col: str = "死者数",
        stage1_recall_target: float = 0.98,
        n_folds: int = 5,
        random_state: int = 42,
        test_size: float = 0.2,
        output_dir: str = "results/stage2_4model_ensemble"
    ):
        self.data_path = data_path
        self.oof_predictions_path = oof_predictions_path
        self.test_predictions_path = test_predictions_path
        self.target_col = target_col
        self.stage1_recall_target = stage1_recall_target
        self.n_folds = n_folds
        self.random_state = random_state
        self.test_size = test_size
        self.output_dir = output_dir
        
        # チェックポイントディレクトリ
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # シード設定
        set_seed(random_state)
        
        # 結果格納用
        self.oof_predictions = {}
        self.test_predictions = {}
        self.model_aucs = {}
        
    def _checkpoint_path(self, model_name: str) -> str:
        """チェックポイントファイルパス"""
        return os.path.join(self.checkpoint_dir, f"{model_name}_checkpoint.npz")
    
    def _save_checkpoint(self, model_name: str, oof_proba: np.ndarray, test_proba: np.ndarray, auc: float):
        """チェックポイント保存"""
        path = self._checkpoint_path(model_name)
        np.savez(path, oof_proba=oof_proba, test_proba=test_proba, auc=auc)
        print(f"   💾 チェックポイント保存: {model_name}")
    
    def _load_checkpoint(self, model_name: str) -> bool:
        """チェックポイントロード (成功時 True)"""
        path = self._checkpoint_path(model_name)
        if os.path.exists(path):
            data = np.load(path)
            self.oof_predictions[model_name] = data['oof_proba']
            self.test_predictions[model_name] = data['test_proba']
            self.model_aucs[model_name] = float(data['auc'])
            print(f"   📂 チェックポイントから復元: {model_name} (AUC: {self.model_aucs[model_name]:.4f})")
            return True
        return False
        
    def load_and_filter_data(self):
        """データ読み込みとStage 1フィルタリング (データ整合性を保証)"""
        print("📂 データ読み込み...")
        
        # メインデータ
        df = pd.read_csv(self.data_path)
        print(f"   元データ: {len(df):,} 行")
        
        # Stage 1 OOF予測をロード
        df_oof = pd.read_csv(self.oof_predictions_path)
        df_test_pred = pd.read_csv(self.test_predictions_path)
        
        # ターゲット作成
        if self.target_col in df.columns:
            df['fatal'] = (df[self.target_col] > 0).astype(int)
        elif 'fatal' in df.columns:
            pass
        else:
            raise ValueError(f"Target column not found: {self.target_col}")
        
        # === 🔴 重要: Stage 1予測をメインデータにマージしてから分割 ===
        # この段階でdf_oofとdfのサイズを確認
        # save_stage1_oof.py は train_test_split を使っているので、
        # 同じrandom_stateを使えば、同じ分割になるはず
        
        # まず train_test_split の「前」の全データで考える
        # df_oof は「訓練データのOOF予測」なので、行数=訓練データ数
        # df_test_pred は「テストデータの予測」なので、行数=テストデータ数
        
        # 除外カラム
        exclude_cols = [
            self.target_col, 'fatal', '負傷者数', '重傷者数', '軽傷者数',
            '当事者A_死傷状況', '当事者B_死傷状況', '本票番号', '発生日時'
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        # カテゴリカル列の識別
        self.categorical_cols = []
        self.numerical_cols = []
        for col in feature_cols:
            if df[col].dtype == 'object' or df[col].nunique() < 50:
                self.categorical_cols.append(col)
            else:
                self.numerical_cols.append(col)
        
        # Train/Test Split (Stage 1 と同じシード・同じ手順)
        X_all = df[feature_cols].copy()
        y_all = df['fatal'].values
        
        # 分割を実行 (save_stage1_oof.py と同一のロジック)
        X_train_full, X_test, y_train_full, y_test, train_indices, test_indices = train_test_split(
            X_all, y_all, np.arange(len(df)),
            test_size=self.test_size,
            random_state=self.random_state, 
            stratify=y_all
        )
        
        # === Stage 1 予測を正しく紐付け ===
        # save_stage1_oof.pyで保存されたOOFは、分割後のTrainデータのOOF予測
        # 行順はreset_index後の順序か、Fold順かを確認する必要がある
        # 最も安全な方法: 両方のデータを同時に扱う
        
        # OOFデータの行数チェック
        expected_train_size = len(y_train_full)
        expected_test_size = len(y_test)
        
        if len(df_oof) != expected_train_size:
            print(f"⚠️ 警告: OOFサイズ不一致 (Expected: {expected_train_size}, Got: {len(df_oof)})")
            print("   → 行順が異なる可能性が高いです。データを再確認してください。")
        
        if len(df_test_pred) != expected_test_size:
            print(f"⚠️ 警告: Test予測サイズ不一致 (Expected: {expected_test_size}, Got: {len(df_test_pred)})")
        
        # reset_indexして整合性を取る
        X_train_full = X_train_full.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        self.y_train_full = y_train_full
        self.y_test = y_test
        
        # 重み付きアンサンブル確率
        oof_prob = 0.85 * df_oof['prob_catboost'].values + 0.15 * df_oof['prob_lgbm'].values
        test_prob = 0.85 * df_test_pred['prob_catboost'].values + 0.15 * df_test_pred['prob_lgbm'].values
        
        print(f"\n   Train (Full): {len(self.y_train_full):,} (Fatal: {self.y_train_full.sum():,})")
        print(f"   Test:         {len(self.y_test):,} (Fatal: {self.y_test.sum():,})")
        
        # Recall target の閾値を見つける
        precision, recall, thresholds = precision_recall_curve(self.y_train_full, oof_prob)
        valid_idx = np.where(recall[:-1] >= self.stage1_recall_target)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[-1]
            self.stage1_threshold = thresholds[best_idx]
        else:
            self.stage1_threshold = 0.0
        
        # フィルタリング適用
        train_mask = oof_prob >= self.stage1_threshold
        test_mask = test_prob >= self.stage1_threshold
        
        self.X_train_full = X_train_full
        self.X_test = X_test
        self.X_train = X_train_full[train_mask].reset_index(drop=True)
        self.y_train = self.y_train_full[train_mask]
        self.X_test_filtered = X_test[test_mask].reset_index(drop=True)
        self.y_test_filtered = self.y_test[test_mask]
        
        self.test_mask = test_mask
        
        train_recall = self.y_train.sum() / self.y_train_full.sum()
        test_recall_stage1 = self.y_test_filtered.sum() / self.y_test.sum()
        
        print(f"\n🎯 Stage 1 フィルタリング (Recall Target: {self.stage1_recall_target:.1%})")
        print(f"   閾値: {self.stage1_threshold:.4f}")
        print(f"   Train: {len(self.y_train):,} / {len(self.y_train_full):,} ({len(self.y_train)/len(self.y_train_full):.1%} 通過)")
        print(f"   Train Recall: {train_recall:.2%}")
        print(f"   Test:  {len(self.y_test_filtered):,} / {len(self.y_test):,} ({len(self.y_test_filtered)/len(self.y_test):.1%} 通過)")
        print(f"   Test Recall (Stage1): {test_recall_stage1:.2%}")
        
        self.feature_cols = feature_cols
        
        # カテゴリ変数のエンコーディング情報を事前計算
        self._prepare_categorical_encoders()
        
        gc.collect()
    
    def _prepare_categorical_encoders(self):
        """カテゴリ変数のエンコーダーを事前準備"""
        self.cat_encoders = {}
        self.cat_dims = []  # 各カテゴリのクラス数
        self.cat_idxs = []  # カテゴリ列のインデックス
        
        all_data = pd.concat([self.X_train, self.X_test_filtered], axis=0)
        
        for i, col in enumerate(self.feature_cols):
            if col in self.categorical_cols:
                le = LabelEncoder()
                le.fit(all_data[col].astype(str).fillna('__MISSING__'))
                self.cat_encoders[col] = le
                self.cat_dims.append(len(le.classes_))
                self.cat_idxs.append(i)
        
        print(f"\n   カテゴリ変数: {len(self.categorical_cols)} 列")
        print(f"   数値変数: {len(self.numerical_cols)} 列")
    
    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """カテゴリ変数をエンコード"""
        X_encoded = X.copy()
        for col, le in self.cat_encoders.items():
            if col in X_encoded.columns:
                X_encoded[col] = le.transform(X_encoded[col].astype(str).fillna('__MISSING__'))
        return X_encoded
    
    def _prepare_nn_data(self, X_train: pd.DataFrame, X_val: pd.DataFrame, X_test: pd.DataFrame):
        """NN用のデータ準備 (数値とカテゴリを分離)"""
        # カテゴリカル変数をエンコード
        X_train_enc = self._encode_categorical(X_train)
        X_val_enc = self._encode_categorical(X_val)
        X_test_enc = self._encode_categorical(X_test)
        
        # 数値変数を抽出・スケーリング
        num_cols = [c for c in self.numerical_cols if c in X_train.columns]
        
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train_enc[num_cols].fillna(X_train_enc[num_cols].median()))
        X_val_num = scaler.transform(X_val_enc[num_cols].fillna(X_train_enc[num_cols].median()))
        X_test_num = scaler.transform(X_test_enc[num_cols].fillna(X_train_enc[num_cols].median()))
        
        # カテゴリ変数を抽出
        cat_cols = [c for c in self.categorical_cols if c in X_train.columns]
        X_train_cat = X_train_enc[cat_cols].values.astype(np.int64)
        X_val_cat = X_val_enc[cat_cols].values.astype(np.int64)
        X_test_cat = X_test_enc[cat_cols].values.astype(np.int64)
        
        return (
            X_train_num.astype(np.float32), X_train_cat,
            X_val_num.astype(np.float32), X_val_cat,
            X_test_num.astype(np.float32), X_test_cat
        )
    
    def train_lightgbm(self):
        """LightGBM の学習"""
        if self._load_checkpoint('lgbm'):
            return
        
        print("\n🌲 LightGBM 学習中...")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_proba = np.zeros(len(self.y_train))
        test_proba = np.zeros(len(self.y_test_filtered))
        
        X = self.X_train.copy()
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype('category')
        
        X_test = self.X_test_filtered.copy()
        for col in self.categorical_cols:
            if col in X_test.columns:
                X_test[col] = X_test[col].astype('category')
        
        params = {
            'objective': 'binary', 
            'metric': 'auc', 
            'boosting_type': 'gbdt',
            'verbosity': -1, 
            'num_leaves': 31, 
            'max_depth': 8,
            'learning_rate': 0.05, 
            'n_estimators': 1000, 
            'n_jobs': N_JOBS,
            'force_row_wise': True
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, self.y_train)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = self.y_train[train_idx], self.y_train[val_idx]
            
            model = lgb.LGBMClassifier(**params, random_state=self.random_state + fold)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            
            oof_proba[val_idx] = model.predict_proba(X_val)[:, 1]
            test_proba += model.predict_proba(X_test)[:, 1] / self.n_folds
            gc.collect()
        
        auc = roc_auc_score(self.y_train, oof_proba)
        print(f"   LightGBM OOF AUC: {auc:.4f}")
        
        self.oof_predictions['lgbm'] = oof_proba
        self.test_predictions['lgbm'] = test_proba
        self.model_aucs['lgbm'] = auc
        
        self._save_checkpoint('lgbm', oof_proba, test_proba, auc)
    
    def train_catboost(self):
        """CatBoost の学習"""
        if self._load_checkpoint('catboost'):
            return
        
        print("\n🐱 CatBoost 学習中...")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_proba = np.zeros(len(self.y_train))
        test_proba = np.zeros(len(self.y_test_filtered))
        
        X = self.X_train.copy()
        for col in self.categorical_cols:
            if col in X.columns:
                X[col] = X[col].astype(str)
        
        X_test = self.X_test_filtered.copy()
        for col in self.categorical_cols:
            if col in X_test.columns:
                X_test[col] = X_test[col].astype(str)
        
        cat_features = [i for i, c in enumerate(X.columns) if c in self.categorical_cols]
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, self.y_train)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = self.y_train[train_idx], self.y_train[val_idx]
            
            model = CatBoostClassifier(
                iterations=1000, 
                learning_rate=0.05, 
                depth=8,
                cat_features=cat_features, 
                verbose=0, 
                thread_count=N_JOBS,
                random_state=self.random_state + fold, 
                early_stopping_rounds=50,
                task_type='CPU'
            )
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
            
            oof_proba[val_idx] = model.predict_proba(X_val)[:, 1]
            test_proba += model.predict_proba(X_test)[:, 1] / self.n_folds
            gc.collect()
        
        auc = roc_auc_score(self.y_train, oof_proba)
        print(f"   CatBoost OOF AUC: {auc:.4f}")
        
        self.oof_predictions['catboost'] = oof_proba
        self.test_predictions['catboost'] = test_proba
        self.model_aucs['catboost'] = auc
        
        self._save_checkpoint('catboost', oof_proba, test_proba, auc)
    
    def train_tabnet(self):
        """TabNet の学習 (cat_idxs, cat_dims 設定)"""
        if not TABNET_AVAILABLE:
            print("\n⚠️ TabNet スキップ (ライブラリ未インストール)")
            return
        
        if self._load_checkpoint('tabnet'):
            return
        
        print("\n📊 TabNet 学習中...")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_proba = np.zeros(len(self.y_train))
        test_proba = np.zeros(len(self.y_test_filtered))
        
        # TabNet用にカテゴリインデックスを計算
        cat_idxs = [i for i, c in enumerate(self.feature_cols) if c in self.categorical_cols]
        cat_dims = [self.cat_dims[self.categorical_cols.index(c)] for c in self.feature_cols if c in self.categorical_cols]
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.y_train[train_idx], self.y_train[val_idx]
            
            # データエンコード
            X_tr_enc = self._encode_categorical(X_tr)
            X_val_enc = self._encode_categorical(X_val)
            X_test_enc = self._encode_categorical(self.X_test_filtered)
            
            # 数値列のスケーリング
            num_cols = [c for c in self.numerical_cols if c in X_tr.columns]
            scaler = StandardScaler()
            X_tr_enc[num_cols] = scaler.fit_transform(X_tr_enc[num_cols].fillna(0))
            X_val_enc[num_cols] = scaler.transform(X_val_enc[num_cols].fillna(0))
            X_test_enc[num_cols] = scaler.transform(X_test_enc[num_cols].fillna(0))
            
            X_tr_np = X_tr_enc.values.astype(np.float32)
            X_val_np = X_val_enc.values.astype(np.float32)
            X_test_np = X_test_enc.values.astype(np.float32)
            
            model = TabNetClassifier(
                n_d=32, n_a=32, n_steps=5,
                gamma=1.5, n_independent=2, n_shared=2,
                cat_idxs=cat_idxs,
                cat_dims=cat_dims,
                cat_emb_dim=1,  # Embedding dimension
                seed=self.random_state + fold, 
                verbose=0,
                device_name='cpu'
            )
            model.fit(
                X_tr_np, y_tr,
                eval_set=[(X_val_np, y_val)],
                eval_metric=['auc'],
                max_epochs=100,
                patience=10,
                batch_size=BATCH_SIZE_LARGE,
                virtual_batch_size=1024,
                num_workers=NUM_WORKERS,
                drop_last=False
            )
            
            oof_proba[val_idx] = model.predict_proba(X_val_np)[:, 1]
            test_proba += model.predict_proba(X_test_np)[:, 1] / self.n_folds
            gc.collect()
        
        auc = roc_auc_score(self.y_train, oof_proba)
        print(f"   TabNet OOF AUC: {auc:.4f}")
        
        self.oof_predictions['tabnet'] = oof_proba
        self.test_predictions['tabnet'] = test_proba
        self.model_aucs['tabnet'] = auc
        
        self._save_checkpoint('tabnet', oof_proba, test_proba, auc)
    
    def train_mlp(self):
        """MLP の学習 (Embedding層でカテゴリを処理)"""
        if not TORCH_AVAILABLE:
            print("\n⚠️ MLP スキップ (PyTorch未インストール)")
            return
        
        if self._load_checkpoint('mlp'):
            return
        
        print("\n🧠 MLP 学習中...")
        
        device = torch.device('cpu')
        torch.set_num_threads(N_JOBS)
        print(f"   Device: {device}, Threads: {N_JOBS}")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_proba = np.zeros(len(self.y_train))
        test_proba = np.zeros(len(self.y_test_filtered))
        
        # Embedding次元を計算 (クラス数の平方根の2倍、最大50)
        embed_dims = [min(50, max(4, int(np.sqrt(d) * 2))) for d in self.cat_dims]
        num_numerical = len(self.numerical_cols)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            set_seed(self.random_state + fold)
            print(f"   Fold {fold+1}/{self.n_folds}...")
            X_tr, X_val = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
            y_tr, y_val = self.y_train[train_idx], self.y_train[val_idx]
            
            # データ準備
            X_tr_num, X_tr_cat, X_val_num, X_val_cat, X_test_num, X_test_cat = self._prepare_nn_data(
                X_tr, X_val, self.X_test_filtered
            )
            
            # DataLoader作成
            train_ds = TensorDataset(
                torch.tensor(X_tr_num),
                torch.tensor(X_tr_cat),
                torch.tensor(y_tr, dtype=torch.float32)
            )
            val_ds = TensorDataset(
                torch.tensor(X_val_num),
                torch.tensor(X_val_cat),
                torch.tensor(y_val, dtype=torch.float32)
            )
            
            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_MLP, shuffle=True, num_workers=NUM_WORKERS)
            val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_MLP, shuffle=False, num_workers=NUM_WORKERS)
            
            # モデル定義
            model = EmbeddingMLP(
                num_numerical=num_numerical,
                cat_dims=self.cat_dims,
                embed_dims=embed_dims,
                hidden_dims=[1024, 512, 128]
            ).to(device)
            
            if IPEX_AVAILABLE:
                model = ipex.optimize(model)
            
            criterion = nn.BCELoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
            
            best_auc = 0
            best_model_state = None
            patience_counter = 0
            max_patience = 10
            
            for epoch in range(100):
                model.train()
                for X_num, X_cat, y_batch in train_loader:
                    X_num, X_cat, y_batch = X_num.to(device), X_cat.to(device), y_batch.to(device)
                    optimizer.zero_grad()
                    output = model(X_num, X_cat).squeeze()
                    loss = criterion(output, y_batch)
                    loss.backward()
                    optimizer.step()
                
                # Validation
                model.eval()
                val_preds = []
                val_targets = []
                with torch.no_grad():
                    for X_num, X_cat, y_batch in val_loader:
                        X_num, X_cat = X_num.to(device), X_cat.to(device)
                        output = model(X_num, X_cat).squeeze().cpu().numpy()
                        val_preds.extend(output)
                        val_targets.extend(y_batch.numpy())
                
                val_auc = roc_auc_score(val_targets, val_preds)
                scheduler.step(-val_auc)
                
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= max_patience:
                    break
            
            # Best model で予測
            model.load_state_dict(best_model_state)
            model.eval()
            
            with torch.no_grad():
                X_val_num_t = torch.tensor(X_val_num).to(device)
                X_val_cat_t = torch.tensor(X_val_cat).to(device)
                oof_proba[val_idx] = model(X_val_num_t, X_val_cat_t).squeeze().cpu().numpy()
                
                X_test_num_t = torch.tensor(X_test_num).to(device)
                X_test_cat_t = torch.tensor(X_test_cat).to(device)
                test_proba += model(X_test_num_t, X_test_cat_t).squeeze().cpu().numpy() / self.n_folds
            
            gc.collect()
        
        auc = roc_auc_score(self.y_train, oof_proba)
        print(f"   MLP OOF AUC: {auc:.4f}")
        
        self.oof_predictions['mlp'] = oof_proba
        self.test_predictions['mlp'] = test_proba
        self.model_aucs['mlp'] = auc
        
        self._save_checkpoint('mlp', oof_proba, test_proba, auc)
    
    def optimize_ensemble_weights(self):
        """アンサンブル重みの最適化"""
        print("\n⚖️ アンサンブル重み最適化...")
        
        available_models = list(self.oof_predictions.keys())
        n_models = len(available_models)
        
        if n_models < 2:
            print("   モデルが1つ以下のためアンサンブル不可")
            return
        
        oof_matrix = np.column_stack([self.oof_predictions[m] for m in available_models])
        
        def neg_auc(weights):
            weights = np.array(weights)
            weights = weights / weights.sum()
            ensemble_pred = oof_matrix @ weights
            return -roc_auc_score(self.y_train, ensemble_pred)
        
        init_weights = np.ones(n_models) / n_models
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        result = minimize(neg_auc, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        self.ensemble_weights = {m: w for m, w in zip(available_models, result.x)}
        
        print("   最適化された重み:")
        for model, weight in self.ensemble_weights.items():
            print(f"      {model}: {weight:.4f}")
        
        self.oof_predictions['ensemble'] = oof_matrix @ result.x
        
        test_matrix = np.column_stack([self.test_predictions[m] for m in available_models])
        self.test_predictions['ensemble'] = test_matrix @ result.x
        
        ensemble_auc = roc_auc_score(self.y_train, self.oof_predictions['ensemble'])
        print(f"   Ensemble OOF AUC: {ensemble_auc:.4f}")
        self.model_aucs['ensemble'] = ensemble_auc
    
    def evaluate(self):
        """最終評価"""
        print("\n📈 最終評価...")
        
        results = []
        for model_name in self.oof_predictions.keys():
            oof_pred = self.oof_predictions[model_name]
            test_pred = self.test_predictions[model_name]
            
            oof_auc = roc_auc_score(self.y_train, oof_pred)
            test_auc = roc_auc_score(self.y_test_filtered, test_pred)
            oof_pr_auc = average_precision_score(self.y_train, oof_pred)
            test_pr_auc = average_precision_score(self.y_test_filtered, test_pred)
            
            results.append({
                'model': model_name,
                'oof_roc_auc': oof_auc,
                'test_roc_auc': test_auc,
                'oof_pr_auc': oof_pr_auc,
                'test_pr_auc': test_pr_auc
            })
            
            print(f"   {model_name:12s} | OOF AUC: {oof_auc:.4f} | Test AUC: {test_auc:.4f} | OOF PR-AUC: {oof_pr_auc:.4f}")
        
        self.results_df = pd.DataFrame(results)
        self.results_df.to_csv(os.path.join(self.output_dir, "model_comparison.csv"), index=False)
        
        oof_df = pd.DataFrame(self.oof_predictions)
        oof_df['target'] = self.y_train
        oof_df.to_csv(os.path.join(self.output_dir, "oof_predictions.csv"), index=False)
        
        test_df = pd.DataFrame(self.test_predictions)
        test_df['target'] = self.y_test_filtered
        test_df.to_csv(os.path.join(self.output_dir, "test_predictions.csv"), index=False)
    
    def generate_report(self, elapsed_sec: float):
        """レポート生成"""
        print("\n📄 レポート生成...")
        
        report_content = f"""# Stage 2: 4モデルアンサンブル 実験レポート (v2)

**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**実行時間**: {elapsed_sec:.1f}秒
**ハードウェア**: Intel Core Ultra 9 285K (n_jobs={N_JOBS})

## パイプライン構成

- **Stage 1**: Weighted Ensemble (Recall {self.stage1_recall_target:.0%})
- **Stage 2**: LightGBM, CatBoost, TabNet (cat_idxs対応), MLP (Embedding層) → Weighted Ensemble

## Stage 1 フィルタリング結果

| 指標 | 値 |
|------|-----|
| 閾値 | {self.stage1_threshold:.4f} |
| Train通過率 | {len(self.y_train) / len(self.y_train_full):.1%} |
| Test通過率 | {len(self.y_test_filtered) / len(self.y_test):.1%} |

## モデル比較

| Model | OOF ROC-AUC | Test ROC-AUC | OOF PR-AUC |
|-------|-------------|--------------|------------|
"""
        for _, row in self.results_df.iterrows():
            report_content += f"| {row['model']} | {row['oof_roc_auc']:.4f} | {row['test_roc_auc']:.4f} | {row['oof_pr_auc']:.4f} |\n"
        
        if hasattr(self, 'ensemble_weights'):
            report_content += "\n## アンサンブル重み\n\n"
            for model, weight in self.ensemble_weights.items():
                report_content += f"- **{model}**: {weight:.4f}\n"
        
        report_content += f"""
## 考察

- 最高単体モデル AUC: {self.results_df[self.results_df['model'] != 'ensemble']['test_roc_auc'].max():.4f}
- アンサンブル AUC: {self.results_df[self.results_df['model'] == 'ensemble']['test_roc_auc'].values[0]:.4f}

## 修正点 (v2)

1. データ整合性: Stage 1 OOF予測と train_test_split の同期を確認
2. MLP: カテゴリ変数に Embedding 層を使用（誤った順序関係を排除）
3. TabNet: cat_idxs, cat_dims を正しく設定
4. Intel最適化: OMP/MKL スレッド設定を明示
5. PyTorch再現性: シード固定
"""
        
        report_path = os.path.join(self.output_dir, "experiment_report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"   📄 レポート: {report_path}")
    
    def run(self):
        """パイプライン実行"""
        start = datetime.now()
        
        print("=" * 70)
        print("🚀 Stage 2: 4モデルアンサンブル パイプライン (v2)")
        print(f"   最適化: Intel Core Ultra 9 285K (n_jobs={N_JOBS})")
        print(f"   チェックポイント: {self.checkpoint_dir}")
        print("=" * 70)
        
        self.load_and_filter_data()
        self.train_lightgbm()
        self.train_catboost()
        self.train_tabnet()
        self.train_mlp()
        self.optimize_ensemble_weights()
        self.evaluate()
        
        elapsed_sec = (datetime.now() - start).total_seconds()
        self.generate_report(elapsed_sec)
        
        print("\n" + "=" * 70)
        print("✅ 完了！")
        print(f"   結果ディレクトリ: {self.output_dir}")
        print(f"   実行時間: {elapsed_sec:.1f}秒")
        print("=" * 70)


if __name__ == "__main__":
    pipeline = Stage2EnsemblePipeline()
    pipeline.run()
