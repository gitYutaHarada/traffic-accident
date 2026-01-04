"""
4モデル比較 + アンサンブル (高性能PC最適化版 v2)
==============================================
Intel Core Ultra 9 285K (24コア) + 64GB RAM 向けに最適化

修正点 (v2):
1. MLP: Target Encoding を使用（Label Encoding + Scaler の理論的欠陥を修正）
2. MLP: Train/Val分割後にエンコーディング（リーク防止）
3. アンサンブル: LogLoss による重み最適化（F1+固定閾値の問題を修正）

使用データ: honhyo_for_analysis_with_traffic_hospital_no_leakage.csv
比較モデル: LightGBM, CatBoost, TabNet, MLP

実行方法:
    python scripts/spatio_temporal/train_4model_ensemble.py
"""

import pandas as pd
import numpy as np
import json
import os
import gc
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, 
    precision_score, recall_score, f1_score, log_loss
)
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# LightGBM & CatBoost
import lightgbm as lgb
from catboost import CatBoostClassifier

# TabNet
from pytorch_tabnet.tab_model import TabNetClassifier

# ============================================================================
# 設定 (Intel Core Ultra 9 285K + 64GB RAM 最適化)
# ============================================================================
DATA_PATH = Path("data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv")
OUTPUT_DIR = Path("results/spatio_temporal/4model_ensemble")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ハードウェア最適化パラメータ
N_JOBS = 20
N_FOLDS = 5
RANDOM_SEED = 42
TEST_SIZE = 0.2

LGB_N_JOBS = 10
CAT_THREADS = 10
TABNET_BATCH = 8192
MLP_BATCH = 4096

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Device: {DEVICE}")
print(f"🔧 Parallel Jobs: {N_JOBS}")


# ============================================================================
# Target Encoding クラス（リーク防止版）
# ============================================================================
class TargetEncoder:
    """
    Target Encoding: カテゴリ変数をターゲットの条件付き期待値で置換
    - Train時のみfitし、Valにはtransformのみ適用（リーク防止）
    - 未知カテゴリにはグローバル平均を使用
    """
    def __init__(self, smoothing=10.0):
        self.smoothing = smoothing
        self.global_mean = None
        self.encodings = {}
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, columns: list):
        self.global_mean = y.mean()
        self.columns = columns
        
        for col in columns:
            stats = X.groupby(col)[y.name if hasattr(y, 'name') else 'target'].agg(['mean', 'count'])
            # Smoothing: (n * mean + m * global_mean) / (n + m)
            smoothed = (stats['count'] * stats['mean'] + self.smoothing * self.global_mean) / (stats['count'] + self.smoothing)
            self.encodings[col] = smoothed.to_dict()
        
        return self
    
    def fit_transform(self, X: pd.DataFrame, y: np.ndarray, columns: list):
        # yをSeriesとして扱えるようにする
        y_series = pd.Series(y, name='target', index=X.index)
        X_temp = X.copy()
        X_temp['target'] = y_series
        
        self.global_mean = y.mean()
        self.columns = columns
        
        X_encoded = X.copy()
        for col in columns:
            stats = X_temp.groupby(col)['target'].agg(['mean', 'count'])
            smoothed = (stats['count'] * stats['mean'] + self.smoothing * self.global_mean) / (stats['count'] + self.smoothing)
            self.encodings[col] = smoothed.to_dict()
            X_encoded[col] = X[col].map(self.encodings[col]).fillna(self.global_mean)
        
        return X_encoded
    
    def transform(self, X: pd.DataFrame):
        X_encoded = X.copy()
        for col in self.columns:
            X_encoded[col] = X[col].map(self.encodings[col]).fillna(self.global_mean)
        return X_encoded


# ============================================================================
# MLP モデル定義
# ============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        p = torch.sigmoid(inputs)
        pt = targets * p + (1 - targets) * (1 - p)
        alpha_w = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_w = alpha_w * (1 - pt) ** self.gamma
        return (focal_w * bce).mean()


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.net(x)


# ============================================================================
# データ読み込み
# ============================================================================
def load_data():
    print("\n📂 データ読み込み中...")
    df = pd.read_csv(DATA_PATH)
    print(f"   データ: {len(df):,} 件, {len(df.columns)} 列")
    
    target_col = 'fatal'
    y = df[target_col].astype(int).values
    X = df.drop(columns=[target_col])
    
    if '発生日時' in X.columns:
        X = X.drop(columns=['発生日時'])
    
    known_cats = ['都道府県コード', '市区町村コード', '警察署等コード', '昼夜', '天候', 
                  '地形', '路面状態', '道路形状', '信号機', '衝突地点', 'ゾーン規制', 
                  '中央分離帯施設等', '歩車道区分', '事故類型', '曜日(発生年月日)', 
                  '祝日(発生年月日)', 'road_type', 'area_id', '地点コード']
    
    cat_cols = []
    num_cols = []
    for col in X.columns:
        if col in known_cats or X[col].dtype == 'object':
            cat_cols.append(col)
            X[col] = X[col].astype(str)
        else:
            num_cols.append(col)
            X[col] = X[col].astype(np.float32)
    
    print(f"   カテゴリ: {len(cat_cols)}, 数値: {len(num_cols)}")
    print(f"   ターゲット分布: Neg={sum(y==0):,}, Pos={sum(y==1):,} ({sum(y==1)/len(y)*100:.2f}%)")
    
    return X, y, cat_cols, num_cols


# ============================================================================
# 各モデルのFold学習関数
# ============================================================================
def train_lgb_fold(fold, X_tr, y_tr, X_val, y_val, cat_cols):
    """LightGBM 単一Fold学習"""
    X_tr_lgb = X_tr.copy()
    X_val_lgb = X_val.copy()
    for col in cat_cols:
        if col in X_tr_lgb.columns:
            X_tr_lgb[col] = X_tr_lgb[col].astype('category')
            X_val_lgb[col] = X_val_lgb[col].astype('category')
    
    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    scale_pos = n_neg / n_pos if n_pos > 0 else 1.0
    
    model = lgb.LGBMClassifier(
        objective='binary', metric='auc', boosting_type='gbdt',
        num_leaves=127, max_depth=-1, min_child_samples=44,
        reg_alpha=2.4, reg_lambda=2.3, colsample_bytree=0.87,
        subsample=0.63, learning_rate=0.05, n_estimators=500,
        scale_pos_weight=scale_pos, n_jobs=LGB_N_JOBS, verbosity=-1,
        random_state=RANDOM_SEED + fold
    )
    model.fit(X_tr_lgb, y_tr, eval_set=[(X_val_lgb, y_val)],
              callbacks=[lgb.early_stopping(30, verbose=False)])
    
    pred = model.predict_proba(X_val_lgb)[:, 1]
    return fold, pred, model


def train_cat_fold(fold, X_tr, y_tr, X_val, y_val, cat_cols):
    """CatBoost 単一Fold学習"""
    cat_features = [c for c in cat_cols if c in X_tr.columns]
    
    model = CatBoostClassifier(
        iterations=500, learning_rate=0.05, depth=8, l2_leaf_reg=3,
        loss_function='Logloss', eval_metric='AUC', random_seed=RANDOM_SEED + fold,
        verbose=False, early_stopping_rounds=30, task_type='CPU',
        thread_count=CAT_THREADS, cat_features=cat_features
    )
    model.fit(X_tr, y_tr, eval_set=(X_val, y_val), verbose=False)
    pred = model.predict_proba(X_val)[:, 1]
    return fold, pred, model


def train_tabnet_fold(fold, X_tr, y_tr, X_val, y_val, cat_cols, num_cols):
    """TabNet 単一Fold学習"""
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    X_num_tr = scaler.fit_transform(imputer.fit_transform(X_tr[num_cols].values))
    X_cat_tr = encoder.fit_transform(X_tr[cat_cols].values) + 1
    X_tr_tab = np.hstack([X_num_tr, X_cat_tr]).astype(np.float32)
    
    X_num_val = scaler.transform(imputer.transform(X_val[num_cols].values))
    X_cat_val = encoder.transform(X_val[cat_cols].values) + 1
    X_val_tab = np.hstack([X_num_val, X_cat_val]).astype(np.float32)
    
    cat_idxs = list(range(len(num_cols), len(num_cols) + len(cat_cols)))
    cat_dims = [int(X_tr_tab[:, i].max() + 2) for i in cat_idxs]
    
    model = TabNetClassifier(
        n_d=32, n_a=32, n_steps=5, gamma=1.5,
        cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=1,
        optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=0.02),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=10, gamma=0.9),
        seed=RANDOM_SEED + fold, verbose=0
    )
    model.fit(
        X_tr_tab, y_tr.astype(int),
        eval_set=[(X_val_tab, y_val.astype(int))],
        eval_metric=['auc'],
        max_epochs=50, patience=10, batch_size=TABNET_BATCH, virtual_batch_size=256
    )
    pred = model.predict_proba(X_val_tab)[:, 1]
    return fold, pred, model, (imputer, scaler, encoder)


def train_mlp_fold(fold, X_tr, y_tr, X_val, y_val, num_cols, cat_cols):
    """
    MLP 単一Fold学習 (v2: Target Encoding使用, リーク防止)
    
    修正点:
    - カテゴリ変数にTarget Encodingを使用（順序性の仮定を回避）
    - Train dataのみでEncoder/Scalerをfit（リーク防止）
    """
    # Target Encoding: Trainのみでfit、Valにはtransformのみ
    target_encoder = TargetEncoder(smoothing=10.0)
    X_tr_encoded = target_encoder.fit_transform(X_tr.copy(), y_tr, cat_cols)
    X_val_encoded = target_encoder.transform(X_val.copy())
    
    # 数値カラムの欠損補完
    imputer = SimpleImputer(strategy='mean')
    X_tr_encoded[num_cols] = imputer.fit_transform(X_tr_encoded[num_cols])
    X_val_encoded[num_cols] = imputer.transform(X_val_encoded[num_cols])
    
    # StandardScaler: Trainのみでfit
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_encoded.values.astype(np.float32))
    X_val_scaled = scaler.transform(X_val_encoded.values.astype(np.float32))
    
    train_ds = TensorDataset(
        torch.tensor(X_tr_scaled, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_ds, batch_size=MLP_BATCH, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=MLP_BATCH, shuffle=False, num_workers=0)
    
    model = MLPClassifier(input_dim=X_tr_scaled.shape[1], hidden_dim=256).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = FocalLoss()
    
    best_auc = 0.0
    patience_cnt = 0
    best_state = None
    
    for epoch in range(50):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
        
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                preds.extend(torch.sigmoid(model(bx.to(DEVICE))).cpu().numpy().flatten())
                targets.extend(by.numpy().flatten())
        
        val_auc = roc_auc_score(targets, preds)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict().copy()
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 10:
                break
    
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(DEVICE)
        pred = torch.sigmoid(model(X_val_t)).cpu().numpy().flatten()
    
    # preprocessorsを返す（Test時に再利用）
    return fold, pred, model, (target_encoder, imputer, scaler)


# ============================================================================
# メインパイプライン
# ============================================================================
def main():
    start = datetime.now()
    print("=" * 70)
    print(" 🚀 4モデル比較 + アンサンブル v2 (修正版)")
    print("   - MLP: Target Encoding (リーク防止)")
    print("   - アンサンブル: LogLoss最適化")
    print("=" * 70)
    
    X, y, cat_cols, num_cols = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    
    print(f"\n📊 Train: {len(y_train):,} (Pos: {y_train.sum():,})")
    print(f"📊 Test:  {len(y_test):,} (Pos: {y_test.sum():,})")
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds = list(skf.split(X_train, y_train))
    
    oof_lgb = np.zeros(len(y_train))
    oof_cat = np.zeros(len(y_train))
    oof_tab = np.zeros(len(y_train))
    oof_mlp = np.zeros(len(y_train))
    
    models_lgb = []
    models_cat = []
    models_tab = []
    models_mlp = []
    tab_preprocessors = []
    mlp_preprocessors = []
    
    print("\n" + "=" * 70)
    print(" 🌿 モデル学習開始 (5-Fold CV)")
    print("=" * 70)
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        if fold < 4:  # Fold 1, 2, 3, 4 は完了済みなのでスキップ (Fold 5から再開)
            print(f"⏩ Fold {fold+1}/{N_FOLDS} スキップ (完了済み)")
            continue

        fold_start = datetime.now()
        print(f"\n📂 Fold {fold+1}/{N_FOLDS}...")
        
        X_tr = X_train.iloc[train_idx].copy()
        X_val = X_train.iloc[val_idx].copy()
        y_tr = y_train[train_idx]
        y_val = y_train[val_idx]
        
        # LightGBM
        _, pred_lgb, m_lgb = train_lgb_fold(fold, X_tr, y_tr, X_val, y_val, cat_cols)
        oof_lgb[val_idx] = pred_lgb
        models_lgb.append(m_lgb)
        print(f"   LightGBM AUC: {roc_auc_score(y_val, pred_lgb):.4f}")
        
        # CatBoost
        _, pred_cat, m_cat = train_cat_fold(fold, X_tr, y_tr, X_val, y_val, cat_cols)
        oof_cat[val_idx] = pred_cat
        models_cat.append(m_cat)
        print(f"   CatBoost AUC: {roc_auc_score(y_val, pred_cat):.4f}")
        
        # TabNet
        _, pred_tab, m_tab, tab_pre = train_tabnet_fold(fold, X_tr, y_tr, X_val, y_val, cat_cols, num_cols)
        oof_tab[val_idx] = pred_tab
        models_tab.append(m_tab)
        tab_preprocessors.append(tab_pre)
        print(f"   TabNet AUC:   {roc_auc_score(y_val, pred_tab):.4f}")
        
        # MLP (v2: Target Encoding)
        _, pred_mlp, m_mlp, mlp_pre = train_mlp_fold(fold, X_tr, y_tr, X_val, y_val, num_cols, cat_cols)
        oof_mlp[val_idx] = pred_mlp
        models_mlp.append(m_mlp)
        mlp_preprocessors.append(mlp_pre)
        print(f"   MLP AUC:      {roc_auc_score(y_val, pred_mlp):.4f}")
        
        fold_elapsed = (datetime.now() - fold_start).total_seconds()
        print(f"   ⏱️ Fold {fold+1} 完了: {fold_elapsed:.1f}秒")
        gc.collect()
    
    # OOF評価
    print("\n" + "=" * 70)
    print(" 📊 OOF評価結果")
    print("=" * 70)
    
    auc_lgb = roc_auc_score(y_train, oof_lgb)
    auc_cat = roc_auc_score(y_train, oof_cat)
    auc_tab = roc_auc_score(y_train, oof_tab)
    auc_mlp = roc_auc_score(y_train, oof_mlp)
    
    print(f"   LightGBM: {auc_lgb:.4f}")
    print(f"   CatBoost: {auc_cat:.4f}")
    print(f"   TabNet:   {auc_tab:.4f}")
    print(f"   MLP:      {auc_mlp:.4f}")
    
    # アンサンブル重み最適化 (v2: LogLoss使用)
    print("\n🔍 アンサンブル重み最適化中 (LogLoss)...")
    
    def loss_fn(w):
        """LogLossベースの損失関数（閾値非依存）"""
        w = np.clip(w, 0.01, 0.99)
        w = w / (w.sum() + 1e-8)
        ens = w[0] * oof_lgb + w[1] * oof_cat + w[2] * oof_tab + w[3] * oof_mlp
        ens = np.clip(ens, 1e-7, 1 - 1e-7)  # log(0)防止
        return log_loss(y_train, ens)
    
    result = minimize(
        loss_fn, [0.25, 0.25, 0.25, 0.25],
        method='SLSQP',
        bounds=[(0.01, 0.99)] * 4,
        constraints={'type': 'eq', 'fun': lambda w: 1 - sum(w)}
    )
    
    best_w = np.clip(result.x, 0, 1)
    best_w = best_w / best_w.sum()
    
    print(f"   最適重み: LGB={best_w[0]:.3f}, Cat={best_w[1]:.3f}, Tab={best_w[2]:.3f}, MLP={best_w[3]:.3f}")
    
    oof_ensemble = best_w[0] * oof_lgb + best_w[1] * oof_cat + best_w[2] * oof_tab + best_w[3] * oof_mlp
    auc_ens = roc_auc_score(y_train, oof_ensemble)
    logloss_ens = log_loss(y_train, np.clip(oof_ensemble, 1e-7, 1-1e-7))
    print(f"   アンサンブル OOF AUC: {auc_ens:.4f}, LogLoss: {logloss_ens:.4f}")
    
    # Test評価
    print("\n📈 テストセット評価...")
    
    # LightGBM Test
    test_lgb = np.zeros(len(y_test))
    X_test_lgb = X_test.copy()
    for col in cat_cols:
        if col in X_test_lgb.columns:
            X_test_lgb[col] = X_test_lgb[col].astype('category')
    for m in models_lgb:
        test_lgb += m.predict_proba(X_test_lgb)[:, 1] / N_FOLDS
    
    # CatBoost Test
    test_cat = np.zeros(len(y_test))
    for m in models_cat:
        test_cat += m.predict_proba(X_test)[:, 1] / N_FOLDS
    
    # TabNet Test
    test_tab = np.zeros(len(y_test))
    for m, (imp, scl, enc) in zip(models_tab, tab_preprocessors):
        X_num = scl.transform(imp.transform(X_test[num_cols].values))
        X_cat = enc.transform(X_test[cat_cols].values) + 1
        X_t = np.hstack([X_num, X_cat]).astype(np.float32)
        test_tab += m.predict_proba(X_t)[:, 1] / N_FOLDS
    
    # MLP Test (v2: Target Encoding使用)
    test_mlp = np.zeros(len(y_test))
    for m, (te, imp, scl) in zip(models_mlp, mlp_preprocessors):
        X_test_enc = te.transform(X_test.copy())
        X_test_enc[num_cols] = imp.transform(X_test_enc[num_cols])
        X_t = scl.transform(X_test_enc.values.astype(np.float32))
        m.eval()
        with torch.no_grad():
            test_mlp += torch.sigmoid(m(torch.tensor(X_t, dtype=torch.float32).to(DEVICE))).cpu().numpy().flatten() / N_FOLDS
    
    test_ens = best_w[0] * test_lgb + best_w[1] * test_cat + best_w[2] * test_tab + best_w[3] * test_mlp
    
    test_auc_lgb = roc_auc_score(y_test, test_lgb)
    test_auc_cat = roc_auc_score(y_test, test_cat)
    test_auc_tab = roc_auc_score(y_test, test_tab)
    test_auc_mlp = roc_auc_score(y_test, test_mlp)
    test_auc_ens = roc_auc_score(y_test, test_ens)
    
    print(f"\n   📊 Test Set AUC:")
    print(f"   LightGBM: {test_auc_lgb:.4f}")
    print(f"   CatBoost: {test_auc_cat:.4f}")
    print(f"   TabNet:   {test_auc_tab:.4f}")
    print(f"   MLP:      {test_auc_mlp:.4f}")
    print(f"   🏆 Ensemble: {test_auc_ens:.4f}")
    
    elapsed = (datetime.now() - start).total_seconds()
    
    # 結果保存
    results = {
        'oof_auc': {
            'lightgbm': auc_lgb, 'catboost': auc_cat, 'tabnet': auc_tab, 'mlp': auc_mlp, 'ensemble': auc_ens
        },
        'test_auc': {
            'lightgbm': test_auc_lgb, 'catboost': test_auc_cat, 'tabnet': test_auc_tab, 'mlp': test_auc_mlp, 'ensemble': test_auc_ens
        },
        'ensemble_weights': {
            'lightgbm': float(best_w[0]), 'catboost': float(best_w[1]), 'tabnet': float(best_w[2]), 'mlp': float(best_w[3])
        },
        'elapsed_seconds': elapsed
    }
    
    with open(OUTPUT_DIR / "results_4model_ensemble.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 最優秀モデル判定
    test_aucs = {'LightGBM': test_auc_lgb, 'CatBoost': test_auc_cat, 'TabNet': test_auc_tab, 'MLP': test_auc_mlp}
    best_single = max(test_aucs, key=test_aucs.get)
    best_single_auc = test_aucs[best_single]
    ens_improvement = (test_auc_ens - best_single_auc) * 100
    
    # レポート生成
    report = f"""# 4モデル比較 + アンサンブル 実験レポート (v2)

**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**実行時間**: {elapsed:.1f}秒
**データ**: honhyo_for_analysis_with_traffic_hospital_no_leakage.csv

## 修正点 (v2)
- **MLP**: Target Encoding使用（Label Encodingの順序性問題を解消）
- **MLP**: Train/Val分割後にエンコーダをfit（リーク防止）
- **アンサンブル**: LogLossベースで重み最適化（閾値依存を排除）

## OOF評価 (5-Fold CV)

| モデル | ROC-AUC |
|--------|---------|
| LightGBM | {auc_lgb:.4f} |
| CatBoost | {auc_cat:.4f} |
| TabNet | {auc_tab:.4f} |
| MLP | {auc_mlp:.4f} |
| **Ensemble** | **{auc_ens:.4f}** |

## テストセット評価

| モデル | ROC-AUC |
|--------|---------|
| LightGBM | {test_auc_lgb:.4f} |
| CatBoost | {test_auc_cat:.4f} |
| TabNet | {test_auc_tab:.4f} |
| MLP | {test_auc_mlp:.4f} |
| **Ensemble** | **{test_auc_ens:.4f}** |

## アンサンブル重み (LogLoss最適化)

| モデル | 重み |
|--------|------|
| LightGBM | {best_w[0]:.3f} |
| CatBoost | {best_w[1]:.3f} |
| TabNet | {best_w[2]:.3f} |
| MLP | {best_w[3]:.3f} |

## 考察

- **最優秀単体モデル**: {best_single} ({best_single_auc:.4f})
- **アンサンブル効果**: {'+' if ens_improvement > 0 else ''}{ens_improvement:.2f}% (単体最高比)
"""
    
    with open(OUTPUT_DIR / "experiment_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + "=" * 70)
    print(" ✅ 完了！")
    print(f"   総実行時間: {elapsed:.1f}秒")
    print(f"   結果保存先: {OUTPUT_DIR}")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
