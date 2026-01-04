"""
MLP vs TabNet vs LightGBM 比較 (病院・交通データ付き)
===================================================
'honhyo_for_analysis_with_traffic_hospital_no_leakage.csv' を使用して
MLPを学習し、既存のTabNet/LightGBMの結果と比較する。

比較条件:
- データソース: honhyo_for_analysis_with_traffic_hospital_no_leakage.csv
- ターゲット: fatal (0=負傷, 1=死亡)
- Train/Test: 80/20 random split (stratified)
- 5-Fold CV
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve, 
    precision_score, recall_score, f1_score, brier_score_loss
)
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- パス設定 ---
DATA_PATH = Path("data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv")
RESULTS_DIR = Path("results/spatio_temporal")
OUTPUT_DIR = RESULTS_DIR / "mlp_hospital_comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ランダムシード
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- モデル定義 ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce = nn.BCEWithLogitsLoss(reduction='none')(inputs, targets)
        p = torch.sigmoid(inputs)
        pt = targets * p + (1 - targets) * (1 - p)
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal_weight = alpha_weight * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
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
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

def load_data():
    print("📂 データ読み込み中 (honhyo_for_analysis_with_traffic_hospital_no_leakage.csv)...")
    df = pd.read_csv(DATA_PATH)
    print(f"   データ: {len(df):,} 行, {len(df.columns)} 列")
    
    target_col = 'fatal'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
        
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])
    if '発生日時' in X.columns:
        X = X.drop(columns=['発生日時'])
    
    print(f"   ターゲット分布: 0={sum(y==0):,}, 1={sum(y==1):,} ({sum(y==1)/len(y)*100:.2f}%)")
    return X, y

def prepare_features(X):
    print("\n🔧 特徴量前処理中...")
    X_numeric = X.copy()
    for col in X_numeric.columns:
        if X_numeric[col].dtype == 'object' or X_numeric[col].dtype.name == 'category':
            X_numeric[col] = pd.Categorical(X_numeric[col]).codes
        else:
            X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce')
    X_numeric = X_numeric.fillna(X_numeric.median())
    return X_numeric.values.astype(np.float32)

def train_mlp_cv(X_train, y_train, n_folds=5):
    print(f"\n🧠 MLP {n_folds}-Fold CV 学習中...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   デバイス: {device}")
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof_proba = np.zeros(len(y_train))
    models = []
    scalers = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"   Fold {fold+1}/{n_folds}...")
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train.iloc[train_idx].values, y_train.iloc[val_idx].values
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        scalers.append(scaler)
        
        train_dataset = TensorDataset(torch.tensor(X_tr_scaled), torch.tensor(y_tr).float().unsqueeze(1))
        val_dataset = TensorDataset(torch.tensor(X_val_scaled), torch.tensor(y_val).float().unsqueeze(1))
        
        train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2048, shuffle=False)
        
        model = MLPClassifier(input_dim=X_tr_scaled.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = FocalLoss()
        
        best_auc = 0
        patience = 10
        patience_counter = 0
        best_state = None
        
        for epoch in range(50): # 高速化のためEpoch削減（十分収束するはず）
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(bx.to(device)), by.to(device))
                loss.backward()
                optimizer.step()
            
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for bx, by in val_loader:
                    preds.extend(torch.sigmoid(model(bx.to(device))).cpu().numpy().flatten())
                    targets.extend(by.numpy().flatten())
            
            val_auc = roc_auc_score(targets, preds)
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        print(f"      Best AUC: {best_auc:.4f}")
        model.load_state_dict(best_state)
        models.append(model)
        
        model.eval()
        with torch.no_grad():
            oof_proba[val_idx] = torch.sigmoid(model(torch.tensor(X_val_scaled).to(device))).cpu().numpy().flatten()
            
    return models, scalers, oof_proba

def main():
    start_time = datetime.now()
    print("="*70 + "\n 🧠 MLP Hospital/Traffic Enriched Comparison \n" + "="*70)
    
    X, y = load_data()
    X_num = prepare_features(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)
    
    models, scalers, oof_proba = train_mlp_cv(X_train, y_train)
    
    # 評価
    auc = roc_auc_score(y_train, oof_proba)
    print(f"\n📊 OOF ROC-AUC: {auc:.4f}")
    
    # テスト評価
    test_proba = np.zeros(len(y_test))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model, scaler in zip(models, scalers):
        model.eval()
        with torch.no_grad():
            test_proba += torch.sigmoid(model(torch.tensor(scaler.transform(X_test)).to(device))).cpu().numpy().flatten() / 5
            
    test_auc = roc_auc_score(y_test, test_proba)
    print(f"📊 Test ROC-AUC: {test_auc:.4f}")
    
    # 結果保存
    results = {'roc_auc': auc, 'test_roc_auc': test_auc}
    with open(OUTPUT_DIR / "results_mlp_hospital.json", 'w') as f:
        json.dump(results, f)
        
    print(f"\n完了: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
