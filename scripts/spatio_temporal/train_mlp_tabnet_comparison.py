"""
MLP vs TabNet 公平比較スクリプト
=================================
TabNetと同じ特徴量セット (honhyo_clean_with_features.csv) を使用して
MLPを学習し、公平な比較を行う。

比較条件を統一:
- 同じデータソース: honhyo_clean_with_features.csv
- 同じTrain/Test分割: 80/20 random split (stratified)
- 同じ5-Fold CV
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
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- パス設定 ---
DATA_PATH = Path("data/processed/honhyo_clean_with_features.csv")
RESULTS_DIR = Path("results/spatio_temporal")
OUTPUT_DIR = RESULTS_DIR / "mlp_tabnet_comparison"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'

# ランダムシード
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --- リーク防止 ---
FORBIDDEN_COLUMNS = [
    '事故内容',
    '人身損傷程度（当事者A）', '人身損傷程度（当事者B）',
    '負傷者数',
    '車両の損壊程度（当事者A）', '車両の損壊程度（当事者B）',
    '車両の衝突部位（当事者A）', '車両の衝突部位（当事者B）',
    'エアバッグの装備（当事者A）', 'エアバッグの装備（当事者B）',
    'サイドエアバッグの装備（当事者A）', 'サイドエアバッグの装備（当事者B）',
]


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
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
    """シンプルな3層MLP"""
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
        
        # 重み初期化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)


def load_data():
    """TabNetと同じデータを読み込む"""
    print("📂 データ読み込み中 (honhyo_clean_with_features.csv)...")
    df = pd.read_csv(DATA_PATH)
    print(f"   データ: {len(df):,} 行, {len(df.columns)} 列")
    
    # ターゲット列
    target_col = '死者数'
    y = (df[target_col] > 0).astype(int)
    
    # 特徴量
    X = df.drop(columns=[target_col])
    if '発生日時' in X.columns:
        X = X.drop(columns=['発生日時'])
    
    # リークチェック
    leaked = [col for col in FORBIDDEN_COLUMNS if col in X.columns]
    if leaked:
        print(f"   ⚠️ リーク警告: {leaked}")
        X = X.drop(columns=leaked)
    
    print(f"   特徴量: {len(X.columns)} 列")
    print(f"   ターゲット分布: 0={sum(y==0):,}, 1={sum(y==1):,} ({sum(y==1)/len(y)*100:.2f}%)")
    
    return X, y


def prepare_features(X):
    """MLP用前処理（スケーリング）"""
    print("\n🔧 特徴量前処理中...")
    
    # すべてを数値に変換
    X_numeric = X.copy()
    
    for col in X_numeric.columns:
        if X_numeric[col].dtype == 'object':
            # カテゴリ列はOrdinalEncoderで変換
            X_numeric[col] = pd.Categorical(X_numeric[col]).codes
        elif X_numeric[col].dtype.name == 'category':
            X_numeric[col] = X_numeric[col].cat.codes
        else:
            X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce')
    
    # 欠損値補完
    X_numeric = X_numeric.fillna(X_numeric.median())
    
    print(f"   特徴量次元: {X_numeric.shape[1]}")
    
    return X_numeric.values.astype(np.float32)


def train_mlp_cv(X_train, y_train, n_folds=5, config=None):
    """5-Fold CVでMLPを学習"""
    print(f"\n🧠 MLP {n_folds}-Fold CV 学習中...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   デバイス: {device}")
    
    if config is None:
        config = {
            'hidden_dim': 128,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 2048,
            'epochs': 100,
            'patience': 15,
        }
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    oof_proba = np.zeros(len(y_train))
    models = []
    scalers = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n   Fold {fold+1}/{n_folds}...")
        
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train.iloc[train_idx].values, y_train.iloc[val_idx].values
        
        # スケーリング
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)
        scalers.append(scaler)
        
        # データローダー
        train_dataset = TensorDataset(
            torch.tensor(X_tr_scaled, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # モデル作成
        model = MLPClassifier(
            input_dim=X_tr_scaled.shape[1],
            hidden_dim=config['hidden_dim'],
            dropout=config['dropout']
        ).to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        criterion = FocalLoss(alpha=0.75, gamma=2.0)
        
        best_val_auc = 0.0
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(config['epochs']):
            # 学習
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
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
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device)
                    outputs = model(batch_x)
                    probs = torch.sigmoid(outputs)
                    val_preds.extend(probs.cpu().numpy().flatten())
                    val_targets.extend(batch_y.numpy().flatten())
            
            val_preds = np.array(val_preds)
            val_targets = np.array(val_targets)
            val_auc = roc_auc_score(val_targets, val_preds)
            
            scheduler.step(val_auc)
            
            # 改善チェック
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                print(f"      Epoch {epoch:3d}: Loss={train_loss:.4f}, Val AUC={val_auc:.4f}")
            
            if patience_counter >= config['patience']:
                print(f"      Early stopping at epoch {epoch}")
                break
        
        # ベストモデルをロード
        model.load_state_dict(best_model_state)
        models.append(model)
        
        # OOF予測
        model.eval()
        with torch.no_grad():
            X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
            outputs = model(X_val_t)
            oof_proba[val_idx] = torch.sigmoid(outputs).cpu().numpy().flatten()
        
        print(f"      Fold {fold+1} Best AUC: {best_val_auc:.4f}")
    
    return models, scalers, oof_proba


def evaluate_metrics(y_true, y_pred_proba):
    """詳細な評価指標を計算"""
    # 基本指標
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    brier = brier_score_loss(y_true, y_pred_proba)
    
    # Top-k Precision
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    top_k_results = {}
    for k in [100, 500, 1000]:
        if k <= len(y_true):
            top_k_idx = sorted_indices[:k]
            top_k_precision = y_true.iloc[top_k_idx].sum() / k
            top_k_results[f'precision_at_{k}'] = float(top_k_precision)
    
    # 特定Recallでの閾値とPrecision
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_true, y_pred_proba)
    recall_targets = {}
    for target_recall in [0.99, 0.95, 0.90]:
        idx = np.searchsorted(recall_curve[::-1], target_recall)
        if idx < len(thresholds):
            thresh = thresholds[::-1][idx] if idx < len(thresholds) else 0.0
            prec = precision_curve[::-1][idx] if idx < len(precision_curve) else 0.0
            recall_targets[f'threshold_at_recall_{int(target_recall*100)}'] = float(thresh)
            recall_targets[f'precision_at_recall_{int(target_recall*100)}'] = float(prec)
    
    # Best F1
    f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-15)
    best_f1_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_idx]
    best_thresh = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
    best_prec = precision_curve[best_f1_idx]
    best_rec = recall_curve[best_f1_idx]
    
    metrics = {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'brier_score': brier,
        'best_f1': best_f1,
        'best_f1_threshold': best_thresh,
        'best_f1_precision': best_prec,
        'best_f1_recall': best_rec,
        **top_k_results,
        **recall_targets,
    }
    
    return metrics


def compare_with_tabnet_and_lgb(mlp_metrics):
    """TabNetとLightGBM結果と比較"""
    print("\n" + "=" * 70)
    print(" 🔄 MLP vs TabNet vs LightGBM 比較 (同一データ)")
    print("=" * 70)
    
    # 比較データ
    tabnet_results = {
        'roc_auc': 0.8393,
        'precision_at_recall_95': 0.0258,
    }
    
    lgb_results = {
        'roc_auc': 0.8661,
        'precision_at_recall_95': 0.0137,
    }
    
    comparisons = [
        ('ROC-AUC', 'roc_auc'),
        ('Recall 95% Precision', 'precision_at_recall_95'),
        ('PR-AUC', 'pr_auc'),
        ('Best F1', 'best_f1'),
    ]
    
    print(f"\n   {'指標':<28} {'TabNet':<12} {'LightGBM':<12} {'MLP':<12}")
    print("   " + "-" * 64)
    
    comparison_results = {}
    for name, key in comparisons:
        tabnet_val = tabnet_results.get(key, 0)
        lgb_val = lgb_results.get(key, 0)
        mlp_val = mlp_metrics.get(key, 0)
        print(f"   {name:<28} {tabnet_val:<12.4f} {lgb_val:<12.4f} {mlp_val:<12.4f}")
        comparison_results[key] = {
            'tabnet': tabnet_val,
            'lgb': lgb_val,
            'mlp': mlp_val
        }
    
    return comparison_results


def save_results(models, scalers, oof_proba, y_train, metrics, comparison_results):
    """結果を保存"""
    print("\n💾 結果を保存中...")
    
    # モデル保存
    for i, model in enumerate(models):
        torch.save(model.state_dict(), OUTPUT_DIR / f"mlp_fold{i+1}.pt")
    
    # メトリクス保存
    results = {
        'model_type': 'mlp',
        'data_source': 'honhyo_clean_with_features.csv',
        'comparison_note': 'TabNetと同じ特徴量セットで学習',
        'oof_metrics': metrics,
        'comparison_3way': comparison_results,
    }
    with open(OUTPUT_DIR / "results_mlp_tabnet_comparison.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # OOF予測保存
    oof_df = pd.DataFrame({
        'true_label': y_train.values,
        'prob': oof_proba
    })
    oof_df.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)
    
    print(f"   結果保存先: {OUTPUT_DIR}")


def main():
    start_time = datetime.now()
    
    print("=" * 70)
    print(" 🧠 MLP vs TabNet 公平比較")
    print(" (同じデータ: honhyo_clean_with_features.csv)")
    print("=" * 70)
    
    # 1. データ読み込み
    X, y = load_data()
    
    # 2. 前処理
    X_numeric = prepare_features(X)
    
    # 3. Train/Test分割 (TabNetと同じ: 80/20)
    print("\n✂️ Train/Test分割 (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"   Train: {len(y_train):,} (Fatal: {y_train.sum():,})")
    print(f"   Test:  {len(y_test):,} (Fatal: {y_test.sum():,})")
    
    # 4. 5-Fold CVで学習
    models, scalers, oof_proba = train_mlp_cv(X_train, y_train)
    
    # 5. OOF評価
    print("\n📊 OOF評価 (Cross Validation)...")
    oof_metrics = evaluate_metrics(y_train, oof_proba)
    
    print(f"\n   ROC-AUC: {oof_metrics['roc_auc']:.4f}")
    print(f"   PR-AUC:  {oof_metrics['pr_auc']:.4f}")
    print(f"   Best F1: {oof_metrics['best_f1']:.4f} (閾値: {oof_metrics['best_f1_threshold']:.4f})")
    print(f"   Recall 95% Precision: {oof_metrics.get('precision_at_recall_95', 0):.4f}")
    
    # 6. テストセット評価
    print("\n📊 テストセット評価...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_proba = np.zeros(len(y_test))
    
    for model, scaler in zip(models, scalers):
        X_test_scaled = scaler.transform(X_test)
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_test_t)
            test_proba += torch.sigmoid(outputs).cpu().numpy().flatten() / len(models)
    
    test_metrics = evaluate_metrics(y_test, test_proba)
    print(f"   Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"   Test PR-AUC:  {test_metrics['pr_auc']:.4f}")
    print(f"   Test Best F1: {test_metrics['best_f1']:.4f}")
    print(f"   Test Recall 95% Precision: {test_metrics.get('precision_at_recall_95', 0):.4f}")
    
    # 7. 比較
    comparison_results = compare_with_tabnet_and_lgb(oof_metrics)
    
    # 8. 結果保存
    save_results(models, scalers, oof_proba, y_train, oof_metrics, comparison_results)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n🎉 MLP学習完了！")
    
    # サマリー出力
    print("\n" + "=" * 70)
    print(" 📋 サマリー")
    print("=" * 70)
    print(f"   データ: honhyo_clean_with_features.csv ({len(X):,} 件)")
    print(f"   特徴量: {X_numeric.shape[1]} 次元")
    print(f"   MLP OOF ROC-AUC: {oof_metrics['roc_auc']:.4f}")
    print(f"   MLP OOF PR-AUC:  {oof_metrics['pr_auc']:.4f}")
    print(f"   MLP OOF Recall95%時Precision: {oof_metrics.get('precision_at_recall_95', 0):.4f}")
    print(f"   MLP Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    print(f"   所要時間: {elapsed:.1f}秒")


if __name__ == "__main__":
    main()
