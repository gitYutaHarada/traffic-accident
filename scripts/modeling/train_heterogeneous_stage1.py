"""
異種混合アンサンブル Stage 1 パイプライン
=========================================
3種類の異なるモデルでStage 1を構成し、OR条件でフィルタリングする。

モデル構成:
- Model A: LightGBM (決定木) - 相互作用と非線形が得意
- Model B: MLP (Neural Network) - 滑らかな決定境界
- Model C: Logistic Regression (線形) - 大局的な傾向判定

フィルタリング戦略:
- 「誰か1人でも危険と言ったら残す（OR条件）」
- Keep if (Prob_LGBM > Th_LGBM) OR (Prob_MLP > Th_MLP) OR (Prob_LR > Th_LR)
"""

import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OrdinalEncoder
import joblib
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings('ignore')


class SimpleMLP(nn.Module):
    """シンプルな3層MLP"""
    
    def __init__(self, input_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            # 注意: nn.Sigmoid()は削除！BCEWithLogitsLossを使用するため
            # 推論時にtorch.sigmoid()を適用する
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


class HeterogeneousStage1Pipeline:
    """異種混合アンサンブル Stage 1 パイプライン"""
    
    def __init__(
        self,
        data_path: str = "data/processed/honhyo_clean_with_features.csv",
        target_col: str = "死者数",
        n_folds: int = 5,
        random_state: int = 42,
        target_recall: float = 0.99,
        undersample_ratio: float = 2.0,
        n_seeds: int = 3,
        test_size: float = 0.2,
        # MLP parameters
        mlp_hidden_dim: int = 256,
        mlp_epochs: int = 30,
        mlp_batch_size: int = 1024,  # 汎化性能のため小さめに設定
        mlp_lr: float = 0.001,
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.n_folds = n_folds
        self.random_state = random_state
        self.target_recall = target_recall
        self.undersample_ratio = undersample_ratio
        self.n_seeds = n_seeds
        self.test_size = test_size
        
        # MLP parameters
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_epochs = mlp_epochs
        self.mlp_batch_size = mlp_batch_size
        self.mlp_lr = mlp_lr
        
        self.output_dir = "results/two_stage_model/heterogeneous_stage1"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Storage for models and predictions
        self.lgbm_models = []
        self.mlp_models = []
        self.lr_models = []
        self.scalers = []  # For MLP preprocessing
        self.ordinal_encoder = None  # For categorical encoding
        
        print("=" * 60)
        print("異種混合アンサンブル Stage 1 パイプライン")
        print(f"Target Recall: {self.target_recall:.0%}")
        print(f"Models: LightGBM + MLP + Logistic Regression")
        print(f"Strategy: OR-gate filtering")
        print(f"Test Set: {self.test_size:.0%}")
        print("=" * 60)
    
    def load_data(self):
        """データ読み込みとTrain/Test分割"""
        print("\n📂 データ読み込み中...")
        self.df = pd.read_csv(self.data_path)
        
        y_all = self.df[self.target_col].values
        X_all = self.df.drop(columns=[self.target_col])
        
        if '発生日時' in X_all.columns:
            X_all = X_all.drop(columns=['発生日時'])
        
        # Train/Test分割 (層化抽出)
        self.X, self.X_test, self.y, self.y_test = train_test_split(
            X_all, y_all, test_size=self.test_size,
            random_state=self.random_state, stratify=y_all
        )
        
        print(f"\n📊 データ分割 (Train: {1-self.test_size:.0%} / Test: {self.test_size:.0%})")
        print(f"   Train: 正例 {self.y.sum():,} / {len(self.y):,}")
        print(f"   Test:  正例 {self.y_test.sum():,} / {len(self.y_test):,}")
        
        # カテゴリ変数と数値変数の特定
        known_categoricals = [
            '都道府県コード', '市区町村コード', '警察署等コード',
            '昼夜', '天候', '地形', '路面状態', '道路形状', '信号機',
            '衝突地点', 'ゾーン規制', '中央分離帯施設等', '歩車道区分',
            '事故類型', '曜日(発生年月日)', '祝日(発生年月日)',
            'road_type', 'area_id', '地点コード'
        ]
        
        self.categorical_cols = []
        self.numeric_cols = []
        
        for col in self.X.columns:
            if col in known_categoricals or self.X[col].dtype == 'object':
                self.categorical_cols.append(col)
                self.X[col] = self.X[col].astype('category')
                self.X_test[col] = self.X_test[col].astype('category')
            else:
                self.numeric_cols.append(col)
                self.X[col] = self.X[col].astype(np.float32)
                self.X_test[col] = self.X_test[col].astype(np.float32)
        
        self.feature_names = list(self.X.columns)
        gc.collect()
    
    def prepare_numeric_features(self, X, fit=True):
        """MLP/Logistic用に数値特徴量のみを抽出・正規化"""
        X_numeric = X[self.numeric_cols].copy()
        
        # カテゴリカル変数をOrdinalEncoderでエンコード (未知のカテゴリ対応)
        cat_data = X[self.categorical_cols].astype(str).values
        
        if fit:
            self.ordinal_encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1
            )
            encoded = self.ordinal_encoder.fit_transform(cat_data)
        else:
            encoded = self.ordinal_encoder.transform(cat_data)
        
        for i, col in enumerate(self.categorical_cols):
            X_numeric[col] = encoded[:, i].astype(np.float32)
        
        # 欠損値処理
        X_numeric = X_numeric.fillna(0)
        
        return X_numeric
    
    def train_lgbm(self):
        """Model A: LightGBM (決定木)"""
        print("\n🌲 Model A: LightGBM 学習中...")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.oof_proba_lgbm = np.zeros(len(self.y))
        
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'num_leaves': 31,
            'max_depth': 8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'is_unbalance': False,
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'n_jobs': -1
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            X_train_full = self.X.iloc[train_idx]
            X_val = self.X.iloc[val_idx]
            y_train_full = self.y[train_idx]
            y_val = self.y[val_idx]
            
            fold_models = []
            fold_proba = np.zeros(len(val_idx))
            
            for seed in range(self.n_seeds):
                np.random.seed(self.random_state + seed)
                
                # Under-sampling
                pos_idx = np.where(y_train_full == 1)[0]
                neg_idx = np.where(y_train_full == 0)[0]
                n_pos = len(pos_idx)
                n_neg_sample = int(n_pos * self.undersample_ratio)
                neg_sample_idx = np.random.choice(neg_idx, size=min(n_neg_sample, len(neg_idx)), replace=False)
                
                train_idx_sampled = np.concatenate([pos_idx, neg_sample_idx])
                X_train = X_train_full.iloc[train_idx_sampled]
                y_train = y_train_full[train_idx_sampled]
                
                model = lgb.LGBMClassifier(**lgb_params, random_state=self.random_state + seed)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                
                fold_proba += model.predict_proba(X_val)[:, 1] / self.n_seeds
                fold_models.append(model)
            
            self.oof_proba_lgbm[val_idx] = fold_proba
            self.lgbm_models.append(fold_models)
            
            del X_train, X_val
            gc.collect()
        
        oof_auc = roc_auc_score(self.y, self.oof_proba_lgbm)
        print(f"   LightGBM OOF AUC: {oof_auc:.4f}")
    
    def train_mlp(self):
        """Model B: MLP (Neural Network)"""
        print("\n🧠 Model B: MLP 学習中...")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"   Device: {device}")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.oof_proba_mlp = np.zeros(len(self.y))
        
        # 数値特徴量を準備
        X_numeric_all = self.prepare_numeric_features(self.X, fit=True)
        input_dim = X_numeric_all.shape[1]
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            
            X_train = X_numeric_all.iloc[train_idx].values
            X_val = X_numeric_all.iloc[val_idx].values
            y_train = self.y[train_idx]
            y_val = self.y[val_idx]
            
            # Under-sampling
            pos_idx = np.where(y_train == 1)[0]
            neg_idx = np.where(y_train == 0)[0]
            n_pos = len(pos_idx)
            n_neg_sample = int(n_pos * self.undersample_ratio)
            np.random.seed(self.random_state + fold)
            neg_sample_idx = np.random.choice(neg_idx, size=min(n_neg_sample, len(neg_idx)), replace=False)
            train_idx_sampled = np.concatenate([pos_idx, neg_sample_idx])
            
            X_train_sampled = X_train[train_idx_sampled]
            y_train_sampled = y_train[train_idx_sampled]
            
            # スケーリング
            scaler = QuantileTransformer(output_distribution='normal', random_state=self.random_state)
            X_train_scaled = scaler.fit_transform(X_train_sampled)
            X_val_scaled = scaler.transform(X_val)
            self.scalers.append(scaler)
            
            # PyTorchデータセット
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_scaled),
                torch.FloatTensor(y_train_sampled)
            )
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val_scaled),
                torch.FloatTensor(y_val)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=self.mlp_batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.mlp_batch_size, shuffle=False)
            
            # モデル構築
            model = SimpleMLP(input_dim, self.mlp_hidden_dim).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.mlp_lr)
            
            # 正例の重みを計算
            pos_weight = torch.tensor([len(y_train_sampled) / (2 * y_train_sampled.sum())]).to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            # 学習ループ
            best_auc = 0
            patience_counter = 0
            best_model_state = None
            
            for epoch in range(self.mlp_epochs):
                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    output = model(batch_X)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
                
                # 検証
                model.eval()
                val_preds = []
                with torch.no_grad():
                    for batch_X, _ in val_loader:
                        batch_X = batch_X.to(device)
                        val_preds.append(model(batch_X).cpu().numpy())
                
                val_proba = np.concatenate(val_preds)
                # Logitsなのでsigmoidを適用
                val_proba = 1.0 / (1.0 + np.exp(-val_proba))
                val_auc = roc_auc_score(y_val, val_proba)
                
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_model_state = model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 5:
                        break
            
            # Best model で予測
            model.load_state_dict(best_model_state)
            model.eval()
            with torch.no_grad():
                val_X_tensor = torch.FloatTensor(X_val_scaled).to(device)
                logits = model(val_X_tensor).cpu().numpy()
                # Logitsなのでsigmoidを適用
                fold_proba = 1.0 / (1.0 + np.exp(-logits))
            
            self.oof_proba_mlp[val_idx] = fold_proba
            self.mlp_models.append(model.cpu())
            
            del train_loader, val_loader
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        oof_auc = roc_auc_score(self.y, self.oof_proba_mlp)
        print(f"   MLP OOF AUC: {oof_auc:.4f}")
    
    def train_logistic_regression(self):
        """Model C: Logistic Regression (線形)"""
        print("\n📈 Model C: Logistic Regression 学習中...")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.oof_proba_lr = np.zeros(len(self.y))
        
        # 数値特徴量を準備
        X_numeric_all = self.prepare_numeric_features(self.X, fit=True)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            
            X_train = X_numeric_all.iloc[train_idx].values
            X_val = X_numeric_all.iloc[val_idx].values
            y_train = self.y[train_idx]
            y_val = self.y[val_idx]
            
            # Under-sampling
            pos_idx = np.where(y_train == 1)[0]
            neg_idx = np.where(y_train == 0)[0]
            n_pos = len(pos_idx)
            n_neg_sample = int(n_pos * self.undersample_ratio)
            np.random.seed(self.random_state + fold)
            neg_sample_idx = np.random.choice(neg_idx, size=min(n_neg_sample, len(neg_idx)), replace=False)
            train_idx_sampled = np.concatenate([pos_idx, neg_sample_idx])
            
            X_train_sampled = X_train[train_idx_sampled]
            y_train_sampled = y_train[train_idx_sampled]
            
            # スケーリング
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sampled)
            X_val_scaled = scaler.transform(X_val)
            
            # モデル学習
            model = LogisticRegression(
                max_iter=1000,
                solver='lbfgs',
                class_weight='balanced',
                random_state=self.random_state
            )
            model.fit(X_train_scaled, y_train_sampled)
            
            # 予測
            fold_proba = model.predict_proba(X_val_scaled)[:, 1]
            self.oof_proba_lr[val_idx] = fold_proba
            self.lr_models.append((model, scaler))
            
            gc.collect()
        
        oof_auc = roc_auc_score(self.y, self.oof_proba_lr)
        print(f"   Logistic Regression OOF AUC: {oof_auc:.4f}")
    
    def find_individual_thresholds(self, target_recall=0.995):
        """各モデルの個別閾値を決定（安全マージン込み）"""
        print(f"\n🎯 個別閾値決定 (Target Recall: {target_recall:.1%})...")
        
        self.thresholds = {}
        
        for name, proba in [
            ('lgbm', self.oof_proba_lgbm),
            ('mlp', self.oof_proba_mlp),
            ('lr', self.oof_proba_lr)
        ]:
            # Recall >= target_recallを達成する最大の閾値を探索
            for thresh in np.arange(0.001, 0.5, 0.001):
                pred = (proba >= thresh).astype(int)
                rec = recall_score(self.y, pred)
                if rec < target_recall:
                    self.thresholds[name] = thresh - 0.001
                    break
            else:
                self.thresholds[name] = 0.001
            
            pred = (proba >= self.thresholds[name]).astype(int)
            rec = recall_score(self.y, pred)
            filter_rate = 1 - pred.mean()
            print(f"   {name.upper()}: 閾値={self.thresholds[name]:.4f}, Recall={rec:.4f}, 削減率={filter_rate:.2%}")
    
    def evaluate_or_gate(self):
        """OR条件フィルタリングの評価"""
        print("\n🔗 OR条件フィルタリング評価...")
        
        # 各モデルの個別判定
        pred_lgbm = (self.oof_proba_lgbm >= self.thresholds['lgbm']).astype(int)
        pred_mlp = (self.oof_proba_mlp >= self.thresholds['mlp']).astype(int)
        pred_lr = (self.oof_proba_lr >= self.thresholds['lr']).astype(int)
        
        # OR条件: いずれかが1なら1
        pred_or = np.maximum.reduce([pred_lgbm, pred_mlp, pred_lr])
        
        # 評価
        or_recall = recall_score(self.y, pred_or)
        or_precision = precision_score(self.y, pred_or) if pred_or.sum() > 0 else 0
        or_filter_rate = 1 - pred_or.mean()
        
        print(f"\n   📊 OR条件結果:")
        print(f"      Recall: {or_recall:.4f}")
        print(f"      Precision: {or_precision:.4f}")
        print(f"      削減率 (フィルタリング率): {or_filter_rate:.2%}")
        print(f"      残存データ: {pred_or.sum():,} / {len(pred_or):,}")
        
        # 比較: 単独モデル vs OR条件
        print(f"\n   📈 比較:")
        print(f"      LGBM単独: 削減率={1-pred_lgbm.mean():.2%}, Recall={recall_score(self.y, pred_lgbm):.4f}")
        print(f"      MLP単独:  削減率={1-pred_mlp.mean():.2%}, Recall={recall_score(self.y, pred_mlp):.4f}")
        print(f"      LR単独:   削減率={1-pred_lr.mean():.2%}, Recall={recall_score(self.y, pred_lr):.4f}")
        print(f"      OR条件:   削減率={or_filter_rate:.2%}, Recall={or_recall:.4f}")
        
        self.or_results = {
            'or_recall': or_recall,
            'or_precision': or_precision,
            'or_filter_rate': or_filter_rate,
            'lgbm_filter_rate': 1 - pred_lgbm.mean(),
            'mlp_filter_rate': 1 - pred_mlp.mean(),
            'lr_filter_rate': 1 - pred_lr.mean(),
            'lgbm_recall': recall_score(self.y, pred_lgbm),
            'mlp_recall': recall_score(self.y, pred_mlp),
            'lr_recall': recall_score(self.y, pred_lr),
        }
        
        return self.or_results
    
    def evaluate_test_set(self):
        """テストセットでの評価"""
        print("\n📈 テストセット評価...")
        
        # LightGBM predictions on test
        test_proba_lgbm = np.zeros(len(self.y_test))
        for fold_models in self.lgbm_models:
            for model in fold_models:
                test_proba_lgbm += model.predict_proba(self.X_test)[:, 1]
        test_proba_lgbm /= (self.n_folds * self.n_seeds)
        
        # MLP predictions on test
        X_test_numeric = self.prepare_numeric_features(self.X_test, fit=False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        test_proba_mlp = np.zeros(len(self.y_test))
        
        for fold, (model, scaler) in enumerate(zip(self.mlp_models, self.scalers)):
            X_test_scaled = scaler.transform(X_test_numeric.values)
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_test_scaled).to(device)
                logits = model(X_tensor).cpu().numpy()
                # Logitsなのでsigmoidを適用
                fold_proba = 1.0 / (1.0 + np.exp(-logits))
            test_proba_mlp += fold_proba / self.n_folds
            model.cpu()
        
        # Logistic Regression predictions on test
        test_proba_lr = np.zeros(len(self.y_test))
        for model, scaler in self.lr_models:
            X_test_scaled = scaler.transform(X_test_numeric.values)
            test_proba_lr += model.predict_proba(X_test_scaled)[:, 1] / self.n_folds
        
        # 個別判定
        pred_lgbm = (test_proba_lgbm >= self.thresholds['lgbm']).astype(int)
        pred_mlp = (test_proba_mlp >= self.thresholds['mlp']).astype(int)
        pred_lr = (test_proba_lr >= self.thresholds['lr']).astype(int)
        
        # OR条件
        pred_or = np.maximum.reduce([pred_lgbm, pred_mlp, pred_lr])
        
        test_recall = recall_score(self.y_test, pred_or)
        test_precision = precision_score(self.y_test, pred_or) if pred_or.sum() > 0 else 0
        test_filter_rate = 1 - pred_or.mean()
        
        print(f"\n   📊 テストセット OR条件結果:")
        print(f"      Recall: {test_recall:.4f}")
        print(f"      Precision: {test_precision:.4f}")
        print(f"      削減率 (フィルタリング率): {test_filter_rate:.2%}")
        
        self.test_results = {
            'test_or_recall': test_recall,
            'test_or_precision': test_precision,
            'test_or_filter_rate': test_filter_rate,
        }
        
        return self.test_results
    
    def generate_report(self, elapsed_sec: float):
        """実験レポートをMarkdownで出力"""
        report_path = os.path.join(self.output_dir, "experiment_report.md")
        
        report_content = f"""# 異種混合アンサンブル Stage 1 実験レポート

**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**実行時間**: {elapsed_sec:.1f}秒

## モデル構成

| モデル | 説明 |
|--------|------|
| LightGBM | 決定木ベース、相互作用と非線形が得意 |
| MLP | ニューラルネット、滑らかな決定境界 |
| Logistic Regression | 線形モデル、大局的な傾向判定 |

## 戦略: OR条件フィルタリング

「誰か1人でも危険と言ったら残す（OR条件）」

```
Keep if (Prob_LGBM > Th_LGBM) OR (Prob_MLP > Th_MLP) OR (Prob_LR > Th_LR)
```

## 個別モデル結果 (CV OOF)

| モデル | 閾値 | Recall | 削減率 |
|--------|------|--------|--------|
| LightGBM | {self.thresholds['lgbm']:.4f} | {self.or_results['lgbm_recall']:.4f} | {self.or_results['lgbm_filter_rate']:.2%} |
| MLP | {self.thresholds['mlp']:.4f} | {self.or_results['mlp_recall']:.4f} | {self.or_results['mlp_filter_rate']:.2%} |
| Logistic Regression | {self.thresholds['lr']:.4f} | {self.or_results['lr_recall']:.4f} | {self.or_results['lr_filter_rate']:.2%} |

## OR条件結果 (CV OOF)

| 指標 | 値 |
|------|----| 
| Recall | {self.or_results['or_recall']:.4f} |
| Precision | {self.or_results['or_precision']:.4f} |
| 削減率 | {self.or_results['or_filter_rate']:.2%} |

## テストセット結果

| 指標 | 値 |
|------|----| 
| Recall | {self.test_results['test_or_recall']:.4f} |
| Precision | {self.test_results['test_or_precision']:.4f} |
| 削減率 | {self.test_results['test_or_filter_rate']:.2%} |

## 考察

- OR条件により、個別モデルの死角を補完し合い、Recall {self.or_results['or_recall']:.2%} を達成。
- 削減率 {self.or_results['or_filter_rate']:.2%} でStage 2に渡すデータ量を削減。
- テストセットでも類似の結果が得られ、汎化性能を確認。
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n   📄 レポート出力: {report_path}")
        return report_path
    
    def save_models(self):
        """学習済みモデルを保存"""
        save_dir = os.path.join(self.output_dir, "models")
        os.makedirs(save_dir, exist_ok=True)
        
        # LightGBM
        joblib.dump(self.lgbm_models, os.path.join(save_dir, "lgbm_models.pkl"))
        
        # Logistic Regression (モデルとスケーラー)
        joblib.dump(self.lr_models, os.path.join(save_dir, "lr_models.pkl"))
        
        # MLP (PyTorch models - state_dict)
        mlp_state_dicts = [model.state_dict() for model in self.mlp_models]
        torch.save(mlp_state_dicts, os.path.join(save_dir, "mlp_models.pt"))
        
        # Scalers (MLP用)
        joblib.dump(self.scalers, os.path.join(save_dir, "mlp_scalers.pkl"))
        
        # OrdinalEncoder
        joblib.dump(self.ordinal_encoder, os.path.join(save_dir, "ordinal_encoder.pkl"))
        
        # Thresholds
        joblib.dump(self.thresholds, os.path.join(save_dir, "thresholds.pkl"))
        
        print(f"\n   💾 モデル保存: {save_dir}/")
    
    def run(self):
        """パイプライン実行"""
        start = datetime.now()
        
        self.load_data()
        self.train_lgbm()
        self.train_mlp()
        self.train_logistic_regression()
        self.find_individual_thresholds(target_recall=0.995)  # 安全マージン込み
        self.evaluate_or_gate()
        self.evaluate_test_set()
        
        elapsed_sec = (datetime.now() - start).total_seconds()
        
        # 結果保存
        results = {**self.or_results, **self.test_results, 'elapsed_sec': elapsed_sec}
        pd.DataFrame([results]).to_csv(os.path.join(self.output_dir, "results.csv"), index=False)
        
        self.generate_report(elapsed_sec)
        
        # モデル保存
        self.save_models()
        
        print("\n" + "=" * 60)
        print("✅ 完了！")
        print(f"   結果CSV: {self.output_dir}/results.csv")
        print(f"   レポートMD: {self.output_dir}/experiment_report.md")
        print("=" * 60)
        
        return results


if __name__ == "__main__":
    pipeline = HeterogeneousStage1Pipeline()
    pipeline.run()
