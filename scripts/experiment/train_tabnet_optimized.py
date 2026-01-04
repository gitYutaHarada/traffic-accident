"""
TabNet 最適化実験: 事前学習 + Optunaチューニング + ハイブリッドアンサンブル
(High-Spec PC Optimized + Checkpoint/Resume Support)

既存の LGBM/CatBoost/MLP 予測値を再利用し、TabNet のみを最適化してアンサンブルに統合。
Intel Core Ultra 9 + NVIDIA RTX GPU 向けにチューニング済み。

チェックポイント機能:
- 事前学習モデル: tabnet_pretrained.zip (存在すればスキップ)
- Optuna Study: optuna_study.db (SQLite永続化、途中から再開可能)
- 最終学習: fold_X_oof.npy (Fold毎に保存、途中から再開可能)

使用法:
    python scripts/experiment/train_tabnet_optimized.py
"""
import os
import gc
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.optimize import minimize

# --- ハイスペックPC向け設定 ---
try:
    N_JOBS = os.cpu_count()
except:
    N_JOBS = 20

os.environ["OMP_NUM_THREADS"] = str(N_JOBS)
os.environ["MKL_NUM_THREADS"] = str(N_JOBS)

warnings.filterwarnings('ignore')

# Optuna
try:
    import optuna
    from optuna.storages import RDBStorage
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna not available. Install with: pip install optuna")

# TabNet & Torch
try:
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    from pytorch_tabnet.pretraining import TabNetPretrainer
    TABNET_AVAILABLE = True
    
    if torch.cuda.is_available():
        DEVICE = 'cuda'
        print(f"🚀 GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = 'cpu'
        print("⚠️ GPU not detected. Using CPU.")
except ImportError:
    TABNET_AVAILABLE = False
    DEVICE = 'cpu'
    print("⚠️ TabNet not available. Install with: pip install pytorch-tabnet")


def set_seed(seed: int):
    """再現性のためのシード設定"""
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


class TabNetOptimizer:
    """TabNet 最適化パイプライン (GPU Optimized + Checkpoint Support)"""
    
    FIXED_TABNET_PARAMS = {
        'n_d': 64,
        'n_a': 64,
        'n_steps': 5,
    }

    def __init__(
        self,
        data_path: str = "data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv",
        oof_predictions_path: str = "data/processed/stage1_oof_predictions.csv",
        existing_oof_path: str = "results/stage2_4model_ensemble/oof_predictions.csv",
        existing_test_path: str = "results/stage2_4model_ensemble/test_predictions.csv",
        target_col: str = "死者数",
        stage1_recall_target: float = 0.98,
        n_folds: int = 5,
        random_state: int = 42,
        test_size: float = 0.2,
        output_dir: str = "results/tabnet_optimized"
    ):
        self.data_path = data_path
        self.oof_predictions_path = oof_predictions_path
        self.existing_oof_path = existing_oof_path
        self.existing_test_path = existing_test_path
        self.target_col = target_col
        self.stage1_recall_target = stage1_recall_target
        self.n_folds = n_folds
        self.random_state = random_state
        self.test_size = test_size
        self.output_dir = output_dir
        
        # チェックポイント用ディレクトリ
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        set_seed(random_state)
        
        self.oof_predictions = {}
        self.test_predictions = {}
        self.model_aucs = {}
    
    # ========== チェックポイント関連 ==========
    
    def _pretrain_checkpoint_path(self) -> str:
        return os.path.join(self.output_dir, "tabnet_pretrained.zip")
    
    def _optuna_db_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "optuna_study.db")
    
    def _best_params_path(self) -> str:
        return os.path.join(self.output_dir, "best_params.json")
    
    def _fold_oof_path(self, fold: int) -> str:
        return os.path.join(self.checkpoint_dir, f"fold_{fold}_oof.npy")
    
    def _fold_test_path(self, fold: int) -> str:
        return os.path.join(self.checkpoint_dir, f"fold_{fold}_test.npy")
    
    def _final_oof_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "tabnet_final_oof.npy")
    
    def _final_test_path(self) -> str:
        return os.path.join(self.checkpoint_dir, "tabnet_final_test.npy")
    
    # ========== データ読み込み ==========
        
    def load_and_filter_data(self):
        """データ読み込みとStage 1フィルタリング"""
        print(f"📂 データ読み込み... (Threads: {N_JOBS})")
        
        df = pd.read_csv(self.data_path)
        print(f"   元データ: {len(df):,} 行")
        
        df_oof = pd.read_csv(self.oof_predictions_path)
        
        if self.target_col in df.columns:
            df['fatal'] = (df[self.target_col] > 0).astype(int)
        
        exclude_cols = [
            self.target_col, 'fatal', '負傷者数', '重傷者数', '軽傷者数',
            '当事者A_死傷状況', '当事者B_死傷状況', '本票番号', '発生日時'
        ]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        self.categorical_cols = []
        self.numerical_cols = []
        for col in feature_cols:
            if df[col].dtype == 'object' or df[col].nunique() < 50:
                self.categorical_cols.append(col)
            else:
                self.numerical_cols.append(col)
        
        X_all = df[feature_cols].copy()
        y_all = df['fatal'].values
        
        X_train_full, X_test, y_train_full, y_test, _, _ = train_test_split(
            X_all, y_all, np.arange(len(df)),
            test_size=self.test_size,
            random_state=self.random_state, 
            stratify=y_all
        )
        
        X_train_full = X_train_full.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        self.y_train_full = y_train_full
        self.y_test = y_test
        
        oof_prob = 0.85 * df_oof['prob_catboost'].values + 0.15 * df_oof['prob_lgbm'].values
        
        precision, recall, thresholds = precision_recall_curve(self.y_train_full, oof_prob)
        valid_idx = np.where(recall[:-1] >= self.stage1_recall_target)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[-1]
            self.stage1_threshold = thresholds[best_idx]
        else:
            self.stage1_threshold = 0.0
        
        train_mask = oof_prob >= self.stage1_threshold
        
        self.X_train_full_unfiltered = X_train_full
        self.X_train = X_train_full[train_mask].reset_index(drop=True)
        self.y_train = self.y_train_full[train_mask]
        
        try:
            df_test_pred = pd.read_csv("data/processed/stage1_test_predictions.csv")
            test_prob = 0.85 * df_test_pred['prob_catboost'].values + 0.15 * df_test_pred['prob_lgbm'].values
            test_mask = test_prob >= self.stage1_threshold
            
            self.X_test_filtered = X_test[test_mask].reset_index(drop=True)
            self.y_test_filtered = self.y_test[test_mask]
        except FileNotFoundError:
            print("⚠️ stage1_test_predictions.csv not found. Using unfiltered test.")
            self.X_test_filtered = X_test
            self.y_test_filtered = y_test
        
        print(f"   Train (Filtered): {len(self.y_train):,} (Fatal: {self.y_train.sum():,})")
        print(f"   Test (Filtered):  {len(self.y_test_filtered):,} (Fatal: {self.y_test_filtered.sum():,})")
        
        self.feature_cols = feature_cols
        self._prepare_categorical_encoders()
        
    def _prepare_categorical_encoders(self):
        """カテゴリ変数のエンコーダー準備"""
        self.cat_encoders = {}
        self.cat_dims = []
        self.cat_idxs = []
        
        all_data = pd.concat([self.X_train, self.X_test_filtered], axis=0)
        
        for i, col in enumerate(self.feature_cols):
            if col in self.categorical_cols:
                le = LabelEncoder()
                le.fit(all_data[col].astype(str).fillna('__MISSING__'))
                self.cat_encoders[col] = le
                self.cat_dims.append(len(le.classes_))
                self.cat_idxs.append(i)
        
        print(f"   カテゴリ変数: {len(self.categorical_cols)} 列")
        print(f"   数値変数: {len(self.numerical_cols)} 列")
    
    def _encode_data(self, X: pd.DataFrame) -> np.ndarray:
        """データをエンコード"""
        X_enc = X.copy()
        for col, le in self.cat_encoders.items():
            if col in X_enc.columns:
                try:
                    X_enc[col] = le.transform(X_enc[col].astype(str).fillna('__MISSING__'))
                except:
                    known_classes = set(le.classes_)
                    X_enc[col] = X_enc[col].astype(str).fillna('__MISSING__').apply(
                        lambda x: le.transform([x])[0] if x in known_classes else 0
                    )
        
        num_cols = [c for c in self.numerical_cols if c in X_enc.columns]
        if not hasattr(self, 'scaler'):
            self.scaler = StandardScaler()
            X_enc[num_cols] = self.scaler.fit_transform(X_enc[num_cols].fillna(0))
        else:
            X_enc[num_cols] = self.scaler.transform(X_enc[num_cols].fillna(0))
        
        return X_enc.values.astype(np.float32)
    
    def load_existing_predictions(self):
        """既存予測値をロード"""
        print("\n📂 既存予測値ロード...")
        try:
            oof_df = pd.read_csv(self.existing_oof_path)
            test_df = pd.read_csv(self.existing_test_path)
            
            for model in ['lgbm', 'catboost', 'mlp']:
                if model in oof_df.columns:
                    if len(oof_df) != len(self.y_train):
                        print(f"⚠️ {model}: 行数不一致 ({len(oof_df)} vs {len(self.y_train)}) - スキップ")
                        continue
                        
                    self.oof_predictions[model] = oof_df[model].values
                    self.test_predictions[model] = test_df[model].values
                    
                    auc = roc_auc_score(self.y_train, self.oof_predictions[model])
                    self.model_aucs[model] = auc
                    print(f"   ✅ {model}: OOF AUC = {auc:.4f}")
        except FileNotFoundError:
            print("⚠️ 既存予測ファイルが見つかりません。TabNet単体で実行します。")
    
    def pretraining(self, max_epochs: int = 100):
        """TabNet 事前学習 (チェックポイント対応)"""
        if not TABNET_AVAILABLE:
            return None
        
        pretrain_path = self._pretrain_checkpoint_path()
        
        # === チェックポイント確認 ===
        if os.path.exists(pretrain_path):
            print(f"\n✅ 事前学習済みモデルを検出: {pretrain_path}")
            print("   → 事前学習をスキップします。")
            return pretrain_path
        
        print("\n🔧 TabNet 事前学習 (Pretraining) [GPU Mode]...")
        
        X_pretrain = self._encode_data(self.X_train_full_unfiltered)
        
        pretrainer = TabNetPretrainer(
            **self.FIXED_TABNET_PARAMS,
            cat_idxs=self.cat_idxs,
            cat_dims=self.cat_dims,
            cat_emb_dim=2,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            mask_type='entmax',
            seed=self.random_state,
            device_name=DEVICE,
            verbose=1
        )
        
        pretrainer.fit(
            X_train=X_pretrain,
            eval_set=[X_pretrain],
            max_epochs=max_epochs,
            patience=20,
            batch_size=8192,
            virtual_batch_size=1024,
            pretraining_ratio=0.8,
            num_workers=0
        )
        
        pretrainer.save_model(pretrain_path)
        print(f"   💾 事前学習モデル保存: {pretrain_path}")
        
        return pretrain_path
    
    def optuna_tuning(self, n_trials: int = 30, timeout: int = 7200):
        """Optuna チューニング (SQLite永続化で再開可能)"""
        if not OPTUNA_AVAILABLE or not TABNET_AVAILABLE:
            return {}
        
        best_params_path = self._best_params_path()
        
        # === チェックポイント確認 ===
        if os.path.exists(best_params_path):
            print(f"\n✅ 最適パラメータを検出: {best_params_path}")
            print("   → Optunaチューニングをスキップします。")
            with open(best_params_path, 'r') as f:
                self.best_params = json.load(f)
            print(f"   パラメータ: {self.best_params}")
            return self.best_params
        
        print(f"\n🔍 Optuna チューニング (GPUモード, Max {n_trials} trials)...")
        
        # SQLite Storage for persistence (途中から再開可能)
        db_path = self._optuna_db_path()
        storage = f"sqlite:///{db_path}"
        study_name = "tabnet_optimization"
        
        print(f"   📁 Study DB: {db_path}")
        
        X_train_enc = self._encode_data(self.X_train)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train_enc, self.y_train,
            test_size=0.2, random_state=self.random_state, stratify=self.y_train
        )
        
        pretrain_path = self._pretrain_checkpoint_path()
        
        def objective(trial):
            params = self.FIXED_TABNET_PARAMS.copy()
            
            params.update({
                'gamma': trial.suggest_float('gamma', 1.0, 2.0),
                'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-6, 1e-3, log=True),
                'momentum': trial.suggest_float('momentum', 0.01, 0.4),
                'mask_type': trial.suggest_categorical('mask_type', ['sparsemax', 'entmax']),
                'optimizer_params': dict(lr=trial.suggest_float('lr', 1e-3, 0.1, log=True)),
            })
            
            batch_size = trial.suggest_categorical('batch_size', [4096, 8192, 16384])
            virtual_batch_size = trial.suggest_categorical('virtual_batch_size', [256, 512, 1024])
            
            model = TabNetClassifier(
                **params,
                cat_idxs=self.cat_idxs,
                cat_dims=self.cat_dims,
                cat_emb_dim=2,
                seed=self.random_state,
                device_name=DEVICE,
                verbose=0
            )
            
            if os.path.exists(pretrain_path):
                try:
                    model.load_model(pretrain_path)
                except:
                    pass
            
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric=['auc'],
                max_epochs=50,
                patience=10,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
                num_workers=0,
                drop_last=False
            )
            
            val_pred = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, val_pred)
        
        # load_if_exists=True で既存のStudyから再開
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction='maximize',
            load_if_exists=True
        )
        
        # 既存の試行数を確認
        completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining_trials = max(0, n_trials - completed_trials)
        
        if completed_trials > 0:
            print(f"   📂 既存の試行を検出: {completed_trials} 完了済み")
            print(f"   → 残り {remaining_trials} 試行を実行します。")
        
        if remaining_trials > 0:
            study.optimize(objective, n_trials=remaining_trials, timeout=timeout, show_progress_bar=True)
        
        print(f"\n   ベストスコア: {study.best_value:.4f}")
        print(f"   ベストパラメータ: {study.best_params}")
        
        with open(best_params_path, 'w') as f:
            json.dump(study.best_params, f, indent=2)
        
        self.best_params = study.best_params
        return study.best_params
    
    def train_final_tabnet(self, params: dict = None):
        """最終学習 (5-Fold CV, Fold毎にチェックポイント)"""
        if not TABNET_AVAILABLE:
            return
        
        final_oof_path = self._final_oof_path()
        final_test_path = self._final_test_path()
        
        # === 完了チェック ===
        if os.path.exists(final_oof_path) and os.path.exists(final_test_path):
            print(f"\n✅ 最終学習済み予測を検出")
            self.oof_predictions['tabnet_optimized'] = np.load(final_oof_path)
            self.test_predictions['tabnet_optimized'] = np.load(final_test_path)
            auc = roc_auc_score(self.y_train, self.oof_predictions['tabnet_optimized'])
            self.model_aucs['tabnet_optimized'] = auc
            print(f"   → TabNet OOF AUC: {auc:.4f}")
            return
        
        print("\n📊 TabNet 最終学習 (5-Fold CV, GPU)...")
        
        if params is None:
            params = self.best_params if hasattr(self, 'best_params') else {}
            
        final_params = self.FIXED_TABNET_PARAMS.copy()
        ignore_keys = ['batch_size', 'virtual_batch_size']
        for k, v in params.items():
            if k not in ignore_keys:
                final_params[k] = v
        
        if 'lr' in params:
             final_params['optimizer_params'] = dict(lr=params.get('lr', 2e-2))
        
        batch_size = params.get('batch_size', 8192)
        virtual_batch_size = params.get('virtual_batch_size', 512)
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_proba = np.zeros(len(self.y_train))
        test_proba = np.zeros(len(self.y_test_filtered))
        
        X_train_enc = self._encode_data(self.X_train)
        X_test_enc = self._encode_data(self.X_test_filtered)
        
        pretrain_path = self._pretrain_checkpoint_path()
        
        # どのFoldまで完了しているか確認
        start_fold = 0
        for fold in range(self.n_folds):
            fold_oof_path = self._fold_oof_path(fold)
            fold_test_path = self._fold_test_path(fold)
            
            if os.path.exists(fold_oof_path) and os.path.exists(fold_test_path):
                fold_oof = np.load(fold_oof_path)
                fold_test = np.load(fold_test_path)
                
                # OOF は該当インデックスに代入が必要なので、別途保存形式を工夫
                # ここでは累積テスト予測を加算
                test_proba += fold_test
                
                # OOFのインデックスを復元するため、全体を保存する方式に変更
                start_fold = fold + 1
                print(f"   ✅ Fold {fold+1} をチェックポイントから復元")
            else:
                break
        
        if start_fold > 0:
            # 途中から再開する場合、OOFを完全に復元するには工夫が必要
            # 簡易版: Fold毎にOOF全体を保存し、最後のFoldで統合する方式
            cumulative_oof_path = os.path.join(self.checkpoint_dir, "cumulative_oof.npy")
            if os.path.exists(cumulative_oof_path):
                oof_proba = np.load(cumulative_oof_path)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train)):
            if fold < start_fold:
                continue  # 完了済みFoldはスキップ
            
            print(f"   Fold {fold+1}/{self.n_folds}...")
            
            X_tr, X_val = X_train_enc[train_idx], X_train_enc[val_idx]
            y_tr, y_val = self.y_train[train_idx], self.y_train[val_idx]
            
            model = TabNetClassifier(
                **final_params,
                cat_idxs=self.cat_idxs,
                cat_dims=self.cat_dims,
                cat_emb_dim=2,
                seed=self.random_state + fold,
                device_name=DEVICE,
                verbose=0
            )
            
            if os.path.exists(pretrain_path):
                try:
                    model.load_model(pretrain_path)
                except:
                    pass
            
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric=['auc'],
                max_epochs=150,
                patience=20,
                batch_size=batch_size,
                virtual_batch_size=virtual_batch_size,
                num_workers=0,
                drop_last=False
            )
            
            fold_oof = model.predict_proba(X_val)[:, 1]
            fold_test = model.predict_proba(X_test_enc)[:, 1] / self.n_folds
            
            oof_proba[val_idx] = fold_oof
            test_proba += fold_test
            
            # === Fold毎にチェックポイント保存 ===
            np.save(self._fold_oof_path(fold), fold_oof)
            np.save(self._fold_test_path(fold), fold_test)
            np.save(os.path.join(self.checkpoint_dir, "cumulative_oof.npy"), oof_proba)
            print(f"      💾 Fold {fold+1} チェックポイント保存完了")
            
            del model
            gc.collect()
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
        
        auc = roc_auc_score(self.y_train, oof_proba)
        print(f"   TabNet (Optimized) OOF AUC: {auc:.4f}")
        
        # 最終結果を保存
        np.save(final_oof_path, oof_proba)
        np.save(final_test_path, test_proba)
        
        self.oof_predictions['tabnet_optimized'] = oof_proba
        self.test_predictions['tabnet_optimized'] = test_proba
        self.model_aucs['tabnet_optimized'] = auc
    
    def optimize_ensemble_weights(self):
        """アンサンブル最適化"""
        print("\n⚖️ アンサンブル重み最適化...")
        available_models = list(self.oof_predictions.keys())
        if len(available_models) < 2:
            print("   モデル数が不足しています (2つ以上必要)")
            return
            
        oof_matrix = np.column_stack([self.oof_predictions[m] for m in available_models])
        
        def neg_auc(weights):
            weights = np.array(weights)
            weights /= weights.sum()
            ensemble_pred = oof_matrix @ weights
            return -roc_auc_score(self.y_train, ensemble_pred)
        
        init_weights = np.ones(len(available_models)) / len(available_models)
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(available_models))]
        
        result = minimize(neg_auc, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        self.ensemble_weights = {m: w for m, w in zip(available_models, result.x)}
        
        print("   最適化された重み:")
        for m, w in self.ensemble_weights.items():
            print(f"     {m}: {w:.4f}")
            
        self.oof_predictions['ensemble'] = oof_matrix @ result.x
        test_matrix = np.column_stack([self.test_predictions[m] for m in available_models])
        self.test_predictions['ensemble'] = test_matrix @ result.x
        
        ensemble_auc = roc_auc_score(self.y_train, self.oof_predictions['ensemble'])
        self.model_aucs['ensemble'] = ensemble_auc
        print(f"   Ensemble OOF AUC: {ensemble_auc:.4f}")

    def evaluate_and_report(self):
        """結果レポート保存"""
        print("\n📈 結果保存...")
        
        results = []
        for model, pred in self.oof_predictions.items():
            test_pred = self.test_predictions.get(model)
            oof_auc = roc_auc_score(self.y_train, pred)
            test_auc = roc_auc_score(self.y_test_filtered, test_pred) if test_pred is not None else None
            results.append({
                'model': model,
                'oof_auc': oof_auc,
                'test_auc': test_auc
            })
            test_auc_str = f"{test_auc:.4f}" if test_auc is not None else "N/A"
            print(f"   {model}: OOF AUC={oof_auc:.4f}, Test AUC={test_auc_str}")
            
        pd.DataFrame(results).to_csv(os.path.join(self.output_dir, "final_scores.csv"), index=False)
        
        # OOF/Test予測も保存
        oof_df = pd.DataFrame(self.oof_predictions)
        oof_df['target'] = self.y_train
        oof_df.to_csv(os.path.join(self.output_dir, "oof_predictions.csv"), index=False)
        
        test_df = pd.DataFrame(self.test_predictions)
        test_df['target'] = self.y_test_filtered
        test_df.to_csv(os.path.join(self.output_dir, "test_predictions.csv"), index=False)
        
        print(f"\n   ✅ 完了: {self.output_dir}")

    def run(self, pretrain_epochs=100, optuna_trials=30, optuna_timeout=7200):
        """パイプライン実行 (チェックポイント対応)"""
        start = datetime.now()
        
        print("=" * 70)
        print("🚀 TabNet 最適化実験 (Checkpoint/Resume Support)")
        print(f"   チェックポイント: {self.checkpoint_dir}")
        print("=" * 70)
        
        self.load_and_filter_data()
        self.load_existing_predictions()
        self.pretraining(max_epochs=pretrain_epochs)
        self.optuna_tuning(n_trials=optuna_trials, timeout=optuna_timeout)
        self.train_final_tabnet()
        self.optimize_ensemble_weights()
        self.evaluate_and_report()
        
        elapsed = (datetime.now() - start).total_seconds()
        print("\n" + "=" * 70)
        print(f"✅ 全工程完了！ 実行時間: {elapsed/60:.1f}分")
        print("=" * 70)


if __name__ == "__main__":
    optimizer = TabNetOptimizer()
    optimizer.run(
        pretrain_epochs=100,
        optuna_trials=30,
        optuna_timeout=7200
    )
