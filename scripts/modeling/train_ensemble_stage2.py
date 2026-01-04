"""
LightGBM + CatBoost + TabNet アンサンブルモデル (v3.1 - 並列処理 + 細粒度チェックポイント版)
================================================================================
Stage 1: LightGBM (99% Recall フィルタリング) - 並列化 + Seed単位チェックポイント
Stage 2: 3モデルアンサンブル (LightGBM + CatBoost + TabNet) - 並列化 + モデル単位チェックポイント

修正点 (v3.1):
- 各Fold内の「モデル単位」で学習済みファイルを保存し、再実行時にスキップ
- 途中停止しても、完了したモデルまではロードして続きから再開可能

実行方法:
    python scripts/modeling/train_ensemble_stage2.py
"""

import pandas as pd
import numpy as np
import os
import gc
import joblib
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')

# TabNet のインポート
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    TABNET_AVAILABLE = True
except ImportError:
    TABNET_AVAILABLE = False
    print("⚠️ TabNetがインストールされていません。pip install pytorch-tabnet でインストールしてください。")


def get_focal_loss_lgb(alpha: float = 0.75, gamma: float = 1.0):
    """LightGBM用 Focal Loss (非並列時用)"""
    from scipy.special import expit
    def focal_loss_lgb(y_true, preds):
        p = expit(preds)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        p_t = y_true * p + (1 - y_true) * (1 - p)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_weight = (1 - p_t) ** gamma
        grad = alpha_t * focal_weight * (p - y_true)
        hess = alpha_t * focal_weight * p * (1 - p)
        hess = np.maximum(hess, 1e-7)
        return grad, hess
    return focal_loss_lgb


# ============================================================================
# 並列処理用の関数（クラス外に定義）
# ============================================================================

def train_stage1_fold(fold, train_idx, val_idx, X, y, lgb_params, undersample_ratio, n_seeds, random_state, categorical_cols, output_dir):
    """Stage 1: 単一Foldの学習（Seed単位チェックポイント付き）"""
    save_dir = os.path.join(output_dir, f"stage1_fold{fold}")
    os.makedirs(save_dir, exist_ok=True)

    X_train_full = X.iloc[train_idx].copy()
    y_train_full = y[train_idx]
    X_val = X.iloc[val_idx].copy()
    y_val = y[val_idx]

    # LightGBM用にobject列をcategory型に変換
    for col in categorical_cols:
        if col in X_train_full.columns:
            X_train_full[col] = X_train_full[col].astype('category')
            X_val[col] = X_val[col].astype('category')

    fold_proba = np.zeros(len(val_idx))
    fold_models = []

    for seed_offset in range(n_seeds):
        seed = random_state + fold * 100 + seed_offset
        ckpt_path = os.path.join(save_dir, f"seed{seed_offset}_model.pkl")
        pred_path = os.path.join(save_dir, f"seed{seed_offset}_pred.npy")

        if os.path.exists(ckpt_path) and os.path.exists(pred_path):
            # チェックポイント読み込み
            print(f"[Fold {fold}] Loading Stage 1 Seed {seed_offset} from checkpoint...")
            model = joblib.load(ckpt_path)
            pred = np.load(pred_path)
        else:
            # 新規学習
            # アンダーサンプリング
            pos_idx = np.where(y_train_full == 1)[0]
            neg_idx = np.where(y_train_full == 0)[0]
            n_neg_sample = int(len(pos_idx) * undersample_ratio)
            np.random.seed(seed)
            sampled_neg_idx = np.random.choice(neg_idx, size=min(n_neg_sample, len(neg_idx)), replace=False)
            sampled_idx = np.concatenate([pos_idx, sampled_neg_idx])
            np.random.shuffle(sampled_idx)
            
            X_train_under = X_train_full.iloc[sampled_idx].copy()
            y_train_under = y_train_full[sampled_idx]

            # LightGBM用にカテゴリ変数を再設定
            for col in categorical_cols:
                if col in X_train_under.columns:
                    X_train_under[col] = X_train_under[col].astype('category')

            model = lgb.LGBMClassifier(**lgb_params, random_state=seed, n_jobs=4)
            model.fit(X_train_under, y_train_under, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            
            pred = model.predict_proba(X_val)[:, 1]
            
            # 保存
            joblib.dump(model, ckpt_path)
            np.save(pred_path, pred)

        fold_proba += pred / n_seeds
        fold_models.append(model)

    return fold, val_idx, fold_proba, fold_models


def train_stage2_fold(fold, train_idx, val_idx, X_s2, y_s2, 
                      focal_alpha, focal_gamma, categorical_cols, numerical_cols,
                      random_state, tabnet_available, output_dir):
    """Stage 2: 単一Foldの学習（モデル単位チェックポイント付き）"""
    save_dir = os.path.join(output_dir, f"stage2_fold{fold}")
    os.makedirs(save_dir, exist_ok=True)

    X_train = X_s2.iloc[train_idx].copy()
    y_train = y_s2[train_idx]
    X_val = X_s2.iloc[val_idx].copy()
    y_val = y_s2[val_idx]
    
    # LightGBM用にobject列をcategory型に変換
    for col in categorical_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')

    results = {
        'fold': fold, 'val_idx': val_idx,
        'oof_lgb': None, 'oof_cat': None, 'oof_tab': None,
        'lgb_model': None, 'cat_model': None, 'tab_model': None,
        'tabnet_preprocessor': None,
    }

    # ============================
    # 1. LightGBM (Binary)
    # ============================
    path_lgb_model = os.path.join(save_dir, "lgb_model.pkl")
    path_lgb_pred = os.path.join(save_dir, "lgb_pred.npy")

    if os.path.exists(path_lgb_model) and os.path.exists(path_lgb_pred):
        print(f"[Fold {fold}] Loading Stage 2 LightGBM from checkpoint...")
        results['lgb_model'] = joblib.load(path_lgb_model)
        results['oof_lgb'] = np.load(path_lgb_pred)
    else:
        # 学習
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        lgb_params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'verbosity': -1, 'num_leaves': 127, 'max_depth': -1, 'min_child_samples': 44,
            'reg_alpha': 2.3897, 'reg_lambda': 2.2842, 'colsample_bytree': 0.8646,
            'subsample': 0.6328, 'learning_rate': 0.0477, 'n_estimators': 1000, 'n_jobs': 4,
            'scale_pos_weight': scale_pos_weight
        }
        lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=random_state + fold)
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        
        pred = lgb_model.predict_proba(X_val)[:, 1]
        
        # 保存
        joblib.dump(lgb_model, path_lgb_model)
        np.save(path_lgb_pred, pred)
        results['lgb_model'] = lgb_model
        results['oof_lgb'] = pred

    # ============================
    # 2. CatBoost
    # ============================
    path_cat_model = os.path.join(save_dir, "cat_model.cbm")  # CatBoost固有形式推奨だがpickleでも可
    path_cat_pred = os.path.join(save_dir, "cat_pred.npy")

    # CatBoostはpickle保存時に少し特殊な挙動をすることがあるので、確実にファイルがあるか確認
    # ここではシンプルにjoblibを使う（CatBoostClassifierはpickle可能）
    path_cat_pkl = os.path.join(save_dir, "cat_model.pkl")

    if os.path.exists(path_cat_pkl) and os.path.exists(path_cat_pred):
        print(f"[Fold {fold}] Loading Stage 2 CatBoost from checkpoint...")
        results['cat_model'] = joblib.load(path_cat_pkl)
        results['oof_cat'] = np.load(path_cat_pred)
    else:
        cat_model = CatBoostClassifier(
            iterations=1000, learning_rate=0.05, depth=8, l2_leaf_reg=3,
            loss_function='Logloss', eval_metric='AUC', random_seed=random_state + fold,
            verbose=False, early_stopping_rounds=50, task_type='CPU', thread_count=4,
            cat_features=[c for c in categorical_cols if c in X_train.columns]
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
        pred = cat_model.predict_proba(X_val)[:, 1]
        
        joblib.dump(cat_model, path_cat_pkl)
        np.save(path_cat_pred, pred)
        results['cat_model'] = cat_model
        results['oof_cat'] = pred

    # ============================
    # 3. TabNet
    # ============================
    if tabnet_available:
        path_tab_model = os.path.join(save_dir, "tab_model.zip") # TabNetはzip保存
        path_tab_pred = os.path.join(save_dir, "tab_pred.npy")
        path_tab_pre = os.path.join(save_dir, "tab_preprocessors.pkl")

        if os.path.exists(path_tab_model) and os.path.exists(path_tab_pred) and os.path.exists(path_tab_pre):
            print(f"[Fold {fold}] Loading Stage 2 TabNet from checkpoint...")
            tab_model = TabNetClassifier()
            tab_model.load_model(path_tab_model)
            results['tab_model'] = tab_model
            results['oof_tab'] = np.load(path_tab_pred)
            results['tabnet_preprocessor'] = joblib.load(path_tab_pre)
        else:
            num_cols = numerical_cols + ['prob_stage1']
            cat_cols = categorical_cols
            
            imputer = SimpleImputer(strategy='mean')
            scaler = StandardScaler()
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            
            X_num_train = imputer.fit_transform(X_train[num_cols].values)
            X_num_train = scaler.fit_transform(X_num_train)
            X_cat_train = encoder.fit_transform(X_train[cat_cols].values) + 1
            X_train_tab = np.hstack([X_num_train, X_cat_train]).astype(np.float32)
            
            X_num_val = scaler.transform(imputer.transform(X_val[num_cols].values))
            X_cat_val = encoder.transform(X_val[cat_cols].values) + 1
            X_val_tab = np.hstack([X_num_val, X_cat_val]).astype(np.float32)
            
            tab_model = TabNetClassifier(
                n_d=32, n_a=32, n_steps=5, gamma=1.5, n_independent=2, n_shared=2,
                lambda_sparse=1e-4, momentum=0.3, clip_value=2.0,
                optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
                scheduler_fn=torch.optim.lr_scheduler.StepLR,
                scheduler_params=dict(step_size=10, gamma=0.9),
                seed=random_state + fold, verbose=0
            )
            tab_model.fit(
                X_train_tab, y_train,
                eval_set=[(X_val_tab, y_val)],
                eval_metric=['auc'],
                max_epochs=50, patience=10, batch_size=4096, virtual_batch_size=128
            )
            pred = tab_model.predict_proba(X_val_tab)[:, 1]
            
            tab_model.save_model(path_tab_model.replace('.zip', '')) # save_modelは拡張子なしパスを期待
            np.save(path_tab_pred, pred)
            joblib.dump((imputer, scaler, encoder), path_tab_pre)
            
            results['tab_model'] = tab_model
            results['oof_tab'] = pred
            results['tabnet_preprocessor'] = (imputer, scaler, encoder)

    return results


class EnsembleStage2Pipeline:
    """3モデルアンサンブルパイプライン (v3.1 - 並列処理 + 細粒度チェックポイント版)"""

    def __init__(
        self,
        data_path: str = "data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv",
        target_col: str = "fatal",
        n_folds: int = 5,
        random_state: int = 42,
        stage1_recall_target: float = 0.99,
        undersample_ratio: float = 2.0,
        n_seeds: int = 3,
        test_size: float = 0.2,
        focal_alpha: float = 0.6321,
        focal_gamma: float = 1.1495,
        output_dir: str = "results/ensemble_stage2",
        n_jobs: int = 5,  # 並列Fold数
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.n_folds = n_folds
        self.random_state = random_state
        self.stage1_recall_target = stage1_recall_target
        self.undersample_ratio = undersample_ratio
        self.n_seeds = n_seeds
        self.test_size = test_size
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.output_dir = output_dir
        self.n_jobs = n_jobs
        
        # チェックポイント用ディレクトリ作成
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.tabnet_preprocessors = []

        print("=" * 80)
        print("LightGBM + CatBoost + TabNet アンサンブルモデル (v3.1)")
        print(f"Stage 1: 1:{int(self.undersample_ratio)} Under-sampling, Recall {self.stage1_recall_target:.0%}")
        print(f"Stage 2: 3モデルアンサンブル (LightGBM / CatBoost / TabNet)")
        print(f"環境: 並列 {self.n_jobs} Folds, チェックポイント保存あり")
        print("=" * 80)

    def load_data(self):
        """データ読み込みとTrain/Test分割"""
        print("\n📂 データ読み込み中...")
        df = pd.read_csv(self.data_path)
        y_all = df[self.target_col].values
        X_all = df.drop(columns=[self.target_col])

        if '発生日時' in X_all.columns:
            X_all = X_all.drop(columns=['発生日時'])

        known_categoricals = [
            '都道府県コード', '市区町村コード', '警察署等コード',
            '昼夜', '天候', '地形', '路面状態', '道路形状', '信号機',
            '衝突地点', 'ゾーン規制', '中央分離帯施設等', '歩車道区分',
            '事故類型', '曜日(発生年月日)', '祝日(発生年月日)',
            'road_type', 'area_id', '地点コード'
        ]

        self.categorical_cols = []
        self.numerical_cols = []
        for col in X_all.columns:
            if col in known_categoricals or X_all[col].dtype == 'object':
                self.categorical_cols.append(col)
                X_all[col] = X_all[col].astype(str)
            else:
                self.numerical_cols.append(col)
                X_all[col] = X_all[col].astype(np.float32)

        self.feature_names = list(X_all.columns)

        print(f"\n📊 データ分割 (Train: {1-self.test_size:.0%} / Test: {self.test_size:.0%})")
        self.X, self.X_test, self.y, self.y_test = train_test_split(
            X_all, y_all, test_size=self.test_size, random_state=self.random_state, stratify=y_all
        )
        self.X = self.X.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)

        print(f"   Train: 正例 {self.y.sum():,} / {len(self.y):,}")
        print(f"   Test:  正例 {self.y_test.sum():,} / {len(self.y_test):,}")
        gc.collect()

    def train_stage1(self):
        """Stage 1: LightGBM OOF学習（並列化+Checkpoint）"""
        print("\n🌿 Stage 1: LightGBM + Under-sampling (並列5Fold)")

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.oof_proba_stage1 = np.zeros(len(self.y))

        lgb_params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'verbosity': -1, 'num_leaves': 31, 'max_depth': 8,
            'reg_alpha': 0.1, 'reg_lambda': 0.1, 'n_estimators': 1000,
            'learning_rate': 0.05
        }

        folds_data = list(skf.split(self.X, self.y))

        print(f"   🚀 {self.n_jobs}並列で{self.n_folds} Folds実行中...")
        start_time = datetime.now()

        # 並列実行
        results = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(train_stage1_fold)(
                fold, train_idx, val_idx, self.X, self.y,
                lgb_params, self.undersample_ratio, self.n_seeds, self.random_state,
                self.categorical_cols, self.ckpt_dir
            ) for fold, (train_idx, val_idx) in enumerate(folds_data)
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"   ✅ Stage 1完了 ({elapsed:.1f}秒)")

        # 結果の集約
        self.stage1_models = [None] * self.n_folds
        for fold, val_idx, fold_proba, fold_models in results:
            self.oof_proba_stage1[val_idx] = fold_proba
            self.stage1_models[fold] = fold_models

        oof_auc = roc_auc_score(self.y, self.oof_proba_stage1)
        print(f"   Stage 1 OOF AUC: {oof_auc:.4f}")

    def find_recall_threshold(self):
        """Recall目標閾値探索"""
        for thresh in np.arange(0.50, 0.001, -0.005):
            y_pred = (self.oof_proba_stage1 >= thresh).astype(int)
            recall = recall_score(self.y, y_pred)
            if recall >= self.stage1_recall_target:
                self.threshold_stage1 = thresh
                break
        else:
            self.threshold_stage1 = 0.001

        y_pred_final = (self.oof_proba_stage1 >= self.threshold_stage1).astype(int)
        self.stage1_recall = recall_score(self.y, y_pred_final)
        n_candidates = y_pred_final.sum()
        self.filter_rate = 1 - (n_candidates / len(self.y))

        print(f"   閾値: {self.threshold_stage1:.4f}, Recall: {self.stage1_recall:.4f}")
        print(f"   フィルタリング率: {self.filter_rate*100:.2f}% 除外, 候補: {n_candidates:,}")

        self.stage2_mask = self.oof_proba_stage1 >= self.threshold_stage1

    def _prepare_data_for_models(self, X_subset, prob_stage1_subset):
        """Stage 2用データ準備"""
        X_out = X_subset.copy()
        X_out['prob_stage1'] = prob_stage1_subset
        return X_out

    def train_stage2_ensemble(self):
        """Stage 2: 3モデルアンサンブル学習（並列化+Checkpoint）"""
        print("\n🌿 Stage 2: 3モデルアンサンブル (並列5Fold)")

        X_s2 = self._prepare_data_for_models(
            self.X[self.stage2_mask].copy(),
            self.oof_proba_stage1[self.stage2_mask]
        ).reset_index(drop=True)
        y_s2 = self.y[self.stage2_mask]

        n_pos, n_neg = y_s2.sum(), len(y_s2) - y_s2.sum()
        print(f"   Stage 2 データ: {len(y_s2):,} (Pos: {n_pos:,}, Neg: {n_neg:,})")

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        folds_data = list(skf.split(X_s2, y_s2))

        print(f"   🚀 {self.n_jobs}並列で{self.n_folds} Folds実行中...")
        start_time = datetime.now()

        # 並列実行
        results = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(train_stage2_fold)(
                fold, train_idx, val_idx, X_s2, y_s2,
                self.focal_alpha, self.focal_gamma,
                self.categorical_cols, self.numerical_cols,
                self.random_state, TABNET_AVAILABLE, self.ckpt_dir
            ) for fold, (train_idx, val_idx) in enumerate(folds_data)
        )

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"   ✅ Stage 2完了 ({elapsed:.1f}秒)")

        # 結果の集約
        self.oof_lgb = np.zeros(len(y_s2))
        self.oof_cat = np.zeros(len(y_s2))
        self.oof_tab = np.zeros(len(y_s2)) if TABNET_AVAILABLE else None

        self.lgb_models = [None] * self.n_folds
        self.cat_models = [None] * self.n_folds
        self.tab_models = [None] * self.n_folds if TABNET_AVAILABLE else None
        self.tabnet_preprocessors = [None] * self.n_folds if TABNET_AVAILABLE else None

        for res in results:
            fold = res['fold']
            val_idx = res['val_idx']
            self.oof_lgb[val_idx] = res['oof_lgb']
            self.oof_cat[val_idx] = res['oof_cat']
            self.lgb_models[fold] = res['lgb_model']
            self.cat_models[fold] = res['cat_model']
            if TABNET_AVAILABLE:
                self.oof_tab[val_idx] = res['oof_tab']
                self.tab_models[fold] = res['tab_model']
                self.tabnet_preprocessors[fold] = res['tabnet_preprocessor']

        # 個別モデルのOOF AUC
        print(f"\n   📊 個別モデル OOF AUC:")
        print(f"      LightGBM: {roc_auc_score(y_s2, self.oof_lgb):.4f}")
        print(f"      CatBoost: {roc_auc_score(y_s2, self.oof_cat):.4f}")
        if TABNET_AVAILABLE:
            print(f"      TabNet:   {roc_auc_score(y_s2, self.oof_tab):.4f}")

        self._optimize_ensemble_weights(y_s2)
        self.y_s2 = y_s2
        self.X_s2 = X_s2

    def _optimize_ensemble_weights(self, y_true):
        """scipy SLSQPでアンサンブル重みを最適化"""
        print("\n   🔍 アンサンブル重み最適化 (SLSQP)...")

        def loss_func(weights):
            weights = np.array(weights)
            weights = np.clip(weights, 0, 1)
            weights /= weights.sum() + 1e-8

            if TABNET_AVAILABLE:
                ens_proba = weights[0] * self.oof_lgb + weights[1] * self.oof_cat + weights[2] * self.oof_tab
            else:
                ens_proba = weights[0] * self.oof_lgb + weights[1] * self.oof_cat

            y_pred = (ens_proba >= 0.5).astype(int)
            return -f1_score(y_true, y_pred)

        if TABNET_AVAILABLE:
            init_weights = [1/3, 1/3, 1/3]
            bounds = [(0.05, 0.9), (0.05, 0.9), (0.05, 0.9)]
            constraints = {'type': 'eq', 'fun': lambda w: 1 - sum(w)}
        else:
            init_weights = [0.5, 0.5]
            bounds = [(0.05, 0.95), (0.05, 0.95)]
            constraints = {'type': 'eq', 'fun': lambda w: 1 - sum(w)}

        result = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)

        best_weights = result.x
        best_weights = np.clip(best_weights, 0, 1)
        best_weights /= best_weights.sum()

        if TABNET_AVAILABLE:
            self.ensemble_weights = (best_weights[0], best_weights[1], best_weights[2])
        else:
            self.ensemble_weights = (best_weights[0], best_weights[1], 0.0)

        print(f"      最適重み: LightGBM={self.ensemble_weights[0]:.3f}, CatBoost={self.ensemble_weights[1]:.3f}", end="")
        if TABNET_AVAILABLE:
            print(f", TabNet={self.ensemble_weights[2]:.3f}")
        else:
            print()

        w = self.ensemble_weights
        if TABNET_AVAILABLE:
            self.oof_ensemble = w[0] * self.oof_lgb + w[1] * self.oof_cat + w[2] * self.oof_tab
        else:
            self.oof_ensemble = w[0] * self.oof_lgb + w[1] * self.oof_cat

        ens_auc = roc_auc_score(y_true, self.oof_ensemble)
        print(f"      アンサンブル OOF AUC: {ens_auc:.4f}")

    def evaluate(self):
        """最終評価（CV OOF）"""
        print("\n📈 最終評価 (Cross Validation OOF)")

        y_s2_true = self.y_s2
        precisions, recalls, thresholds = precision_recall_curve(y_s2_true, self.oof_ensemble)

        self.dynamic_results = {}
        target_recalls = [0.99, 0.98, 0.95]

        print("\n   📊 動的閾値評価:")
        for target_recall in target_recalls:
            idx = np.where(recalls >= target_recall)[0]
            if len(idx) > 0:
                idx = idx[-1]
                best_thresh = thresholds[idx] if idx < len(thresholds) else 0.0
                best_prec = precisions[idx]
            else:
                best_thresh, best_prec = 0.0, 0.0

            self.dynamic_results[target_recall] = {'threshold': best_thresh, 'precision': best_prec}
            print(f"      Recall ~{target_recall:.0%}: 閾値={best_thresh:.4f}, Precision={best_prec:.4f}")

        final_proba = np.zeros(len(self.y))
        final_proba[self.stage2_mask] = self.oof_ensemble
        y_pred = (final_proba >= 0.5).astype(int)

        self.final_precision = precision_score(self.y, y_pred) if y_pred.sum() > 0 else 0
        self.final_recall = recall_score(self.y, y_pred)
        self.final_f1 = f1_score(self.y, y_pred)
        self.final_auc = roc_auc_score(self.y, final_proba)

        print(f"\n   [閾値0.5] Precision: {self.final_precision:.4f}, Recall: {self.final_recall:.4f}, F1: {self.final_f1:.4f}")
        print(f"   [AUC]: {self.final_auc:.4f}")

        return {
            'final_precision': self.final_precision,
            'final_recall': self.final_recall,
            'final_f1': self.final_f1,
            'final_auc': self.final_auc,
        }

    def evaluate_test_set(self):
        """テストセット評価"""
        print("\n📈 テストセット評価 (Hold-Out)")

        test_proba_stage1 = np.zeros(len(self.y_test))
        
        # Testデータもcategory型変換
        X_test_converted = self.X_test.copy()
        for col in self.categorical_cols:
            if col in X_test_converted.columns:
                X_test_converted[col] = X_test_converted[col].astype('category')

        for fold_models in self.stage1_models:
            for model in fold_models:
                test_proba_stage1 += model.predict_proba(X_test_converted)[:, 1]
        test_proba_stage1 /= (self.n_folds * self.n_seeds)

        test_stage2_mask = test_proba_stage1 >= self.threshold_stage1
        n_candidates = test_stage2_mask.sum()
        print(f"   Stage 1 フィルタリング後: {n_candidates:,} / {len(self.y_test):,}")

        if n_candidates == 0:
            print("   ⚠️ Stage 2に進むデータがありません")
            return {}

        X_test_s2 = self._prepare_data_for_models(
            self.X_test[test_stage2_mask].copy(),
            test_proba_stage1[test_stage2_mask]
        )
        y_test_s2 = self.y_test[test_stage2_mask]
        
        # Stage 2用 Testデータもcategory型変換
        for col in self.categorical_cols:
            if col in X_test_s2.columns:
                X_test_s2[col] = X_test_s2[col].astype('category')

        test_lgb = np.zeros(len(y_test_s2))
        test_cat = np.zeros(len(y_test_s2))
        test_tab = np.zeros(len(y_test_s2)) if TABNET_AVAILABLE else None

        for model in self.lgb_models:
            raw_score = model.predict(X_test_s2, raw_score=True)
            # Parallel化の修正: バイナリ+scale_pos_weightなので、predict_probaではシグモイドは自動適用済み
            # ただし raw_score=True でとった場合はシグモイドする
            test_lgb += (1.0 / (1.0 + np.exp(-raw_score))) / self.n_folds

        for model in self.cat_models:
            test_cat += model.predict_proba(X_test_s2)[:, 1] / self.n_folds

        if TABNET_AVAILABLE:
            num_cols = self.numerical_cols + ['prob_stage1']
            cat_cols = self.categorical_cols
            for fold_idx, (model, (imputer, scaler, encoder)) in enumerate(
                zip(self.tab_models, self.tabnet_preprocessors)
            ):
                X_num = scaler.transform(imputer.transform(X_test_s2[num_cols].values))
                X_cat = encoder.transform(X_test_s2[cat_cols].values) + 1
                X_test_tab = np.hstack([X_num, X_cat]).astype(np.float32)
                test_tab += model.predict_proba(X_test_tab)[:, 1] / self.n_folds

        w = self.ensemble_weights
        if TABNET_AVAILABLE:
            test_ensemble = w[0] * test_lgb + w[1] * test_cat + w[2] * test_tab
        else:
            test_ensemble = w[0] * test_lgb + w[1] * test_cat

        final_test_proba = np.zeros(len(self.y_test))
        final_test_proba[test_stage2_mask] = test_ensemble
        y_test_pred = (final_test_proba >= 0.5).astype(int)

        test_precision = precision_score(self.y_test, y_test_pred) if y_test_pred.sum() > 0 else 0
        test_recall = recall_score(self.y_test, y_test_pred)
        test_f1 = f1_score(self.y_test, y_test_pred)
        test_auc = roc_auc_score(self.y_test, final_test_proba)

        print(f"\n   [テスト閾値0.5] Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
        print(f"   [テストAUC]: {test_auc:.4f}")

        self.test_results = {
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auc': test_auc,
        }
        return self.test_results

    def generate_report(self, results, elapsed_sec):
        """実験レポート生成"""
        report_path = os.path.join(self.output_dir, "experiment_report.md")

        w = self.ensemble_weights
        report_content = f"""# アンサンブルモデル実験レポート (v3.1 - 並列処理 + 細粒度チェックポイント)

**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**実行時間**: {elapsed_sec:.1f}秒
**並列設定**: {self.n_jobs} Folds同時実行

## モデル構成
- **Stage 1**: LightGBM + Under-sampling (1:2) + 3-Seed
- **Stage 2 アンサンブル**:
  - LightGBM (Binary + ScalePosWeight): 重み {w[0]:.3f}
  - CatBoost: 重み {w[1]:.3f}
  - TabNet: 重み {w[2]:.3f}

## Stage 1 結果
- 閾値: {self.threshold_stage1:.4f}
- Recall: {self.stage1_recall:.4f}
- フィルタリング率: {self.filter_rate*100:.2f}%

## Stage 2 個別モデル OOF AUC
- LightGBM: {roc_auc_score(self.y_s2, self.oof_lgb):.4f}
- CatBoost: {roc_auc_score(self.y_s2, self.oof_cat):.4f}
{f"- TabNet: {roc_auc_score(self.y_s2, self.oof_tab):.4f}" if TABNET_AVAILABLE else "- TabNet: N/A"}

## アンサンブル結果

### CV OOF (閾値0.5)
| 指標 | 値 |
|------|-----|
| Precision | {results['final_precision']:.4f} |
| Recall | {results['final_recall']:.4f} |
| F1 | {results['final_f1']:.4f} |
| AUC | {results['final_auc']:.4f} |

### テストセット (閾値0.5)
| 指標 | 値 |
|------|-----|
| Precision | {self.test_results.get('test_precision', 0):.4f} |
| Recall | {self.test_results.get('test_recall', 0):.4f} |
| F1 | {self.test_results.get('test_f1', 0):.4f} |
| AUC | {self.test_results.get('test_auc', 0):.4f} |

## 動的閾値評価 (CV OOF)
| Target Recall | 閾値 | Precision |
|---------------|------|-----------|
| 99% | {self.dynamic_results.get(0.99, {}).get('threshold', 0):.4f} | {self.dynamic_results.get(0.99, {}).get('precision', 0):.4f} |
| 98% | {self.dynamic_results.get(0.98, {}).get('threshold', 0):.4f} | {self.dynamic_results.get(0.98, {}).get('precision', 0):.4f} |
| 95% | {self.dynamic_results.get(0.95, {}).get('threshold', 0):.4f} | {self.dynamic_results.get(0.95, {}).get('precision', 0):.4f} |
"""

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        print(f"\n   📄 レポート出力: {report_path}")

    def run(self):
        """パイプライン実行"""
        start = datetime.now()
        self.load_data()
        self.train_stage1()
        self.find_recall_threshold()
        self.train_stage2_ensemble()
        results = self.evaluate()
        test_results = self.evaluate_test_set()
        results.update(test_results)

        elapsed_sec = (datetime.now() - start).total_seconds()
        results['elapsed_sec'] = elapsed_sec

        pd.DataFrame([results]).to_csv(os.path.join(self.output_dir, "final_results.csv"), index=False)
        self.generate_report(results, elapsed_sec)

        # チェックポイントのクリーンアップは意図的に行わない（次回のため、またはモデル再利用のため）
        # 必要なら del self.stage1_models; gc.collect() はメモリ解放のためにだけ行う
        del self.stage1_models
        gc.collect()

        print("\n" + "=" * 70)
        print("✅ 完了！")
        print(f"   総実行時間: {elapsed_sec:.1f}秒")
        print(f"   結果CSV: {self.output_dir}/final_results.csv")
        print(f"   レポートMD: {self.output_dir}/experiment_report.md")
        print("=" * 70)

        return results


if __name__ == "__main__":
    pipeline = EnsembleStage2Pipeline()
    pipeline.run()
