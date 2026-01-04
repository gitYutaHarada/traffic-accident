"""
Mixture of Experts (MoE) Stage 2 モデル
======================================
Stage 2 を2つの専門家モデルに分割:
- Expert A (Urban Specialist): 市街地・信号なし・カーブなど「誤検知が多かった領域」に特化
- Generalist: それ以外の「標準的な」領域

Expert A最適化:
- 目標: Precision最大化、ただし Recall >= 80% を保証
- 評価指標: F0.5 Score

実行方法:
    python scripts/modeling/train_moe_stage2.py
"""

import pandas as pd
import numpy as np
import os
import gc
import joblib
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, fbeta_score
)
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from catboost import CatBoostClassifier
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# ドメイン判定（ルーター）
# =============================================================================

def is_urban_domain(row, terrain_col='地形', signal_col='信号機', road_col='道路形状'):
    """Urban Domain判定 (Expert A担当領域)"""
    def to_float(x):
        try: return float(x)
        except: return -1
    
    terrain = to_float(row.get(terrain_col, -1))
    signal = to_float(row.get(signal_col, -1))
    road = to_float(row.get(road_col, -1))
    
    return (terrain == 3) or (signal == 7) or (road == 13)


def create_domain_mask(df):
    """データフレーム全体に対してUrban判定を行い、マスクを返す"""
    def to_float(x):
        try: return float(x)
        except: return -1
    
    terrain = df['地形'].apply(to_float)
    signal = df['信号機'].apply(to_float)
    road = df['道路形状'].apply(to_float)
    
    return (terrain == 3) | (signal == 7) | (road == 13)


# =============================================================================
# 単一Fold学習関数（並列用）
# =============================================================================

def train_expert_fold(fold, train_idx, val_idx, X, y, categorical_cols, random_state, expert_name, output_dir):
    """Expert / Generalist: 単一Foldの学習"""
    save_dir = os.path.join(output_dir, f"{expert_name}_fold{fold}")
    os.makedirs(save_dir, exist_ok=True)

    X_train = X.iloc[train_idx].copy()
    y_train = y[train_idx]
    X_val = X.iloc[val_idx].copy()
    y_val = y[val_idx]

    # Categorical変換
    for col in categorical_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
            X_val[col] = X_val[col].astype('category')

    results = {
        'fold': fold, 'val_idx': val_idx,
        'oof_lgb': None, 'oof_cat': None,
        'lgb_model': None, 'cat_model': None,
    }

    # --- LightGBM ---
    path_lgb_model = os.path.join(save_dir, "lgb_model.pkl")
    path_lgb_pred = os.path.join(save_dir, "lgb_pred.npy")

    if os.path.exists(path_lgb_model) and os.path.exists(path_lgb_pred):
        print(f"[{expert_name} Fold {fold}] Loading LightGBM from checkpoint...")
        results['lgb_model'] = joblib.load(path_lgb_model)
        results['oof_lgb'] = np.load(path_lgb_pred)
    else:
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
        
        joblib.dump(lgb_model, path_lgb_model)
        np.save(path_lgb_pred, pred)
        results['lgb_model'] = lgb_model
        results['oof_lgb'] = pred

    # --- CatBoost ---
    path_cat_pkl = os.path.join(save_dir, "cat_model.pkl")
    path_cat_pred = os.path.join(save_dir, "cat_pred.npy")

    if os.path.exists(path_cat_pkl) and os.path.exists(path_cat_pred):
        print(f"[{expert_name} Fold {fold}] Loading CatBoost from checkpoint...")
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

    return results


# =============================================================================
# MoE Pipeline
# =============================================================================

class MoEStage2Pipeline:
    """Mixture of Experts Stage 2 Pipeline"""

    def __init__(
        self,
        data_path: str = "data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv",
        target_col: str = "fatal",
        n_folds: int = 5,
        random_state: int = 42,
        stage1_threshold: float = 0.0400,
        test_size: float = 0.2,
        output_dir: str = "results/moe_stage2",
        n_jobs: int = 5,
        recall_constraint: float = 0.80,  # Expert A: Recall >= 80% 制約
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.n_folds = n_folds
        self.random_state = random_state
        self.stage1_threshold = stage1_threshold
        self.test_size = test_size
        self.output_dir = output_dir
        self.n_jobs = n_jobs
        self.recall_constraint = recall_constraint
        self.n_seeds = 3

        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        print("=" * 80)
        print("Mixture of Experts (MoE) Stage 2 モデル")
        print(f"  Expert A: Urban Domain (Recall >= {self.recall_constraint:.0%} 制約付きPrecision最大化)")
        print(f"  Generalist: Non-Urban Domain")
        print(f"  並列設定: {self.n_jobs} Folds")
        print("=" * 80)

    def load_data(self):
        """データ読み込み & Stage 1 マスク適用"""
        print("\n📂 データ読み込み & Stage 1 マスク適用...")
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

        # Train/Test分割
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=self.test_size, random_state=self.random_state, stratify=y_all
        )
        self.X_train_full = X_train.reset_index(drop=True)
        self.y_train_full = y_train
        self.X_test = X_test.reset_index(drop=True)
        self.y_test = y_test

        # Stage 1 OOF読み込み（前回の実験結果を再利用）
        stage1_ckpt = "results/ensemble_stage2/checkpoints"
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_stage1 = np.zeros(len(self.y_train_full))
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train_full, self.y_train_full)):
            fold_dir = os.path.join(stage1_ckpt, f"stage1_fold{fold}")
            fold_pred = np.zeros(len(val_idx))
            for seed in range(self.n_seeds):
                pred_path = os.path.join(fold_dir, f"seed{seed}_pred.npy")
                if os.path.exists(pred_path):
                    fold_pred += np.load(pred_path)
            oof_stage1[val_idx] = fold_pred / self.n_seeds

        self.stage2_mask = oof_stage1 >= self.stage1_threshold
        self.oof_stage1 = oof_stage1

        self.X_s2 = self.X_train_full[self.stage2_mask].reset_index(drop=True)
        self.y_s2 = self.y_train_full[self.stage2_mask]

        print(f"   Stage 2 データ: {len(self.y_s2):,} (Pos: {self.y_s2.sum():,})")

    def split_domains(self):
        """Urban / Non-Urban にデータを分割"""
        print("\n🏙️  ドメイン分割 (Urban vs Non-Urban)...")

        self.urban_mask = create_domain_mask(self.X_s2)
        self.X_urban = self.X_s2[self.urban_mask].reset_index(drop=True)
        self.y_urban = self.y_s2[self.urban_mask]

        self.X_general = self.X_s2[~self.urban_mask].reset_index(drop=True)
        self.y_general = self.y_s2[~self.urban_mask]

        print(f"   Urban Domain (Expert A): {len(self.y_urban):,} (Pos: {self.y_urban.sum():,})")
        print(f"   Non-Urban (Generalist): {len(self.y_general):,} (Pos: {self.y_general.sum():,})")

    def _train_expert(self, X, y, expert_name):
        """Expert / Generalist の学習 (並列CV)"""
        print(f"\n🔧 {expert_name} 学習中...")
        start = datetime.now()

        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        folds_data = list(skf.split(X, y))

        results = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(train_expert_fold)(
                fold, train_idx, val_idx, X, y,
                self.categorical_cols, self.random_state, expert_name, self.ckpt_dir
            ) for fold, (train_idx, val_idx) in enumerate(folds_data)
        )

        elapsed = (datetime.now() - start).total_seconds()
        print(f"   ✅ {expert_name} 完了 ({elapsed:.1f}秒)")

        # OOF集約
        oof_lgb = np.zeros(len(y))
        oof_cat = np.zeros(len(y))
        lgb_models = [None] * self.n_folds
        cat_models = [None] * self.n_folds

        for res in results:
            fold = res['fold']
            val_idx = res['val_idx']
            oof_lgb[val_idx] = res['oof_lgb']
            oof_cat[val_idx] = res['oof_cat']
            lgb_models[fold] = res['lgb_model']
            cat_models[fold] = res['cat_model']

        return oof_lgb, oof_cat, lgb_models, cat_models

    def train_experts(self):
        """Expert A と Generalist を学習"""
        # Expert A (Urban)
        self.oof_urban_lgb, self.oof_urban_cat, self.urban_lgb_models, self.urban_cat_models = \
            self._train_expert(self.X_urban, self.y_urban, "ExpertA_Urban")

        # Generalist (Non-Urban)
        self.oof_general_lgb, self.oof_general_cat, self.general_lgb_models, self.general_cat_models = \
            self._train_expert(self.X_general, self.y_general, "Generalist_NonUrban")

    def optimize_thresholds(self):
        """各Expertの閾値を最適化 (制約付き)"""
        print("\n🎯 閾値最適化 (Recall >= {:.0%} 制約付き Precision最大化)...".format(self.recall_constraint))

        # Expert A: Ensemble (0.5:0.5)
        oof_urban_ens = 0.5 * self.oof_urban_lgb + 0.5 * self.oof_urban_cat

        # Constrained Optimization for Expert A
        precisions, recalls, thresholds = precision_recall_curve(self.y_urban, oof_urban_ens)
        
        # Recall >= recall_constraint を満たす中で最大のPrecision
        valid_idx = np.where(recalls[:-1] >= self.recall_constraint)[0]
        if len(valid_idx) > 0:
            # valid_idx[-1] が最も高い閾値（Precisionが高い傾向）
            best_idx = valid_idx[-1]
            self.urban_threshold = thresholds[best_idx]
            self.urban_precision = precisions[best_idx]
            self.urban_recall = recalls[best_idx]
        else:
            # 制約を満たせない場合はF0.5最大化に切り替え
            f05_scores = (1 + 0.5**2) * (precisions * recalls) / (0.5**2 * precisions + recalls + 1e-8)
            best_idx = np.argmax(f05_scores[:-1])
            self.urban_threshold = thresholds[best_idx]
            self.urban_precision = precisions[best_idx]
            self.urban_recall = recalls[best_idx]


        print(f"   [Expert A] Threshold: {self.urban_threshold:.4f}, Precision: {self.urban_precision:.4f}, Recall: {self.urban_recall:.4f}")

        # Generalist: F1最大化 (標準)
        oof_general_ens = 0.5 * self.oof_general_lgb + 0.5 * self.oof_general_cat
        precisions_g, recalls_g, thresholds_g = precision_recall_curve(self.y_general, oof_general_ens)
        f1_scores = 2 * (precisions_g * recalls_g) / (precisions_g + recalls_g + 1e-8)
        best_idx_g = np.argmax(f1_scores[:-1])
        self.general_threshold = thresholds_g[best_idx_g]
        self.general_precision = precisions_g[best_idx_g]
        self.general_recall = recalls_g[best_idx_g]

        print(f"   [Generalist] Threshold: {self.general_threshold:.4f}, Precision: {self.general_precision:.4f}, Recall: {self.general_recall:.4f}")

        self.oof_urban_ens = oof_urban_ens
        self.oof_general_ens = oof_general_ens

    def evaluate(self):
        """全体評価 (Urban + Non-Urban を統合)"""
        print("\n📈 MoE 統合評価...")

        # 統合予測
        # Urban: 閾値 urban_threshold で判定
        # Non-Urban: 閾値 general_threshold で判定
        y_pred_urban = (self.oof_urban_ens >= self.urban_threshold).astype(int)
        y_pred_general = (self.oof_general_ens >= self.general_threshold).astype(int)

        # Stage 2 全体に対する予測
        y_s2_true = self.y_s2
        y_s2_pred = np.zeros(len(y_s2_true))
        y_s2_pred[self.urban_mask.values] = y_pred_urban
        y_s2_pred[~self.urban_mask.values] = y_pred_general

        moe_precision = precision_score(y_s2_true, y_s2_pred) if y_s2_pred.sum() > 0 else 0
        moe_recall = recall_score(y_s2_true, y_s2_pred)
        moe_f1 = f1_score(y_s2_true, y_s2_pred)
        moe_f05 = fbeta_score(y_s2_true, y_s2_pred, beta=0.5)

        print(f"\n   === MoE Stage 2 (OOF) ===")
        print(f"   Precision: {moe_precision:.4f}")
        print(f"   Recall:    {moe_recall:.4f}")
        print(f"   F1 Score:  {moe_f1:.4f}")
        print(f"   F0.5 Score:{moe_f05:.4f}")

        self.moe_results = {
            'moe_precision': moe_precision,
            'moe_recall': moe_recall,
            'moe_f1': moe_f1,
            'moe_f05': moe_f05,
        }

        return self.moe_results

    def evaluate_test_set(self):
        """テストセット評価 (Hold-out)"""
        print("\n📈 テストセット評価 (Hold-Out)...")

        # --- Stage 1 フィルタリング（テストデータ用）---
        # Stage 1のモデルを使ってテストデータを予測
        stage1_ckpt = "results/ensemble_stage2/checkpoints"
        test_proba_stage1 = np.zeros(len(self.y_test))

        # テストデータをカテゴリ変換
        X_test_converted = self.X_test.copy()
        for col in self.categorical_cols:
            if col in X_test_converted.columns:
                X_test_converted[col] = X_test_converted[col].astype('category')

        # Stage 1 モデルの読み込みと予測
        for fold in range(self.n_folds):
            fold_dir = os.path.join(stage1_ckpt, f"stage1_fold{fold}")
            for seed in range(self.n_seeds):
                model_path = os.path.join(fold_dir, f"seed{seed}_model.pkl")
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    test_proba_stage1 += model.predict_proba(X_test_converted)[:, 1]

        test_proba_stage1 /= (self.n_folds * self.n_seeds)

        # Stage 1 Threshold適用
        test_stage2_mask = test_proba_stage1 >= self.stage1_threshold
        n_candidates = test_stage2_mask.sum()
        print(f"   Stage 1 フィルタリング後: {n_candidates:,} / {len(self.y_test):,}")

        if n_candidates == 0:
            print("   ⚠️ Stage 2に進むデータがありません")
            return {}

        X_test_s2 = self.X_test[test_stage2_mask].reset_index(drop=True)
        y_test_s2 = self.y_test[test_stage2_mask]

        # --- ドメイン分割 (テストデータ) ---
        test_urban_mask = create_domain_mask(X_test_s2)
        X_test_urban = X_test_s2[test_urban_mask].reset_index(drop=True)
        y_test_urban = y_test_s2[test_urban_mask]
        X_test_general = X_test_s2[~test_urban_mask].reset_index(drop=True)
        y_test_general = y_test_s2[~test_urban_mask]

        print(f"   Test Urban: {len(y_test_urban):,} (Pos: {y_test_urban.sum():,})")
        print(f"   Test Non-Urban: {len(y_test_general):,} (Pos: {y_test_general.sum():,})")

        # カテゴリ型変換
        for col in self.categorical_cols:
            if col in X_test_urban.columns:
                X_test_urban[col] = X_test_urban[col].astype('category')
            if col in X_test_general.columns:
                X_test_general[col] = X_test_general[col].astype('category')

        # --- Expert A (Urban) 予測 ---
        test_urban_lgb = np.zeros(len(y_test_urban))
        test_urban_cat = np.zeros(len(y_test_urban))

        for model in self.urban_lgb_models:
            test_urban_lgb += model.predict_proba(X_test_urban)[:, 1] / self.n_folds
        for model in self.urban_cat_models:
            test_urban_cat += model.predict_proba(X_test_urban)[:, 1] / self.n_folds

        test_urban_ens = 0.5 * test_urban_lgb + 0.5 * test_urban_cat
        y_pred_test_urban = (test_urban_ens >= self.urban_threshold).astype(int)

        # --- Generalist (Non-Urban) 予測 ---
        test_general_lgb = np.zeros(len(y_test_general))
        test_general_cat = np.zeros(len(y_test_general))

        for model in self.general_lgb_models:
            test_general_lgb += model.predict_proba(X_test_general)[:, 1] / self.n_folds
        for model in self.general_cat_models:
            test_general_cat += model.predict_proba(X_test_general)[:, 1] / self.n_folds

        test_general_ens = 0.5 * test_general_lgb + 0.5 * test_general_cat
        y_pred_test_general = (test_general_ens >= self.general_threshold).astype(int)

        # --- 統合評価 ---
        y_test_s2_pred = np.zeros(len(y_test_s2))
        y_test_s2_pred[test_urban_mask.values] = y_pred_test_urban
        y_test_s2_pred[~test_urban_mask.values] = y_pred_test_general

        test_precision = precision_score(y_test_s2, y_test_s2_pred) if y_test_s2_pred.sum() > 0 else 0
        test_recall = recall_score(y_test_s2, y_test_s2_pred)
        test_f1 = f1_score(y_test_s2, y_test_s2_pred)
        test_f05 = fbeta_score(y_test_s2, y_test_s2_pred, beta=0.5)

        print(f"\n   === MoE Stage 2 (Test Set) ===")
        print(f"   Precision: {test_precision:.4f}")
        print(f"   Recall:    {test_recall:.4f}")
        print(f"   F1 Score:  {test_f1:.4f}")
        print(f"   F0.5 Score:{test_f05:.4f}")

        self.test_results = {
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_f05': test_f05,
        }

        return self.test_results

    def run(self):
        """パイプライン実行"""
        start = datetime.now()
        self.load_data()
        self.split_domains()
        self.train_experts()
        self.optimize_thresholds()
        results = self.evaluate()
        test_results = self.evaluate_test_set()
        results.update(test_results)
        elapsed_sec = (datetime.now() - start).total_seconds()
        results['elapsed_sec'] = elapsed_sec

        # 結果保存
        pd.DataFrame([results]).to_csv(os.path.join(self.output_dir, "moe_results.csv"), index=False)

        print("\n" + "=" * 70)
        print("✅ MoE Stage 2 完了!")
        print(f"   総実行時間: {elapsed_sec:.1f}秒")
        print(f"   結果CSV: {self.output_dir}/moe_results.csv")
        print("=" * 70)

        return results


if __name__ == "__main__":
    pipeline = MoEStage2Pipeline()
    pipeline.run()
