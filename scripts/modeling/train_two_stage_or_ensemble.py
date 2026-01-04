"""
OR条件アンサンブル 2段階パイプライン
=====================================
Stage 1: LightGBM + CatBoost OR条件アンサンブル
Stage 2: LightGBM + Focal Loss

実行方法:
    python scripts/modeling/train_two_stage_or_ensemble.py
"""

import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.special import expit
import warnings

warnings.filterwarnings('ignore')


def get_focal_loss_lgb(alpha: float = 0.75, gamma: float = 1.0):
    """LightGBM用 Focal Loss"""
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


class TwoStageORensemblePipeline:
    """OR条件アンサンブル 2段階パイプライン"""
    
    def __init__(
        self,
        data_path: str = "data/processed/honhyo_clean_with_features.csv",
        target_col: str = "死者数",
        n_folds: int = 5,
        random_state: int = 42,
        stage1_recall_target: float = 0.995,  # 安全マージン込み
        undersample_ratio: float = 2.0,
        n_seeds: int = 3,
        test_size: float = 0.2,
        focal_alpha: float = 0.6321,
        focal_gamma: float = 1.1495,
        output_dir: str = "results/two_stage_or_ensemble",
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
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 70)
        print("OR条件アンサンブル 2段階パイプライン")
        print(f"Stage 1: LightGBM + CatBoost OR条件")
        print(f"Stage 2: LightGBM + Focal Loss")
        print(f"Target Recall: {self.stage1_recall_target:.1%}")
        print(f"Test Set: {self.test_size:.0%}")
        print("=" * 70)
    
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
            else:
                self.numerical_cols.append(col)
                X_all[col] = X_all[col].astype(np.float32)
        
        self.feature_names = list(X_all.columns)
        
        print(f"\n📊 データ分割 (Train: {1-self.test_size:.0%} / Test: {self.test_size:.0%})")
        self.X, self.X_test, self.y, self.y_test = train_test_split(
            X_all, y_all, test_size=self.test_size,
            random_state=self.random_state, stratify=y_all
        )
        self.X = self.X.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        
        print(f"   Train: 正例 {self.y.sum():,} / {len(self.y):,}")
        print(f"   Test:  正例 {self.y_test.sum():,} / {len(self.y_test):,}")
        gc.collect()
    
    def undersample(self, X, y, seed):
        """負例をアンダーサンプリング"""
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        n_neg_sample = int(len(pos_idx) * self.undersample_ratio)
        np.random.seed(seed)
        sampled_neg_idx = np.random.choice(neg_idx, size=min(n_neg_sample, len(neg_idx)), replace=False)
        sampled_idx = np.concatenate([pos_idx, sampled_neg_idx])
        np.random.shuffle(sampled_idx)
        return X.iloc[sampled_idx], y[sampled_idx]
    
    def train_stage1_lightgbm(self):
        """Stage 1: LightGBM 学習"""
        print("\n🌲 Stage 1 LightGBM 学習中...")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.oof_proba_lgbm = np.zeros(len(self.y))
        
        lgb_params = {
            'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
            'verbosity': -1, 'num_leaves': 31, 'max_depth': 8,
            'reg_alpha': 0.1, 'reg_lambda': 0.1, 'n_estimators': 1000,
            'learning_rate': 0.05, 'n_jobs': -1
        }
        
        self.lgbm_models = []
        
        X_lgbm = self.X.copy()
        for col in self.categorical_cols:
            if col in X_lgbm.columns:
                X_lgbm[col] = X_lgbm[col].astype('category')
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_lgbm, self.y)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            X_train_full = X_lgbm.iloc[train_idx]
            y_train_full = self.y[train_idx]
            X_val = X_lgbm.iloc[val_idx]
            y_val = self.y[val_idx]
            
            fold_proba = np.zeros(len(val_idx))
            fold_models = []
            
            for seed_offset in range(self.n_seeds):
                seed = self.random_state + fold * 100 + seed_offset
                X_train_under, y_train_under = self.undersample(X_train_full, y_train_full, seed)
                
                for col in self.categorical_cols:
                    if col in X_train_under.columns:
                        X_train_under[col] = X_train_under[col].astype('category')
                
                model = lgb.LGBMClassifier(**lgb_params, random_state=seed)
                model.fit(X_train_under, y_train_under, eval_set=[(X_val, y_val)],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
                
                fold_proba += model.predict_proba(X_val)[:, 1] / self.n_seeds
                fold_models.append(model)
            
            self.oof_proba_lgbm[val_idx] = fold_proba
            self.lgbm_models.append(fold_models)
            gc.collect()
        
        oof_auc = roc_auc_score(self.y, self.oof_proba_lgbm)
        print(f"   LightGBM OOF AUC: {oof_auc:.4f}")
        self.lgbm_oof_auc = oof_auc
    
    def train_stage1_catboost(self):
        """Stage 1: CatBoost 学習"""
        print("\n🐱 Stage 1 CatBoost 学習中...")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.oof_proba_catboost = np.zeros(len(self.y))
        
        X_cat = self.X.copy()
        for col in self.categorical_cols:
            if col in X_cat.columns:
                X_cat[col] = X_cat[col].astype(str)
        
        cat_feature_indices = [X_cat.columns.get_loc(c) for c in self.categorical_cols if c in X_cat.columns]
        
        self.catboost_models = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_cat, self.y)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            X_train_full = X_cat.iloc[train_idx]
            y_train_full = self.y[train_idx]
            X_val = X_cat.iloc[val_idx]
            y_val = self.y[val_idx]
            
            fold_proba = np.zeros(len(val_idx))
            fold_models = []
            
            for seed_offset in range(self.n_seeds):
                seed = self.random_state + fold * 100 + seed_offset
                X_train_under, y_train_under = self.undersample(X_train_full, y_train_full, seed)
                
                model = CatBoostClassifier(
                    iterations=1000, learning_rate=0.05, depth=8, l2_leaf_reg=3,
                    loss_function='Logloss', eval_metric='AUC', random_seed=seed,
                    verbose=False, early_stopping_rounds=50, task_type='CPU',
                    cat_features=cat_feature_indices
                )
                model.fit(X_train_under, y_train_under, eval_set=(X_val, y_val), verbose=False)
                
                fold_proba += model.predict_proba(X_val)[:, 1] / self.n_seeds
                fold_models.append(model)
            
            self.oof_proba_catboost[val_idx] = fold_proba
            self.catboost_models.append(fold_models)
            gc.collect()
        
        oof_auc = roc_auc_score(self.y, self.oof_proba_catboost)
        print(f"   CatBoost OOF AUC: {oof_auc:.4f}")
        self.catboost_oof_auc = oof_auc
    
    def find_or_ensemble_threshold(self):
        """OR条件アンサンブルの閾値決定"""
        print("\n🎯 OR条件閾値決定...")
        
        # 各モデルの閾値探索
        self.thresholds = {}
        for name, proba in [('lgbm', self.oof_proba_lgbm), ('catboost', self.oof_proba_catboost)]:
            for thresh in np.arange(0.001, 0.5, 0.001):
                pred = (proba >= thresh).astype(int)
                rec = recall_score(self.y, pred)
                if rec < self.stage1_recall_target:
                    self.thresholds[name] = thresh - 0.001
                    break
            else:
                self.thresholds[name] = 0.001
        
        # OR条件判定
        pred_lgbm = (self.oof_proba_lgbm >= self.thresholds['lgbm']).astype(int)
        pred_catboost = (self.oof_proba_catboost >= self.thresholds['catboost']).astype(int)
        self.stage2_mask = np.maximum(pred_lgbm, pred_catboost) == 1
        
        # 結果出力
        or_recall = recall_score(self.y, self.stage2_mask.astype(int))
        filter_rate = 1 - self.stage2_mask.mean()
        
        print(f"   LightGBM閾値: {self.thresholds['lgbm']:.4f}")
        print(f"   CatBoost閾値: {self.thresholds['catboost']:.4f}")
        print(f"   OR条件 Recall: {or_recall:.4f}")
        print(f"   フィルタリング率: {filter_rate:.2%}")
        print(f"   Stage 2 候補数: {self.stage2_mask.sum():,} / {len(self.y):,}")
        
        self.stage1_recall = or_recall
        self.filter_rate = filter_rate
        
        # モデル間相関
        self.model_correlation = np.corrcoef(self.oof_proba_lgbm, self.oof_proba_catboost)[0, 1]
        print(f"\n   🔗 モデル間相関: {self.model_correlation:.4f}")
    
    def generate_stage2_features(self, X_subset, prob_lgbm, prob_catboost):
        """Stage 2用特徴量生成"""
        X_out = X_subset.copy()
        X_out['prob_lgbm'] = prob_lgbm
        X_out['prob_catboost'] = prob_catboost
        X_out['prob_avg'] = (prob_lgbm + prob_catboost) / 2
        X_out['prob_max'] = np.maximum(prob_lgbm, prob_catboost)
        return X_out
    
    def train_stage2(self):
        """Stage 2: LightGBM + Focal Loss"""
        print("\n🌿 Stage 2: LightGBM + Focal Loss 学習中...")
        
        # Stage 2用データ準備
        stage2_indices = np.where(self.stage2_mask)[0]
        
        X_s2 = self.generate_stage2_features(
            self.X.iloc[stage2_indices].copy(),
            self.oof_proba_lgbm[stage2_indices],
            self.oof_proba_catboost[stage2_indices]
        ).reset_index(drop=True)
        
        y_s2 = self.y[stage2_indices]
        
        # カテゴリ型変換
        for col in self.categorical_cols:
            if col in X_s2.columns:
                X_s2[col] = X_s2[col].astype('category')
        
        n_pos, n_neg = y_s2.sum(), len(y_s2) - y_s2.sum()
        print(f"   Stage 2 データ: {len(y_s2):,} (Pos: {n_pos:,}, Neg: {n_neg:,})")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.oof_proba_stage2 = np.zeros(len(y_s2))
        self.stage2_models = []
        
        focal_loss_fn = get_focal_loss_lgb(alpha=self.focal_alpha, gamma=self.focal_gamma)
        lgb_params = {
            'objective': focal_loss_fn, 'metric': 'auc', 'boosting_type': 'gbdt',
            'verbosity': -1, 'num_leaves': 127, 'max_depth': -1, 'min_child_samples': 44,
            'reg_alpha': 2.3897, 'reg_lambda': 2.2842, 'colsample_bytree': 0.8646,
            'subsample': 0.6328, 'learning_rate': 0.0477, 'n_estimators': 1000, 'n_jobs': -1
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_s2, y_s2)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            X_train, y_train = X_s2.iloc[train_idx], y_s2[train_idx]
            X_val, y_val = X_s2.iloc[val_idx], y_s2[val_idx]
            
            model = lgb.LGBMClassifier(**lgb_params, random_state=self.random_state + fold)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
            
            y_pred_raw = model.predict(X_val, raw_score=True)
            y_pred_proba = 1.0 / (1.0 + np.exp(-y_pred_raw))
            
            self.oof_proba_stage2[val_idx] = y_pred_proba
            self.stage2_models.append(model)
            gc.collect()
        
        oof_auc = roc_auc_score(y_s2, self.oof_proba_stage2)
        print(f"   Stage 2 OOF AUC: {oof_auc:.4f}")
        self.stage2_oof_auc = oof_auc
        self.y_s2 = y_s2
    
    def evaluate(self):
        """最終評価 (CV OOF)"""
        print("\n📈 最終評価 (Cross Validation OOF)")
        
        y_prob_s2 = self.oof_proba_stage2
        y_s2_true = self.y_s2
        
        # 動的閾値
        precisions, recalls, thresholds = precision_recall_curve(y_s2_true, y_prob_s2)
        self.dynamic_results = {}
        
        print("\n   📊 動的閾値評価:")
        for target_recall in [0.99, 0.98, 0.95]:
            idx = np.where(recalls >= target_recall)[0]
            if len(idx) > 0:
                idx = idx[-1]
                best_thresh = thresholds[idx] if idx < len(thresholds) else 0.0
                best_prec = precisions[idx]
            else:
                best_thresh, best_prec = 0.0, 0.0
            self.dynamic_results[target_recall] = {'threshold': best_thresh, 'precision': best_prec}
            print(f"      Recall ~{target_recall:.0%}: 閾値={best_thresh:.4f}, Precision={best_prec:.4f}")
        
        # 固定閾値0.5
        final_proba = np.zeros(len(self.y))
        final_proba[self.stage2_mask] = y_prob_s2
        y_pred = (final_proba >= 0.5).astype(int)
        
        self.final_precision = precision_score(self.y, y_pred) if y_pred.sum() > 0 else 0
        self.final_recall = recall_score(self.y, y_pred)
        self.final_f1 = f1_score(self.y, y_pred)
        self.final_auc = roc_auc_score(self.y, final_proba)
        
        print(f"\n   [閾値0.5] Precision: {self.final_precision:.4f}, Recall: {self.final_recall:.4f}, F1: {self.final_f1:.4f}")
        print(f"   [AUC]: {self.final_auc:.4f}")
        
        return {
            'stage1_lgbm_auc': self.lgbm_oof_auc,
            'stage1_catboost_auc': self.catboost_oof_auc,
            'stage1_recall': self.stage1_recall,
            'filter_rate': self.filter_rate,
            'model_correlation': self.model_correlation,
            'stage2_auc': self.stage2_oof_auc,
            'final_precision': self.final_precision,
            'final_recall': self.final_recall,
            'final_f1': self.final_f1,
            'final_auc': self.final_auc,
        }
    
    def evaluate_test_set(self):
        """テストセット評価"""
        print("\n📈 テストセット評価 (Hold-Out)")
        
        # Stage 1 LightGBM
        X_test_lgbm = self.X_test.copy()
        for col in self.categorical_cols:
            if col in X_test_lgbm.columns:
                X_test_lgbm[col] = X_test_lgbm[col].astype('category')
        
        test_proba_lgbm = np.zeros(len(self.y_test))
        for fold_models in self.lgbm_models:
            for model in fold_models:
                test_proba_lgbm += model.predict_proba(X_test_lgbm)[:, 1]
        test_proba_lgbm /= (self.n_folds * self.n_seeds)
        
        # Stage 1 CatBoost
        X_test_cat = self.X_test.copy()
        for col in self.categorical_cols:
            if col in X_test_cat.columns:
                X_test_cat[col] = X_test_cat[col].astype(str)
        
        test_proba_catboost = np.zeros(len(self.y_test))
        for fold_models in self.catboost_models:
            for model in fold_models:
                test_proba_catboost += model.predict_proba(X_test_cat)[:, 1]
        test_proba_catboost /= (self.n_folds * self.n_seeds)
        
        # OR条件
        pred_lgbm = (test_proba_lgbm >= self.thresholds['lgbm']).astype(int)
        pred_catboost = (test_proba_catboost >= self.thresholds['catboost']).astype(int)
        test_stage2_mask = np.maximum(pred_lgbm, pred_catboost) == 1
        
        test_stage1_recall = recall_score(self.y_test, test_stage2_mask.astype(int))
        print(f"   Stage 1 OR Recall: {test_stage1_recall:.4f}")
        print(f"   Stage 2 候補数: {test_stage2_mask.sum():,} / {len(self.y_test):,}")
        
        if test_stage2_mask.sum() == 0:
            print("   ⚠️ Stage 2に進むデータがありません")
            return {}
        
        # Stage 2
        X_test_s2 = self.generate_stage2_features(
            self.X_test[test_stage2_mask].copy(),
            test_proba_lgbm[test_stage2_mask],
            test_proba_catboost[test_stage2_mask]
        )
        
        for col in self.categorical_cols:
            if col in X_test_s2.columns:
                X_test_s2[col] = X_test_s2[col].astype('category')
        
        test_proba_s2 = np.zeros(test_stage2_mask.sum())
        for model in self.stage2_models:
            raw_score = model.predict(X_test_s2, raw_score=True)
            proba = 1.0 / (1.0 + np.exp(-raw_score))
            test_proba_s2 += proba
        test_proba_s2 /= self.n_folds
        
        # 固定閾値0.5
        final_test_proba = np.zeros(len(self.y_test))
        final_test_proba[test_stage2_mask] = test_proba_s2
        y_test_pred = (final_test_proba >= 0.5).astype(int)
        
        test_precision = precision_score(self.y_test, y_test_pred) if y_test_pred.sum() > 0 else 0
        test_recall = recall_score(self.y_test, y_test_pred)
        test_f1 = f1_score(self.y_test, y_test_pred)
        test_auc = roc_auc_score(self.y_test, final_test_proba)
        
        print(f"\n   [テスト閾値0.5] Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
        print(f"   [テストAUC]: {test_auc:.4f}")
        
        self.test_results = {
            'test_stage1_recall': test_stage1_recall,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_auc': test_auc,
        }
        return self.test_results
    
    def generate_report(self, results, elapsed_sec):
        """レポート生成"""
        report_path = os.path.join(self.output_dir, "experiment_report.md")
        
        report_content = f"""# OR条件アンサンブル 2段階パイプライン レポート

**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**実行時間**: {elapsed_sec:.1f}秒

## パイプライン構成

- **Stage 1**: LightGBM + CatBoost OR条件アンサンブル
- **Stage 2**: LightGBM + Focal Loss (alpha={self.focal_alpha:.4f}, gamma={self.focal_gamma:.4f})

## Stage 1 結果 (CV OOF)

| モデル | OOF AUC | 閾値 |
|--------|---------|------|
| LightGBM | {self.lgbm_oof_auc:.4f} | {self.thresholds['lgbm']:.4f} |
| CatBoost | {self.catboost_oof_auc:.4f} | {self.thresholds['catboost']:.4f} |

### OR条件アンサンブル

| 指標 | 値 |
|------|-----|
| Recall | {self.stage1_recall:.4f} |
| フィルタリング率 | {self.filter_rate:.2%} |
| モデル間相関 | {self.model_correlation:.4f} |

## Stage 2 結果 (CV OOF)

| 指標 | 値 |
|------|-----|
| OOF AUC | {self.stage2_oof_auc:.4f} |

### 動的閾値評価

| Target Recall | 閾値 | Precision |
|---------------|------|-----------|
| 99% | {self.dynamic_results.get(0.99, {}).get('threshold', 0):.4f} | {self.dynamic_results.get(0.99, {}).get('precision', 0):.4f} |
| 98% | {self.dynamic_results.get(0.98, {}).get('threshold', 0):.4f} | {self.dynamic_results.get(0.98, {}).get('precision', 0):.4f} |
| 95% | {self.dynamic_results.get(0.95, {}).get('threshold', 0):.4f} | {self.dynamic_results.get(0.95, {}).get('precision', 0):.4f} |

## 最終結果 (閾値0.5)

| 指標 | CV OOF | Test |
|------|--------|------|
| Precision | {results['final_precision']:.4f} | {self.test_results.get('test_precision', 0):.4f} |
| Recall | {results['final_recall']:.4f} | {self.test_results.get('test_recall', 0):.4f} |
| F1 | {results['final_f1']:.4f} | {self.test_results.get('test_f1', 0):.4f} |
| AUC | {results['final_auc']:.4f} | {self.test_results.get('test_auc', 0):.4f} |

## 考察

- Stage 1 OR条件により Recall {self.stage1_recall:.4f} を達成
- モデル間相関 {self.model_correlation:.4f} で相補性を確認
- Stage 2 で精度改善を実施
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n   📄 レポート出力: {report_path}")
    
    def run(self):
        """パイプライン実行"""
        start = datetime.now()
        
        self.load_data()
        self.train_stage1_lightgbm()
        self.train_stage1_catboost()
        self.find_or_ensemble_threshold()
        self.train_stage2()
        results = self.evaluate()
        test_results = self.evaluate_test_set()
        results.update(test_results)
        
        elapsed_sec = (datetime.now() - start).total_seconds()
        results['elapsed_sec'] = elapsed_sec
        
        pd.DataFrame([results]).to_csv(os.path.join(self.output_dir, "results.csv"), index=False)
        self.generate_report(results, elapsed_sec)
        
        print("\n" + "=" * 70)
        print("✅ 完了！")
        print(f"   結果CSV: {self.output_dir}/results.csv")
        print(f"   レポートMD: {self.output_dir}/experiment_report.md")
        print("=" * 70)
        
        return results


if __name__ == "__main__":
    pipeline = TwoStageORensemblePipeline()
    pipeline.run()
