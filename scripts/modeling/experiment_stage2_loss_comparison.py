"""
Stage 2 損失関数比較実験
========================
Stage 1 (Recall 95%) でフィルタリング後、Stage 2 で LogLoss vs Focal Loss を比較する。

実験設定:
- Stage 1: LightGBM + Under-sampling, Target Recall = 95%
- Stage 2 Model A: LightGBM + LogLoss (is_unbalance=True)
- Stage 2 Model B: LightGBM + Focal Loss (alpha/gamma optimized)
- 特徴量: 元特徴量 + prob_stage1 + Cat-Interaction

出力:
- PR曲線の比較プロット
- 各モデルの Precision@Recall テーブル
- 実験レポート (Markdown)
"""

import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import lightgbm as lgb
import matplotlib.pyplot as plt
import platform
from scipy.special import expit
import warnings

warnings.filterwarnings('ignore')

# 日本語フォント設定 (プラットフォーム別)
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'MS Gothic'
elif platform.system() == 'Darwin':  # Mac
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    # Linux - DejaVu Sansをフォールバックとして使用
    plt.rcParams['font.family'] = 'DejaVu Sans'


def get_focal_loss_lgb(alpha: float, gamma: float):
    """
    Focal Loss for LightGBM (Robust Implementation)
    
    - scipy.special.expit を使用して数値安定性を確保
    - Gradient: 重み付きLogLoss近似 (GBDTでは収束が早い)
    - Hessian: Gauss-Newton近似
    """
    def focal_loss(preds, train_data):
        y_true = train_data.get_label()
        
        # シグモイド関数 (数値安定版)
        p = expit(preds)
        
        # 正例/負例の重み計算
        is_pos = (y_true == 1)
        is_neg = (y_true == 0)
        
        weights = np.zeros_like(preds)
        weights[is_pos] = alpha * np.power(1 - p[is_pos], gamma)
        weights[is_neg] = (1 - alpha) * np.power(p[is_neg], gamma)
        
        # Gradient (1次微分): 重み付き(p - y)
        grad = weights * (p - y_true)
        
        # Hessian (2次微分): Gauss-Newton近似
        hess = weights * p * (1 - p)
        hess = np.clip(hess, 1e-7, 1e7)
        
        return grad, hess
    
    def focal_eval(preds, train_data):
        y_true = train_data.get_label()
        p = expit(preds)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        loss = np.zeros_like(preds)
        pos_mask = (y_true == 1)
        neg_mask = (y_true == 0)
        
        loss[pos_mask] = -alpha * np.power(1 - p[pos_mask], gamma) * np.log(p[pos_mask])
        loss[neg_mask] = -(1 - alpha) * np.power(p[neg_mask], gamma) * np.log(1 - p[neg_mask])
        
        return 'focal_loss', np.mean(loss), False
    
    return focal_loss, focal_eval


class Stage2LossComparisonExperiment:
    """Stage 2 損失関数比較実験パイプライン"""
    
    def __init__(
        self,
        data_path: str = "data/processed/honhyo_clean_with_features.csv",
        target_col: str = "死者数",
        n_folds: int = 5,
        random_state: int = 42,
        # Stage 1 設定
        stage1_recall_target: float = 0.95,
        undersample_ratio: float = 2.0,
        n_seeds: int = 3,
        test_size: float = 0.2,
        # Focal Loss パラメータ
        focal_alpha: float = 0.6321,
        focal_gamma: float = 1.1495,
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
        
        self.output_dir = "results/experiments/stage2_loss_comparison"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 60)
        print("Stage 2 損失関数比較実験")
        print(f"Stage 1: Recall {self.stage1_recall_target:.0%} フィルタリング")
        print(f"Stage 2: LogLoss vs Focal Loss (α={self.focal_alpha:.4f}, γ={self.focal_gamma:.4f})")
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
    
    def train_stage1(self):
        """Stage 1: LightGBM + Under-sampling"""
        print(f"\n🌿 Stage 1: LightGBM + Under-sampling (1:{int(self.undersample_ratio)})")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.oof_proba_stage1 = np.zeros(len(self.y))
        self.stage1_models = []
        
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'num_leaves': 31,
            'max_depth': 8,
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
            
            self.oof_proba_stage1[val_idx] = fold_proba
            self.stage1_models.append(fold_models)
            
            del X_train, X_val
            gc.collect()
        
        oof_auc = roc_auc_score(self.y, self.oof_proba_stage1)
        print(f"   Stage 1 OOF AUC: {oof_auc:.4f}")
        
        # 閾値決定 (Target Recall)
        self.stage1_threshold = self._find_threshold_for_recall(
            self.y, self.oof_proba_stage1, self.stage1_recall_target
        )
        
        pred_stage1 = (self.oof_proba_stage1 >= self.stage1_threshold).astype(int)
        stage1_recall = recall_score(self.y, pred_stage1)
        filter_rate = 1 - pred_stage1.mean()
        
        print(f"   閾値: {self.stage1_threshold:.4f}, Recall: {stage1_recall:.4f}")
        print(f"   フィルタリング率: {filter_rate:.2%} 除外, 候補: {pred_stage1.sum():,}")
        
        self.stage1_results = {
            'auc': oof_auc,
            'threshold': self.stage1_threshold,
            'recall': stage1_recall,
            'filter_rate': filter_rate,
        }
    
    def _find_threshold_for_recall(self, y_true, y_proba, target_recall):
        """指定したRecallを達成する閾値を探索"""
        for thresh in np.arange(0.001, 0.5, 0.001):
            pred = (y_proba >= thresh).astype(int)
            rec = recall_score(y_true, pred)
            if rec < target_recall:
                return thresh - 0.001
        return 0.001
    
    def generate_stage2_features(self, X_subset, prob_stage1_subset, fit_categories=True):
        """Stage 2用の拡張特徴量を生成"""
        X_out = X_subset.copy()
        
        # prob_stage1 追加
        X_out['prob_stage1'] = prob_stage1_subset
        
        # Cat-Interaction特徴量
        top_cat_cols = ['警察署等コード', '市区町村コード', '都道府県コード']
        top_cat_cols = [c for c in top_cat_cols if c in X_subset.columns]
        
        self.interaction_categories = getattr(self, 'interaction_categories', {})
        
        for i, col1 in enumerate(top_cat_cols):
            for col2 in top_cat_cols[i+1:]:
                name = f'{col1}_{col2}_interaction'
                interaction_values = X_subset[col1].astype(str) + '_' + X_subset[col2].astype(str)
                
                if fit_categories:
                    cat_type = pd.CategoricalDtype(categories=list(interaction_values.unique()) + ['__UNKNOWN__'])
                    self.interaction_categories[name] = cat_type
                    X_out[name] = pd.Categorical(interaction_values, dtype=cat_type)
                else:
                    if name in self.interaction_categories:
                        known_cats = set(self.interaction_categories[name].categories)
                        interaction_values = interaction_values.apply(
                            lambda x: x if x in known_cats else '__UNKNOWN__'
                        )
                        X_out[name] = pd.Categorical(interaction_values, dtype=self.interaction_categories[name])
                    else:
                        X_out[name] = interaction_values.astype('category')
        
        return X_out
    
    def train_stage2(self):
        """Stage 2: LogLoss vs Focal Loss 比較"""
        print(f"\n🔬 Stage 2: LogLoss vs Focal Loss 比較")
        
        # Stage 1でPositiveと判定されたデータのみを使用
        stage1_positive_mask = self.oof_proba_stage1 >= self.stage1_threshold
        
        X_stage2 = self.X[stage1_positive_mask].copy()
        y_stage2 = self.y[stage1_positive_mask]
        prob_stage1_stage2 = self.oof_proba_stage1[stage1_positive_mask]
        
        print(f"   Stage 2 データ: {len(y_stage2):,} (Pos: {y_stage2.sum():,}, Neg: {(y_stage2 == 0).sum():,})")
        
        # Stage 2 のデータ比率に基づいて alpha を動的に計算/表示
        n_pos_s2 = y_stage2.sum()
        n_neg_s2 = len(y_stage2) - n_pos_s2
        pos_ratio_s2 = n_pos_s2 / len(y_stage2)
        suggested_alpha = 1 - pos_ratio_s2  # 逆頻度重み付け
        
        print(f"   Stage 2 Positive Rate: {pos_ratio_s2:.2%}")
        print(f"   Suggested Alpha (Inverse Freq): {suggested_alpha:.4f}")
        print(f"   Using Alpha: {self.focal_alpha:.4f} (設定値)")
        
        # オプション: 動的にalphaを使用する場合は以下をコメントアウト解除
        # self.focal_alpha = suggested_alpha
        
        # 特徴量生成
        X_stage2_features = self.generate_stage2_features(X_stage2, prob_stage1_stage2, fit_categories=True)
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # OOF predictions
        self.oof_proba_logloss = np.full(len(self.y), np.nan)
        self.oof_proba_focal = np.full(len(self.y), np.nan)
        
        # Stage 2のインデックスマッピング
        stage2_original_indices = np.where(stage1_positive_mask)[0]
        
        # 共通LightGBMパラメータ
        lgb_params_base = {
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'num_leaves': 127,
            'max_depth': 6,
            'min_child_samples': 44,
            'reg_alpha': 2.3897,
            'reg_lambda': 2.2842,
            'colsample_bytree': 0.8646,
            'subsample': 0.6328,
            'n_estimators': 2000,
            'learning_rate': 0.0477,
            'n_jobs': -1
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_stage2_features, y_stage2)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            
            X_train = X_stage2_features.iloc[train_idx]
            X_val = X_stage2_features.iloc[val_idx]
            y_train = y_stage2[train_idx]
            y_val = y_stage2[val_idx]
            
            original_val_indices = stage2_original_indices[val_idx]
            
            # -------------------- Model A: LogLoss --------------------
            lgb_params_ll = lgb_params_base.copy()
            lgb_params_ll['objective'] = 'binary'
            lgb_params_ll['metric'] = 'auc'
            lgb_params_ll['is_unbalance'] = True
            
            model_ll = lgb.LGBMClassifier(**lgb_params_ll, random_state=self.random_state + fold)
            model_ll.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
            
            self.oof_proba_logloss[original_val_indices] = model_ll.predict_proba(X_val)[:, 1]
            
            # -------------------- Model B: Focal Loss --------------------
            focal_obj, focal_eval = get_focal_loss_lgb(self.focal_alpha, self.focal_gamma)
            
            lgb_params_focal = lgb_params_base.copy()
            # 新しいLightGBM APIでは objective パラメータに関数を直接渡す
            lgb_params_focal['objective'] = focal_obj
            lgb_params_focal['metric'] = 'None'  # カスタム損失ではデフォルトmetricを無効化
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model_focal = lgb.train(
                lgb_params_focal,
                train_data,
                num_boost_round=2000,
                valid_sets=[val_data],
                valid_names=['valid'],
                feval=focal_eval,
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
            
            # Logits -> Probability
            logits = model_focal.predict(X_val)
            self.oof_proba_focal[original_val_indices] = 1.0 / (1.0 + np.exp(-logits))
            
            gc.collect()
        
        # NaNを除外して評価
        valid_mask = ~np.isnan(self.oof_proba_logloss)
        y_valid = self.y[valid_mask]
        oof_ll = self.oof_proba_logloss[valid_mask]
        oof_focal = self.oof_proba_focal[valid_mask]
        
        auc_ll = roc_auc_score(y_valid, oof_ll)
        auc_focal = roc_auc_score(y_valid, oof_focal)
        ap_ll = average_precision_score(y_valid, oof_ll)
        ap_focal = average_precision_score(y_valid, oof_focal)
        
        print(f"\n   📊 Stage 2 結果 (CV OOF):")
        print(f"      LogLoss:    AUC={auc_ll:.4f}, AP={ap_ll:.4f}")
        print(f"      Focal Loss: AUC={auc_focal:.4f}, AP={ap_focal:.4f}")
        
        self.stage2_results = {
            'logloss_auc': auc_ll,
            'focal_auc': auc_focal,
            'logloss_ap': ap_ll,
            'focal_ap': ap_focal,
        }
    
    def evaluate_precision_at_recall(self):
        """Precision@Recall テーブルを生成"""
        print("\n📈 Precision@Recall 評価...")
        
        valid_mask = ~np.isnan(self.oof_proba_logloss)
        y_valid = self.y[valid_mask]
        oof_ll = self.oof_proba_logloss[valid_mask]
        oof_focal = self.oof_proba_focal[valid_mask]
        
        # 全体Recall = Stage1 Recall × Stage2 Recall
        # Stage 1 Recall は stage1_results['recall'] で固定
        stage1_recall = self.stage1_results['recall']
        
        recall_targets = [0.99, 0.95, 0.90, 0.85, 0.80]
        self.precision_table = []
        
        for r in recall_targets:
            # Stage 2 のターゲット Recall (Stage 1は固定)
            # 全体Recall = r を達成するには、Stage 2 Recall = r / stage1_recall
            # ただし、Stage 2 内の Recall を直接計算
            
            for name, proba in [('LogLoss', oof_ll), ('Focal Loss', oof_focal)]:
                thresh = self._find_threshold_for_recall(y_valid, proba, r)
                pred = (proba >= thresh).astype(int)
                prec = precision_score(y_valid, pred) if pred.sum() > 0 else 0
                rec = recall_score(y_valid, pred)
                
                # 全体Recall (Stage 1 × Stage 2)
                total_recall = stage1_recall * rec
                
                self.precision_table.append({
                    'Model': name,
                    'Stage2_Recall_Target': f'{r:.0%}',
                    'Stage2_Recall_Actual': f'{rec:.2%}',
                    'Total_Recall': f'{total_recall:.2%}',
                    'Precision': f'{prec:.4f}',
                    'Threshold': f'{thresh:.4f}',
                })
        
        df_prec = pd.DataFrame(self.precision_table)
        print(df_prec.to_string(index=False))
        
        return df_prec
    
    def plot_pr_curves(self):
        """PR曲線の比較プロット"""
        print("\n📊 PR曲線プロット...")
        
        valid_mask = ~np.isnan(self.oof_proba_logloss)
        y_valid = self.y[valid_mask]
        oof_ll = self.oof_proba_logloss[valid_mask]
        oof_focal = self.oof_proba_focal[valid_mask]
        
        prec_ll, rec_ll, _ = precision_recall_curve(y_valid, oof_ll)
        prec_focal, rec_focal, _ = precision_recall_curve(y_valid, oof_focal)
        
        ap_ll = average_precision_score(y_valid, oof_ll)
        ap_focal = average_precision_score(y_valid, oof_focal)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.plot(rec_ll, prec_ll, label=f'LogLoss (AP={ap_ll:.4f})', linewidth=2)
        ax.plot(rec_focal, prec_focal, label=f'Focal Loss (AP={ap_focal:.4f})', linewidth=2, linestyle='--')
        
        ax.set_xlabel('Recall (Stage 2)', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(f'Stage 2: LogLoss vs Focal Loss\n(Stage 1 Recall = {self.stage1_results["recall"]:.2%})', fontsize=14)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plot_path = os.path.join(self.output_dir, "pr_curve_comparison.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"   📄 プロット保存: {plot_path}")
        return plot_path
    
    def generate_report(self, elapsed_sec: float):
        """実験レポートをMarkdownで出力"""
        report_path = os.path.join(self.output_dir, "experiment_report.md")
        
        df_prec = pd.DataFrame(self.precision_table)
        # 手動でMarkdownテーブルを生成 (tabulate依存を回避)
        prec_table_md = df_prec.to_string(index=False)
        
        report_content = f"""# Stage 2 損失関数比較実験レポート

**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**実行時間**: {elapsed_sec:.1f}秒

## 実験設定

| 項目 | 値 |
|------|----| 
| Stage 1 Target Recall | {self.stage1_recall_target:.0%} |
| Stage 1 Actual Recall | {self.stage1_results['recall']:.2%} |
| Stage 1 フィルタリング率 | {self.stage1_results['filter_rate']:.2%} |
| Focal Loss α | {self.focal_alpha:.4f} |
| Focal Loss γ | {self.focal_gamma:.4f} |

## Stage 2 結果 (CV OOF)

| モデル | AUC | Average Precision |
|--------|-----|-------------------|
| LogLoss | {self.stage2_results['logloss_auc']:.4f} | {self.stage2_results['logloss_ap']:.4f} |
| Focal Loss | {self.stage2_results['focal_auc']:.4f} | {self.stage2_results['focal_ap']:.4f} |

## Precision @ Recall テーブル

{prec_table_md}

## PR曲線

![PR Curve Comparison](pr_curve_comparison.png)

## 考察

- Stage 1 Recall = {self.stage1_results['recall']:.2%} により、{self.stage1_results['filter_rate']:.2%} のデータを除外。
- Stage 2 での LogLoss vs Focal Loss を比較。
- Average Precision (AP) は PR曲線の下面積であり、全体的な性能を表す。
- 高Recall領域での Precision に注目してモデルを選択する。
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n   📄 レポート出力: {report_path}")
        return report_path
    
    def run(self):
        """パイプライン実行"""
        start = datetime.now()
        
        self.load_data()
        self.train_stage1()
        self.train_stage2()
        self.evaluate_precision_at_recall()
        self.plot_pr_curves()
        
        elapsed_sec = (datetime.now() - start).total_seconds()
        
        # 結果保存
        all_results = {**self.stage1_results, **self.stage2_results, 'elapsed_sec': elapsed_sec}
        pd.DataFrame([all_results]).to_csv(os.path.join(self.output_dir, "results.csv"), index=False)
        
        self.generate_report(elapsed_sec)
        
        print("\n" + "=" * 60)
        print("✅ 完了！")
        print(f"   結果CSV: {self.output_dir}/results.csv")
        print(f"   レポートMD: {self.output_dir}/experiment_report.md")
        print(f"   PR曲線: {self.output_dir}/pr_curve_comparison.png")
        print("=" * 60)
        
        return all_results


if __name__ == "__main__":
    experiment = Stage2LossComparisonExperiment()
    experiment.run()
