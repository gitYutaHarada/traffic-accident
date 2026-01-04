"""
Stage 1 CatBoost vs LightGBM 比較実験
======================================
LightGBM と CatBoost の単体性能、および OR 条件アンサンブルによる
フィルタリング率改善を検証する。

実行方法:
    python scripts/modeling/train_stage1_catboost_comparison.py
"""

import pandas as pd
import numpy as np
import os
import gc
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')


class Stage1CatBoostComparisonPipeline:
    """Stage 1 CatBoost vs LightGBM 比較パイプライン"""
    
    def __init__(
        self,
        data_path: str = "data/processed/honhyo_clean_with_features.csv",
        target_col: str = "死者数",
        n_folds: int = 5,
        random_state: int = 42,
        target_recall: float = 0.99,
        undersample_ratio: float = 2.0,  # 1:2
        n_seeds: int = 3,
        test_size: float = 0.2,
        output_dir: str = "results/stage1_catboost_comparison",
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.n_folds = n_folds
        self.random_state = random_state
        self.target_recall = target_recall
        self.undersample_ratio = undersample_ratio
        self.n_seeds = n_seeds
        self.test_size = test_size
        self.output_dir = output_dir
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 70)
        print("Stage 1 CatBoost vs LightGBM 比較実験")
        print(f"Target Recall: {self.target_recall:.0%}")
        print(f"Under-sampling: 1:{int(self.undersample_ratio)}")
        print(f"Seeds: {self.n_seeds}")
        print(f"Test Set: {self.test_size:.0%}")
        print("=" * 70)
    
    def load_data(self):
        """データ読み込みとTrain/Test分割"""
        print("\n📂 データ読み込み中...")
        df = pd.read_csv(self.data_path)
        y_all = df[self.target_col].values
        X_all = df.drop(columns=[self.target_col])
        
        # 発生日時は除外
        if '発生日時' in X_all.columns:
            X_all = X_all.drop(columns=['発生日時'])
        
        # カテゴリカル変数の特定
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
        
        # Train/Test分割
        print(f"\n📊 データ分割 (Train: {1-self.test_size:.0%} / Test: {self.test_size:.0%})")
        self.X, self.X_test, self.y, self.y_test = train_test_split(
            X_all, y_all,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y_all
        )
        self.X = self.X.reset_index(drop=True)
        self.X_test = self.X_test.reset_index(drop=True)
        
        print(f"   Train: 正例 {self.y.sum():,} / {len(self.y):,}")
        print(f"   Test:  正例 {self.y_test.sum():,} / {len(self.y_test):,}")
        print(f"   カテゴリカル変数: {len(self.categorical_cols)}個")
        print(f"   数値変数: {len(self.numerical_cols)}個")
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
    
    def train_lightgbm(self):
        """LightGBM Stage 1 学習"""
        print("\n🌲 LightGBM 学習中...")
        
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
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'n_jobs': -1
        }
        
        self.lgbm_models = []
        
        # LightGBM用: カテゴリ変数をcategory型に変換
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
                
                # カテゴリ型を再適用
                for col in self.categorical_cols:
                    if col in X_train_under.columns:
                        X_train_under[col] = X_train_under[col].astype('category')
                
                model = lgb.LGBMClassifier(**lgb_params, random_state=seed)
                model.fit(
                    X_train_under, y_train_under,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)]
                )
                
                fold_proba += model.predict_proba(X_val)[:, 1] / self.n_seeds
                fold_models.append(model)
                
                del model
                gc.collect()
            
            self.oof_proba_lgbm[val_idx] = fold_proba
            self.lgbm_models.append(fold_models)
        
        oof_auc = roc_auc_score(self.y, self.oof_proba_lgbm)
        print(f"   LightGBM OOF AUC: {oof_auc:.4f}")
        self.lgbm_oof_auc = oof_auc
    
    def train_catboost(self):
        """CatBoost Stage 1 学習"""
        print("\n🐱 CatBoost 学習中...")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.oof_proba_catboost = np.zeros(len(self.y))
        
        # CatBoost用: カテゴリ変数をstr型に変換
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
                    iterations=1000,
                    learning_rate=0.05,
                    depth=8,
                    l2_leaf_reg=3,
                    loss_function='Logloss',
                    eval_metric='AUC',
                    random_seed=seed,
                    verbose=False,
                    early_stopping_rounds=50,
                    task_type='CPU',
                    cat_features=cat_feature_indices
                )
                
                model.fit(
                    X_train_under, y_train_under,
                    eval_set=(X_val, y_val),
                    verbose=False
                )
                
                fold_proba += model.predict_proba(X_val)[:, 1] / self.n_seeds
                fold_models.append(model)
                
                del model
                gc.collect()
            
            self.oof_proba_catboost[val_idx] = fold_proba
            self.catboost_models.append(fold_models)
        
        oof_auc = roc_auc_score(self.y, self.oof_proba_catboost)
        print(f"   CatBoost OOF AUC: {oof_auc:.4f}")
        self.catboost_oof_auc = oof_auc
    
    def find_individual_thresholds(self, safety_recall=0.995):
        """各モデルの個別閾値を決定（安全マージン込み）"""
        print(f"\n🎯 個別閾値決定 (Target Recall: {safety_recall:.1%})...")
        
        self.thresholds = {}
        self.individual_results = {}
        
        for name, proba in [
            ('lgbm', self.oof_proba_lgbm),
            ('catboost', self.oof_proba_catboost)
        ]:
            # Recall >= safety_recall を達成する最大の閾値を探索
            for thresh in np.arange(0.001, 0.5, 0.001):
                pred = (proba >= thresh).astype(int)
                rec = recall_score(self.y, pred)
                if rec < safety_recall:
                    self.thresholds[name] = thresh - 0.001
                    break
            else:
                self.thresholds[name] = 0.001
            
            pred = (proba >= self.thresholds[name]).astype(int)
            rec = recall_score(self.y, pred)
            filter_rate = 1 - pred.mean()
            
            self.individual_results[name] = {
                'threshold': self.thresholds[name],
                'recall': rec,
                'filter_rate': filter_rate,
                'n_candidates': pred.sum()
            }
            
            print(f"   {name.upper()}: 閾値={self.thresholds[name]:.4f}, Recall={rec:.4f}, フィルタリング率={filter_rate:.2%}")
    
    def evaluate_or_ensemble(self):
        """OR条件アンサンブルの評価"""
        print("\n🔗 OR条件アンサンブル評価...")
        
        # 各モデルの個別判定
        pred_lgbm = (self.oof_proba_lgbm >= self.thresholds['lgbm']).astype(int)
        pred_catboost = (self.oof_proba_catboost >= self.thresholds['catboost']).astype(int)
        
        # OR条件: いずれかが1なら1（Stage 2に進める）
        pred_or = np.maximum(pred_lgbm, pred_catboost)
        
        # 評価
        or_recall = recall_score(self.y, pred_or)
        or_precision = precision_score(self.y, pred_or) if pred_or.sum() > 0 else 0
        or_filter_rate = 1 - pred_or.mean()
        
        print(f"\n   📊 OR条件結果:")
        print(f"      Recall: {or_recall:.4f}")
        print(f"      Precision: {or_precision:.4f}")
        print(f"      フィルタリング率: {or_filter_rate:.2%}")
        print(f"      Stage 2 候補数: {pred_or.sum():,} / {len(pred_or):,}")
        
        # 比較サマリ
        print(f"\n   📈 比較:")
        print(f"      LightGBM単独:  フィルタリング率={self.individual_results['lgbm']['filter_rate']:.2%}, Recall={self.individual_results['lgbm']['recall']:.4f}")
        print(f"      CatBoost単独:  フィルタリング率={self.individual_results['catboost']['filter_rate']:.2%}, Recall={self.individual_results['catboost']['recall']:.4f}")
        print(f"      OR条件:        フィルタリング率={or_filter_rate:.2%}, Recall={or_recall:.4f}")
        
        # 見逃し分析: どちらかだけが捕まえた正例
        pos_mask = self.y == 1
        lgbm_only_caught = ((pred_lgbm == 1) & (pred_catboost == 0) & pos_mask).sum()
        catboost_only_caught = ((pred_catboost == 1) & (pred_lgbm == 0) & pos_mask).sum()
        both_caught = ((pred_lgbm == 1) & (pred_catboost == 1) & pos_mask).sum()
        neither_caught = ((pred_lgbm == 0) & (pred_catboost == 0) & pos_mask).sum()
        
        print(f"\n   🔍 正例の検出パターン分析:")
        print(f"      両方で検出:      {both_caught:,}")
        print(f"      LightGBMのみ:    {lgbm_only_caught:,}")
        print(f"      CatBoostのみ:    {catboost_only_caught:,}")
        print(f"      両方見逃し:      {neither_caught:,}")
        
        self.or_results = {
            'or_recall': or_recall,
            'or_precision': or_precision,
            'or_filter_rate': or_filter_rate,
            'lgbm_only_caught': lgbm_only_caught,
            'catboost_only_caught': catboost_only_caught,
            'both_caught': both_caught,
            'neither_caught': neither_caught,
        }
        
        return self.or_results
    
    def evaluate_test_set(self):
        """テストセットでの評価"""
        print("\n📈 テストセット評価...")
        
        # LightGBM: テスト予測
        X_test_lgbm = self.X_test.copy()
        for col in self.categorical_cols:
            if col in X_test_lgbm.columns:
                X_test_lgbm[col] = X_test_lgbm[col].astype('category')
        
        test_proba_lgbm = np.zeros(len(self.y_test))
        for fold_models in self.lgbm_models:
            for model in fold_models:
                test_proba_lgbm += model.predict_proba(X_test_lgbm)[:, 1]
        test_proba_lgbm /= (self.n_folds * self.n_seeds)
        
        # CatBoost: テスト予測
        X_test_cat = self.X_test.copy()
        for col in self.categorical_cols:
            if col in X_test_cat.columns:
                X_test_cat[col] = X_test_cat[col].astype(str)
        
        test_proba_catboost = np.zeros(len(self.y_test))
        for fold_models in self.catboost_models:
            for model in fold_models:
                test_proba_catboost += model.predict_proba(X_test_cat)[:, 1]
        test_proba_catboost /= (self.n_folds * self.n_seeds)
        
        # 個別判定
        pred_lgbm = (test_proba_lgbm >= self.thresholds['lgbm']).astype(int)
        pred_catboost = (test_proba_catboost >= self.thresholds['catboost']).astype(int)
        
        # OR条件
        pred_or = np.maximum(pred_lgbm, pred_catboost)
        
        # 結果
        test_lgbm_recall = recall_score(self.y_test, pred_lgbm)
        test_lgbm_filter = 1 - pred_lgbm.mean()
        
        test_cat_recall = recall_score(self.y_test, pred_catboost)
        test_cat_filter = 1 - pred_catboost.mean()
        
        test_or_recall = recall_score(self.y_test, pred_or)
        test_or_filter = 1 - pred_or.mean()
        
        print(f"\n   📊 テストセット結果:")
        print(f"      LightGBM:  Recall={test_lgbm_recall:.4f}, フィルタリング率={test_lgbm_filter:.2%}")
        print(f"      CatBoost:  Recall={test_cat_recall:.4f}, フィルタリング率={test_cat_filter:.2%}")
        print(f"      OR条件:    Recall={test_or_recall:.4f}, フィルタリング率={test_or_filter:.2%}")
        
        self.test_results = {
            'test_lgbm_recall': test_lgbm_recall,
            'test_lgbm_filter_rate': test_lgbm_filter,
            'test_catboost_recall': test_cat_recall,
            'test_catboost_filter_rate': test_cat_filter,
            'test_or_recall': test_or_recall,
            'test_or_filter_rate': test_or_filter,
        }
        
        return self.test_results
    
    def check_correlation(self):
        """モデル間予測相関係数の確認"""
        corr = np.corrcoef(self.oof_proba_lgbm, self.oof_proba_catboost)[0, 1]
        print(f"\n🔗 モデル間予測相関係数: {corr:.4f}")
        self.model_correlation = corr
        return corr
    
    def plot_feature_importance(self):
        """特徴量重要度の可視化と保存"""
        print("\n📊 特徴量重要度を計算中...")
        
        # LightGBM Importance
        lgbm_imp = np.zeros(len(self.feature_names))
        for fold_models in self.lgbm_models:
            for model in fold_models:
                lgbm_imp += model.feature_importances_
        
        # CatBoost Importance
        cat_imp = np.zeros(len(self.feature_names))
        for fold_models in self.catboost_models:
            for model in fold_models:
                cat_imp += model.get_feature_importance()
        
        # DataFrame化
        df_imp = pd.DataFrame({
            'feature': self.feature_names,
            'lgbm_importance': lgbm_imp / (self.n_folds * self.n_seeds),
            'catboost_importance': cat_imp / (self.n_folds * self.n_seeds)
        })
        
        # 正規化（比較しやすくするため）
        df_imp['lgbm_norm'] = df_imp['lgbm_importance'] / df_imp['lgbm_importance'].sum()
        df_imp['catboost_norm'] = df_imp['catboost_importance'] / df_imp['catboost_importance'].sum()
        
        # 重要度の差分（どちらがより重視しているか）
        df_imp['importance_diff'] = df_imp['lgbm_norm'] - df_imp['catboost_norm']
        
        # ソート（LightGBM重要度順）
        df_imp = df_imp.sort_values('lgbm_importance', ascending=False)
        
        # CSV保存
        imp_path = os.path.join(self.output_dir, "feature_importance.csv")
        df_imp.to_csv(imp_path, index=False)
        print(f"   💾 特徴量重要度を保存: {imp_path}")
        
        # Top 10 表示
        print("\n   📈 Top 10 特徴量 (LightGBM重要度順):")
        for i, row in df_imp.head(10).iterrows():
            print(f"      {row['feature']}: LGBM={row['lgbm_norm']:.4f}, Cat={row['catboost_norm']:.4f}, Diff={row['importance_diff']:+.4f}")
        
        # モデル間で重要度が大きく異なる特徴量
        print("\n   🔍 重要度の差が大きい特徴量 (Top 5):")
        df_diff = df_imp.copy()
        df_diff['abs_diff'] = df_diff['importance_diff'].abs()
        df_diff = df_diff.sort_values('abs_diff', ascending=False)
        for i, row in df_diff.head(5).iterrows():
            if row['importance_diff'] > 0:
                bias = "LightGBM寄り"
            else:
                bias = "CatBoost寄り"
            print(f"      {row['feature']}: {bias} (差={row['importance_diff']:+.4f})")
        
        self.feature_importance_df = df_imp
        return df_imp
    
    def generate_report(self, elapsed_sec: float):
        """実験レポートをMarkdownで出力"""
        report_path = os.path.join(self.output_dir, "experiment_report.md")
        
        report_content = f"""# Stage 1 CatBoost vs LightGBM 比較実験レポート

**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**実行時間**: {elapsed_sec:.1f}秒

## 実験設定

| パラメータ | 値 |
|-----------|-----|
| Under-sampling | 1:{int(self.undersample_ratio)} |
| Seeds | {self.n_seeds} |
| Folds | {self.n_folds} |
| Target Recall | {self.target_recall:.0%} |
| Test Set | {self.test_size:.0%} |

## 個別モデル結果 (CV OOF)

| モデル | OOF AUC | 閾値 | Recall | フィルタリング率 |
|--------|---------|------|--------|-----------------|
| LightGBM | {self.lgbm_oof_auc:.4f} | {self.thresholds['lgbm']:.4f} | {self.individual_results['lgbm']['recall']:.4f} | {self.individual_results['lgbm']['filter_rate']:.2%} |
| CatBoost | {self.catboost_oof_auc:.4f} | {self.thresholds['catboost']:.4f} | {self.individual_results['catboost']['recall']:.4f} | {self.individual_results['catboost']['filter_rate']:.2%} |

## OR条件アンサンブル (CV OOF)

| 指標 | 値 |
|------|-----|
| Recall | {self.or_results['or_recall']:.4f} |
| Precision | {self.or_results['or_precision']:.4f} |
| フィルタリング率 | {self.or_results['or_filter_rate']:.2%} |

### 正例検出パターン分析

| パターン | 件数 |
|----------|------|
| 両方で検出 | {self.or_results['both_caught']:,} |
| LightGBMのみ | {self.or_results['lgbm_only_caught']:,} |
| CatBoostのみ | {self.or_results['catboost_only_caught']:,} |
| 両方見逃し | {self.or_results['neither_caught']:,} |

## テストセット結果

| モデル | Recall | フィルタリング率 |
|--------|--------|-----------------|
| LightGBM | {self.test_results['test_lgbm_recall']:.4f} | {self.test_results['test_lgbm_filter_rate']:.2%} |
| CatBoost | {self.test_results['test_catboost_recall']:.4f} | {self.test_results['test_catboost_filter_rate']:.2%} |
| OR条件 | {self.test_results['test_or_recall']:.4f} | {self.test_results['test_or_filter_rate']:.2%} |

## モデル間予測相関

| 指標 | 値 |
|------|-----|
| 予測確率の相関係数 | {self.model_correlation:.4f} |

> 相関係数が低いほど（例: 0.8-0.9）、モデルが「異なる視点」で予測していることを示し、OR条件アンサンブルの相補効果が期待できます。

## 考察

- **AUC比較**: LightGBM ({self.lgbm_oof_auc:.4f}) vs CatBoost ({self.catboost_oof_auc:.4f})
- **モデル相関**: {self.model_correlation:.4f} → {'相補性が高い（異なる視点）' if self.model_correlation < 0.95 else '類似した予測傾向'}
- **相補性**: LightGBMのみで検出={self.or_results['lgbm_only_caught']}件, CatBoostのみで検出={self.or_results['catboost_only_caught']}件
- **OR条件の効果**: 
  - Recall {self.or_results['or_recall']:.4f} を維持
  - フィルタリング効率の変化をCV/Testで確認

## 特徴量重要度

詳細は `feature_importance.csv` を参照してください。

モデル間で重視する特徴量が異なる場合、それがOR条件アンサンブルの有効性を裏付ける証拠となります。
"""
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n   📄 レポート出力: {report_path}")
        return report_path
    
    def run(self):
        """パイプライン実行"""
        start = datetime.now()
        
        self.load_data()
        self.train_lightgbm()
        self.train_catboost()
        self.find_individual_thresholds(safety_recall=0.995)
        self.check_correlation()
        self.evaluate_or_ensemble()
        self.evaluate_test_set()
        self.plot_feature_importance()
        
        elapsed_sec = (datetime.now() - start).total_seconds()
        
        # 結果CSV保存
        results = {
            'lgbm_oof_auc': self.lgbm_oof_auc,
            'catboost_oof_auc': self.catboost_oof_auc,
            'lgbm_threshold': self.thresholds['lgbm'],
            'catboost_threshold': self.thresholds['catboost'],
            **{f'lgbm_{k}': v for k, v in self.individual_results['lgbm'].items()},
            **{f'catboost_{k}': v for k, v in self.individual_results['catboost'].items()},
            **self.or_results,
            **self.test_results,
            'model_correlation': self.model_correlation,
            'elapsed_sec': elapsed_sec
        }
        pd.DataFrame([results]).to_csv(os.path.join(self.output_dir, "results.csv"), index=False)
        
        self.generate_report(elapsed_sec)
        
        print("\n" + "=" * 70)
        print("✅ 完了！")
        print(f"   結果CSV: {self.output_dir}/results.csv")
        print(f"   レポートMD: {self.output_dir}/experiment_report.md")
        print("=" * 70)
        
        return results


if __name__ == "__main__":
    pipeline = Stage1CatBoostComparisonPipeline()
    pipeline.run()
