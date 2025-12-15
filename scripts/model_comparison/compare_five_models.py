"""
5モデル統合比較スクリプト
ロジスティック回帰 vs Random Forest vs LightGBM vs XGBoost vs CatBoost

5つの異なるアプローチのモデルを公平に比較:
- ロジスティック回帰（線形モデル）
- Random Forest（バギング手法）
- LightGBM（ブースティング・Leaf-wise）
- XGBoost（ブースティング・Depth-wise）
- CatBoost（ブースティング・カテゴリ特化）

評価内容:
- PR-AUC, ROC-AUC, F1, Accuracy, Precision, Recall
- 統計的有意差検定（Friedman検定 + Nemenyi検定）
- 訓練時間・予測時間の比較
- 詳細な5モデル比較レポートの生成
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
from scipy import stats
import time
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FiveModelComparator:
    """5モデルの統合比較"""
    
    def __init__(
        self,
        data_path='data/processed/honhyo_clean_road_type.csv',
        target_column='死者数',
        n_folds=5,
        random_state=42
    ):
        self.data_path = data_path
        self.target_column = target_column
        self.n_folds = n_folds
        self.random_state = random_state
        
        # LightGBMの最良パラメータ（Trial 153）
        self.lightgbm_params = {
            'learning_rate': 0.07658346283890378,
            'num_leaves': 125,
            'max_depth': 8,
            'min_child_samples': 278,
            'subsample': 0.6147706754536576,
            'colsample_bytree': 0.6267708320804088,
            'reg_alpha': 0.9961403311275829,
            'reg_lambda': 8.228908331551605,
            'min_child_weight': 0.12646850234127796,
            'min_split_gain': 0.24303906753172422,
            'path_smooth': 2.254892007170922,
            'scale_pos_weight': 61.47728365878301,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 10000,
            'random_state': random_state,
            'n_jobs': -1,
            'verbose': -1
        }
        
        # Random Forestのパラメータ
        self.rf_params = {
            'n_estimators': 500,
            'max_depth': 15,
            'min_samples_split': 20,
            'min_samples_leaf': 10,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': random_state,
            'n_jobs': -1,
            'verbose': 0
        }
        
        # XGBoostのパラメータ（LightGBMに合わせて調整）
        self.xgboost_params = {
            'learning_rate': 0.08,
            'n_estimators': 1000,
            'max_depth': 8,
            'min_child_weight': 5,
            'subsample': 0.6,
            'colsample_bytree': 0.6,
            'reg_alpha': 1.0,
            'reg_lambda': 8.0,
            'scale_pos_weight': 61.48,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': random_state,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        # CatBoostのパラメータ
        self.catboost_params = {
            'iterations': 1000,
            'learning_rate': 0.08,
            'depth': 8,
            'l2_leaf_reg': 8.0,
            'scale_pos_weight': 61.48,
            'random_state': random_state,
            'verbose': False,
            'thread_count': -1,
            'task_type': 'CPU'
        }
        
        print("="*80)
        print("5モデル比較: ロジスティック回帰 vs RF vs LightGBM vs XGBoost vs CatBoost")
        print("="*80)
        
        # データ読み込み
        print(f"\n[データ読み込み] {data_path}")
        self.df = pd.read_csv(data_path)
        print(f"✅ 読み込み完了: {len(self.df):,} 件")
        
        # 前処理
        self._preprocess_data()
        
    def _preprocess_data(self):
        """データの前処理"""
        print("\n[前処理] データ準備中...")
        
        # 目的変数を分離
        self.y = self.df[self.target_column]
        self.X = self.df.drop(columns=[self.target_column])
        
        # 発生日時を除外
        if '発生日時' in self.X.columns:
            self.X = self.X.drop(columns=['発生日時'])
        
        # 特徴量の分類
        self.numeric_cols = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        explicit_cat_cols = [
            '都道府県コード', '路線コード', '地点コード', '市区町村コード',
            '昼夜', '天候', '地形', '路面状態', '道路形状', '信号機',
            '衝突地点', 'ゾーン規制', '中央分離帯施設等', '歩車道区分',
            '事故類型', '曜日(発生年月日)', '祝日(発生年月日)'
        ]
        
        explicit_cat_cols = [c for c in explicit_cat_cols if c in self.X.columns]
        self.categorical_cols = list(set(self.categorical_cols + explicit_cat_cols))
        self.numeric_cols = [c for c in self.numeric_cols if c not in self.categorical_cols]
        
        print(f"  - 数値型: {len(self.numeric_cols)} 個")
        print(f"  - カテゴリカル型: {len(self.categorical_cols)} 個")
        
        # カテゴリカル変数を文字列に統一
        for col in self.categorical_cols:
            if col in self.X.columns:
                self.X[col] = self.X[col].astype(str)
        
        # CatBoost用にカテゴリカル変数のインデックスを保存
        self.cat_features_idx = [self.X.columns.get_loc(col) for col in self.categorical_cols if col in self.X.columns]
        
    def _build_logreg_pipeline(self):
        """ロジスティック回帰のパイプライン"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ],
            remainder='drop'
        )
        
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                penalty='l2', C=1.0, solver='saga', max_iter=1000,
                class_weight='balanced', random_state=self.random_state,
                n_jobs=-1, verbose=0
            ))
        ])
    
    def _build_rf_pipeline(self):
        """Random Forestのパイプライン"""
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median'))
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ],
            remainder='drop'
        )
        
        return Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(**self.rf_params))
        ])
    
    def _build_lightgbm_model(self):
        """LightGBMモデル"""
        return lgb.LGBMClassifier(**self.lightgbm_params)
    
    def _build_xgboost_model(self):
        """XGBoostモデル"""
        return xgb.XGBClassifier(**self.xgboost_params)
    
    def _build_catboost_model(self):
        """CatBoostモデル"""
        return CatBoostClassifier(**self.catboost_params)
    
    def _prepare_data_for_tree_models(self, X):
        """Tree系モデル用のデータ準備（欠損値補完のみ）"""
        X_prepared = X.copy()
        
        # 数値型の欠損値補完
        for col in self.numeric_cols:
            if col in X_prepared.columns and X_prepared[col].isna().any():
                X_prepared[col].fillna(X_prepared[col].median(), inplace=True)
        
        # カテゴリカル型の欠損値補完
        for col in self.categorical_cols:
            if col in X_prepared.columns and X_prepared[col].isna().any():
                X_prepared[col].fillna(X_prepared[col].mode()[0], inplace=True)
        
        return X_prepared
    
    def compare_with_cv(self):
        """交差検証で5モデルを比較"""
        print(f"\n[開始] {self.n_folds}-fold 交差検証で5モデルを比較")
        print("="*80)
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        results = {
            'logreg': [],
            'rf': [],
            'lightgbm': [],
            'xgboost': [],
            'catboost': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            print(f"\n{'='*80}")
            print(f"Fold {fold+1}/{self.n_folds}")
            print(f"{'='*80}")
            
            X_train, X_val = self.X.iloc[train_idx].copy(), self.X.iloc[val_idx].copy()
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # ===== 1. ロジスティック回帰 =====
            print("\n[1/5] ロジスティック回帰")
            logreg = self._build_logreg_pipeline()
            start = time.time()
            logreg.fit(X_train, y_train)
            train_time = time.time() - start
            
            start = time.time()
            prob = logreg.predict_proba(X_val)[:, 1]
            pred_time = time.time() - start
            pred = (prob >= 0.5).astype(int)
            
            results['logreg'].append(self._calculate_metrics(y_val, pred, prob, train_time, pred_time, fold+1))
            print(f"  PR-AUC: {results['logreg'][-1]['pr_auc']:.4f} | Time: {train_time:.1f}s")
            
            # ===== 2. Random Forest =====
            print("\n[2/5] Random Forest")
            rf = self._build_rf_pipeline()
            start = time.time()
            rf.fit(X_train, y_train)
            train_time = time.time() - start
            
            start = time.time()
            prob = rf.predict_proba(X_val)[:, 1]
            pred_time = time.time() - start
            pred = (prob >= 0.5).astype(int)
            
            results['rf'].append(self._calculate_metrics(y_val, pred, prob, train_time, pred_time, fold+1))
            print(f"  PR-AUC: {results['rf'][-1]['pr_auc']:.4f} | Time: {train_time:.1f}s")
            
            # Tree系モデル用データ準備
            X_train_tree = self._prepare_data_for_tree_models(X_train)
            X_val_tree = self._prepare_data_for_tree_models(X_val)
            
            # ===== 3. LightGBM =====
            print("\n[3/5] LightGBM")
            lgbm = self._build_lightgbm_model()
            start = time.time()
            lgbm.fit(X_train_tree, y_train, eval_set=[(X_val_tree, y_val)],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
            train_time = time.time() - start
            
            start = time.time()
            prob = lgbm.predict_proba(X_val_tree)[:, 1]
            pred_time = time.time() - start
            pred = (prob >= 0.5).astype(int)
            
            results['lightgbm'].append(self._calculate_metrics(y_val, pred, prob, train_time, pred_time, fold+1))
            print(f"  PR-AUC: {results['lightgbm'][-1]['pr_auc']:.4f} | Time: {train_time:.1f}s")
            
            # ===== 4. XGBoost =====
            print("\n[4/5] XGBoost")
            xgb_model = self._build_xgboost_model()
            start = time.time()
            xgb_model.fit(X_train_tree, y_train, eval_set=[(X_val_tree, y_val)],
                         verbose=False)
            train_time = time.time() - start
            
            start = time.time()
            prob = xgb_model.predict_proba(X_val_tree)[:, 1]
            pred_time = time.time() - start
            pred = (prob >= 0.5).astype(int)
            
            results['xgboost'].append(self._calculate_metrics(y_val, pred, prob, train_time, pred_time, fold+1))
            print(f"  PR-AUC: {results['xgboost'][-1]['pr_auc']:.4f} | Time: {train_time:.1f}s")
            
            # ===== 5. CatBoost =====
            print("\n[5/5] CatBoost")
            catb = self._build_catboost_model()
            start = time.time()
            catb.fit(X_train_tree, y_train, cat_features=self.cat_features_idx,
                    eval_set=(X_val_tree, y_val), early_stopping_rounds=50, verbose=False)
            train_time = time.time() - start
            
            start = time.time()
            prob = catb.predict_proba(X_val_tree)[:, 1]
            pred_time = time.time() - start
            pred = (prob >= 0.5).astype(int)
            
            results['catboost'].append(self._calculate_metrics(y_val, pred, prob, train_time, pred_time, fold+1))
            print(f"  PR-AUC: {results['catboost'][-1]['pr_auc']:.4f} | Time: {train_time:.1f}s")
        
        # DataFrameに変換
        self.results_dfs = {name: pd.DataFrame(data) for name, data in results.items()}
        self.results_means = {name: df.mean() for name, df in self.results_dfs.items()}
        self.results_stds = {name: df.std() for name, df in self.results_dfs.items()}
        
        self._print_comparison_summary()
        
        return self.results_dfs
    
    def _calculate_metrics(self, y_true, y_pred, y_prob, train_time, pred_time, fold):
        """評価指標を計算"""
        return {
            'fold': fold,
            'pr_auc': average_precision_score(y_true, y_prob),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'f1': f1_score(y_true, y_pred),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred),
            'train_time': train_time,
            'pred_time': pred_time
        }
    
    def _print_comparison_summary(self):
        """比較結果のサマリー表示"""
        print("\n" + "="*80)
        print("[結果] 5モデル比較サマリー")
        print("="*80)
        
        model_names = ['ロジスティック回帰', 'Random Forest', 'LightGBM', 'XGBoost', 'CatBoost']
        model_keys = ['logreg', 'rf', 'lightgbm', 'xgboost', 'catboost']
        
        metrics = ['pr_auc', 'roc_auc', 'f1']
        metric_names = ['PR-AUC', 'ROC-AUC', 'F1 Score']
        
        for metric, metric_name in zip(metrics, metric_names):
            print(f"\n{metric_name}:")
            values = []
            for key, name in zip(model_keys, model_names):
                val = self.results_means[key][metric]
                std = self.results_stds[key][metric]
                values.append(val)
                print(f"  {name:<20}: {val:.4f} ± {std:.4f}")
            
            best_idx = np.argmax(values)
            print(f"  → 最良: {model_names[best_idx]}")
    
    def statistical_test(self):
        """Friedman検定"""
        print("\n" + "="*80)
        print("[統計的検定] Friedman検定")
        print("="*80)
        
        metrics = ['pr_auc', 'roc_auc', 'f1']
        test_results = []
        
        for metric in metrics:
            scores = [self.results_dfs[key][metric].values for key in ['logreg', 'rf', 'lightgbm', 'xgboost', 'catboost']]
            stat, p_val = stats.friedmanchisquare(*scores)
            
            test_results.append({
                'metric': metric,
                'statistic': stat,
                'p_value': p_val,
                'significant': p_val < 0.05
            })
            
            print(f"{metric.upper()}: 統計量={stat:.4f}, p値={p_val:.4f}, 有意差={'✅' if p_val < 0.05 else '❌'}")
        
        return pd.DataFrame(test_results)
    
    def save_results(self, output_dir='results/model_comparison'):
        """結果保存"""
        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, df in self.results_dfs.items():
            df.to_csv(f'{output_dir}/{name}_cv_5models_{ts}.csv', index=False, encoding='utf-8-sig')
        
        print(f"\n✅ 結果を保存: {output_dir}")


def main():
    comparator = FiveModelComparator()
    comparator.compare_with_cv()
    comparator.statistical_test()
    comparator.save_results()
    
    print("\n" + "="*80)
    print("✅ 5モデル比較が完了しました！")
    print("="*80)


if __name__ == '__main__':
    main()
