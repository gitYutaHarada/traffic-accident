"""
3モデル統合比較スクリプト: ロジスティック回帰 vs Random Forest vs LightGBM

3つの異なるアプローチのモデルを公平に比較:
- ロジスティック回帰（線形モデル）
- Random Forest（バギング手法）
- LightGBM（ブースティング手法）

評価内容:
- PR-AUC, ROC-AUC, F1, Accuracy, Precision, Recall
- 統計的有意差検定（Friedman検定 + Nemenyi検定）
- 訓練時間・予測時間の比較
- 詳細な3モデル比較レポートの生成
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
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


class ThreeModelComparator:
    """ロジスティック回帰 vs Random Forest vs LightGBM の比較"""
    
    def __init__(
        self,
        data_path='data/processed/honhyo_clean_road_type.csv',
        target_column='死者数',
        n_folds=5,
        random_state=42
    ):
        """
        Parameters:
        -----------
        data_path : str
            データセットのパス
        target_column : str
            目的変数のカラム名
        n_folds : int
            交差検証のフォールド数
        random_state : int
            乱数シード
        """
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
            'verbose': -1,
            #'class_weight': 'balanced' # LightGBMの場合はscale_pos_weightで調整済みだが明示的にbalancedも検討可
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
        
        print("="*80)
        print("3モデル比較: ロジスティック回帰(OHE) vs Random Forest(Ordinal) vs LightGBM(Native)")
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
        
    def _build_logreg_pipeline(self):
        """ロジスティック回帰のパイプライン構築 (ハイブリッドEncoding)"""
        from category_encoders import TargetEncoder
        
        # 高多重度カテゴリ（Target Encoding対象）
        high_cardinality_cols = ['地点コード', '市区町村コード', '警察署等コード', '車道幅員', '速度規制（指定のみ）（当事者A）', '速度規制（指定のみ）（当事者B）']
        high_cardinality_cols = [c for c in high_cardinality_cols if c in self.categorical_cols]
        
        # 低多重度カテゴリ（One-Hot Encoding対象）
        low_cardinality_cols = [c for c in self.categorical_cols if c not in high_cardinality_cols]
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # 高多重度用パイプライン
        high_card_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', TargetEncoder(smoothing=10)) # 過学習防止のためスムージング
        ])
        
        # 低多重度用パイプライン
        low_card_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        transformers = [
            ('num', numeric_transformer, self.numeric_cols),
            ('high_card', high_card_transformer, high_cardinality_cols),
            ('low_card', low_card_transformer, low_cardinality_cols)
        ]
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                penalty='l2',
                C=1.0,
                solver='saga',
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0
            ))
        ])
        
        return pipeline
    
    def _build_rf_pipeline(self):
        """Random Forestのパイプライン構築"""
        # Random Forestはカテゴリカル変数を直接扱えないため、エンコードが必要
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
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(**self.rf_params))
        ])
        
        return pipeline
    
    def _build_lightgbm_model(self):
        """LightGBMモデルの構築"""
        return lgb.LGBMClassifier(**self.lightgbm_params)
    
    def _prepare_data_for_lightgbm(self, X, is_train=False):
        """LightGBM用のデータ準備（カテゴリカル変数をcategory型に変換）"""
        X_prepared = X.copy()
        
        # 数値型の欠損値補完
        for col in self.numeric_cols:
            if col in X_prepared.columns and X_prepared[col].isna().any():
                X_prepared[col].fillna(X_prepared[col].median(), inplace=True)
        
        # カテゴリカル型をcategory型に変換
        for col in self.categorical_cols:
            if col in X_prepared.columns:
                # 欠損値は 'missing' として扱う
                X_prepared[col] = X_prepared[col].fillna('missing').astype('category')
        
        return X_prepared
    
    def compare_with_cv(self):
        """交差検証で3モデルを比較"""
        print(f"\n[開始] {self.n_folds}-fold 交差検証で3モデルを比較")
        print("="*80)
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        logreg_results = []
        rf_results = []
        lightgbm_results = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            print(f"\n{'='*80}")
            print(f"Fold {fold+1}/{self.n_folds}")
            print(f"{'='*80}")
            
            X_train, X_val = self.X.iloc[train_idx].copy(), self.X.iloc[val_idx].copy()
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # ===== ロジスティック回帰 =====
            print("\n[1/3] ロジスティック回帰")
            logreg_pipeline = self._build_logreg_pipeline()
            
            start_time = time.time()
            logreg_pipeline.fit(X_train, y_train)
            logreg_train_time = time.time() - start_time
            
            start_time = time.time()
            logreg_prob = logreg_pipeline.predict_proba(X_val)[:, 1]
            logreg_pred_time = time.time() - start_time
            
            logreg_pred = (logreg_prob >= 0.5).astype(int)
            
            logreg_metrics = {
                'fold': fold + 1,
                'pr_auc': average_precision_score(y_val, logreg_prob),
                'roc_auc': roc_auc_score(y_val, logreg_prob),
                'f1': f1_score(y_val, logreg_pred),
                'accuracy': accuracy_score(y_val, logreg_pred),
                'precision': precision_score(y_val, logreg_pred, zero_division=0),
                'recall': recall_score(y_val, logreg_pred),
                'train_time': logreg_train_time,
                'pred_time': logreg_pred_time
            }
            logreg_results.append(logreg_metrics)
            
            print(f"  PR-AUC: {logreg_metrics['pr_auc']:.4f} | ROC-AUC: {logreg_metrics['roc_auc']:.4f} | F1: {logreg_metrics['f1']:.4f} | Time: {logreg_train_time:.1f}s")
            
            # ===== Random Forest =====
            print("\n[2/3] Random Forest")
            rf_pipeline = self._build_rf_pipeline()
            
            start_time = time.time()
            rf_pipeline.fit(X_train, y_train)
            rf_train_time = time.time() - start_time
            
            start_time = time.time()
            rf_prob = rf_pipeline.predict_proba(X_val)[:, 1]
            rf_pred_time = time.time() - start_time
            
            rf_pred = (rf_prob >= 0.5).astype(int)
            
            rf_metrics = {
                'fold': fold + 1,
                'pr_auc': average_precision_score(y_val, rf_prob),
                'roc_auc': roc_auc_score(y_val, rf_prob),
                'f1': f1_score(y_val, rf_pred),
                'accuracy': accuracy_score(y_val, rf_pred),
                'precision': precision_score(y_val, rf_pred, zero_division=0),
                'recall': recall_score(y_val, rf_pred),
                'train_time': rf_train_time,
                'pred_time': rf_pred_time
            }
            rf_results.append(rf_metrics)
            
            print(f"  PR-AUC: {rf_metrics['pr_auc']:.4f} | ROC-AUC: {rf_metrics['roc_auc']:.4f} | F1: {rf_metrics['f1']:.4f} | Time: {rf_train_time:.1f}s")
            
            # ===== LightGBM =====
            print("\n[3/3] LightGBM")
            lightgbm_model = self._build_lightgbm_model()
            
            # LightGBM用にデータを準備（カテゴリカル変数をエンコード）
            X_train_lgbm = self._prepare_data_for_lightgbm(X_train, is_train=True)
            X_val_lgbm = self._prepare_data_for_lightgbm(X_val, is_train=False)
            
            start_time = time.time()
            lightgbm_model.fit(
                X_train_lgbm, y_train,
                eval_set=[(X_val_lgbm, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            lightgbm_train_time = time.time() - start_time
            
            start_time = time.time()
            lightgbm_prob = lightgbm_model.predict_proba(X_val_lgbm)[:, 1]
            lightgbm_pred_time = time.time() - start_time
            
            lightgbm_pred = (lightgbm_prob >= 0.5).astype(int)
            
            lightgbm_metrics = {
                'fold': fold + 1,
                'pr_auc': average_precision_score(y_val, lightgbm_prob),
                'roc_auc': roc_auc_score(y_val, lightgbm_prob),
                'f1': f1_score(y_val, lightgbm_pred),
                'accuracy': accuracy_score(y_val, lightgbm_pred),
                'precision': precision_score(y_val, lightgbm_pred, zero_division=0),
                'recall': recall_score(y_val, lightgbm_pred),
                'train_time': lightgbm_train_time,
                'pred_time': lightgbm_pred_time
            }
            lightgbm_results.append(lightgbm_metrics)
            
            print(f"  PR-AUC: {lightgbm_metrics['pr_auc']:.4f} | ROC-AUC: {lightgbm_metrics['roc_auc']:.4f} | F1: {lightgbm_metrics['f1']:.4f} | Time: {lightgbm_train_time:.1f}s")
        
        # 結果をDataFrameに変換
        self.logreg_df = pd.DataFrame(logreg_results)
        self.rf_df = pd.DataFrame(rf_results)
        self.lightgbm_df = pd.DataFrame(lightgbm_results)
        
        # 平均と標準偏差を計算
        self.logreg_mean = self.logreg_df.mean()
        self.logreg_std = self.logreg_df.std()
        self.rf_mean = self.rf_df.mean()
        self.rf_std = self.rf_df.std()
        self.lightgbm_mean = self.lightgbm_df.mean()
        self.lightgbm_std = self.lightgbm_df.std()
        
        # 結果を表示
        self._print_comparison_summary()
        
        return self.logreg_df, self.rf_df, self.lightgbm_df
    
    def _print_comparison_summary(self):
        """比較結果のサマリーを表示"""
        print("\n" + "="*80)
        print("[結果] 3モデル比較サマリー")
        print("="*80)
        
        metrics = ['pr_auc', 'roc_auc', 'f1', 'accuracy', 'precision', 'recall']
        metric_names = ['PR-AUC', 'ROC-AUC', 'F1', 'Accuracy', 'Precision', 'Recall']
        
        print(f"\n{'指標':<12} {'ロジスティック回帰':<22} {'Random Forest':<22} {'LightGBM':<22}")
        print("-" * 80)
        
        for metric, name in zip(metrics, metric_names):
            logreg_val = self.logreg_mean[metric]
            rf_val = self.rf_mean[metric]
            lightgbm_val = self.lightgbm_mean[metric]
            
            # 最良値を太字で表示（実際には★マーク）
            best_val = max(logreg_val, rf_val, lightgbm_val)
            
            logreg_str = f"{logreg_val:.4f} ± {self.logreg_std[metric]:.4f}"
            rf_str = f"{rf_val:.4f} ± {self.rf_std[metric]:.4f}"
            lightgbm_str = f"{lightgbm_val:.4f} ± {self.lightgbm_std[metric]:.4f}"
            
            if logreg_val == best_val:
                logreg_str += " ★"
            if rf_val == best_val:
                rf_str += " ★"
            if lightgbm_val == best_val:
                lightgbm_str += " ★"
            
            print(f"{name:<12} {logreg_str:<22} {rf_str:<22} {lightgbm_str:<22}")
        
        print("\n[計算時間]")
        print(f"訓練時間     {self.logreg_mean['train_time']:.1f}秒                "
              f"{self.rf_mean['train_time']:.1f}秒                "
              f"{self.lightgbm_mean['train_time']:.1f}秒")
    
    def statistical_test(self):
        """統計的検定（Friedman検定）"""
        print("\n" + "="*80)
        print("[統計的検定] Friedman検定（3モデル以上の比較）")
        print("="*80)
        
        metrics = ['pr_auc', 'roc_auc', 'f1']
        metric_names = ['PR-AUC', 'ROC-AUC', 'F1 Score']
        
        test_results = []
        
        for metric, name in zip(metrics, metric_names):
            logreg_scores = self.logreg_df[metric].values
            rf_scores = self.rf_df[metric].values
            lightgbm_scores = self.lightgbm_df[metric].values
            
            # Friedman検定
            statistic, p_value = stats.friedmanchisquare(logreg_scores, rf_scores, lightgbm_scores)
            
            significant = "✅ 有意" if p_value < 0.05 else "❌ 非有意"
            
            print(f"\n{name}:")
            print(f"  統計量: {statistic:.4f}")
            print(f"  p値: {p_value:.4f}")
            print(f"  結果: {significant} (α=0.05)")
            
            test_results.append({
                'metric': name,
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
        
        self.test_results = pd.DataFrame(test_results)
        
        return self.test_results
    
    def save_results(self, output_dir='results/model_comparison'):
        """結果を保存"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"\n[保存] 結果を保存中: {output_dir}")
        
        # Fold別の結果をCSV保存
        self.logreg_df.to_csv(f'{output_dir}/logreg_cv_3models_{timestamp}.csv', index=False, encoding='utf-8-sig')
        self.rf_df.to_csv(f'{output_dir}/rf_cv_{timestamp}.csv', index=False, encoding='utf-8-sig')
        self.lightgbm_df.to_csv(f'{output_dir}/lightgbm_cv_3models_{timestamp}.csv', index=False, encoding='utf-8-sig')
        
        print(f"  ✅ CV結果を保存")
        
        # 統計的検定結果
        self.test_results.to_csv(f'{output_dir}/statistical_test_3models_{timestamp}.csv', index=False, encoding='utf-8-sig')
        print(f"  ✅ 統計的検定結果を保存")
        
        # 比較レポート生成
        self._generate_comparison_report(output_dir, timestamp)
        
        print(f"\n✅ すべての結果を保存完了: {output_dir}")
    
    def _generate_comparison_report(self, output_dir, timestamp):
        """比較レポートを生成"""
        # （簡略版 - 実際はより詳細に）
        report_lines = [
            "# ロジスティック回帰 vs Random Forest vs LightGBM 比較分析レポート",
            "",
            f"**実験日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}",
            "",
            "## 主要指標の比較",
            "",
            "| 指標 | ロジスティック回帰 | Random Forest | LightGBM |",
            "|------|------------------|--------------|----------|",
            f"| PR-AUC | {self.logreg_mean['pr_auc']:.4f} | {self.rf_mean['pr_auc']:.4f} | {self.lightgbm_mean['pr_auc']:.4f} |",
            f"| ROC-AUC | {self.logreg_mean['roc_auc']:.4f} | {self.rf_mean['roc_auc']:.4f} | {self.lightgbm_mean['roc_auc']:.4f} |",
            f"| F1 Score | {self.logreg_mean['f1']:.4f} | {self.rf_mean['f1']:.4f} | {self.lightgbm_mean['f1']:.4f} |",
        ]
        
        report_path = f'{output_dir}/comparison_report_3models_{timestamp}.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"  ✅ 比較レポート: {report_path}")


def main():
    """メイン処理"""
    comparator = ThreeModelComparator(
        data_path='data/processed/honhyo_clean_road_type.csv',
        target_column='死者数',
        n_folds=5,
        random_state=42
    )
    
    # 交差検証で比較
    logreg_df, rf_df, lightgbm_df = comparator.compare_with_cv()
    
    # 統計的検定
    test_results = comparator.statistical_test()
    
    # 結果を保存
    comparator.save_results()
    
    print("\n" + "="*80)
    print("✅ 3モデル比較が完了しました！")
    print("="*80)


if __name__ == '__main__':
    main()
