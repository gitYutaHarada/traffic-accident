"""
LightGBMによる交互作用特徴量の重要度評価スクリプト

各交互作用特徴量を既存の特徴量に追加し、LightGBMで5-fold CVを実行。
PR-AUCの向上度合いで重要度をランキング化します。

処理フロー:
1. ベースラインモデル（交互作用特徴量なし）のPR-AUCを測定
2. 各交互作用特徴量を1つずつ追加してPR-AUCを測定
3. PR-AUCの向上度（delta PR-AUC）でランキング
4. 結果をCSVとMarkdownレポートで出力
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, 
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import pickle
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class InteractionFeatureEvaluator:
    """LightGBMで交互作用特徴量の重要度を評価するクラス"""
    
    def __init__(
        self, 
        data_path, 
        interaction_metadata_path,
        interaction_dir,
        target_column='死者数',
        n_folds=5,
        random_state=42
    ):
        """
        Parameters:
        -----------
        data_path : str
            元データのパス
        interaction_metadata_path : str
            交互作用特徴量のメタデータCSVパス
        interaction_dir : str
            交互作用特徴量のpickleファイルが保存されているディレクトリ
        target_column : str
            目的変数のカラム名
        n_folds : int
            交差検証のフォールド数
        random_state : int
            乱数シード
        """
        self.data_path = data_path
        self.interaction_metadata_path = interaction_metadata_path
        self.interaction_dir = Path(interaction_dir)
        self.target_column = target_column
        self.n_folds = n_folds
        self.random_state = random_state
        
        # 最良のLightGBMパラメータ（チューニング結果から）
        self.best_params = {
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
        
        # データ読み込み
        print(f"データを読み込み中: {data_path}")
        self.df = pd.read_csv(data_path)
        
        # 目的変数を分離
        self.y = self.df[target_column]
        self.X = self.df.drop(columns=[target_column])
        
        # 発生日時は除外（日時型はそのままでは使えない）
        if '発生日時' in self.X.columns:
            self.X = self.X.drop(columns=['発生日時'])
        
        print(f"データ形状: X={self.X.shape}, y={self.y.shape}")
        print(f"クラス不均衡比: {(self.y == 0).sum() / (self.y == 1).sum():.2f}")
        
        # メタデータ読み込み
        print(f"\nメタデータを読み込み中: {interaction_metadata_path}")
        self.metadata = pd.read_csv(interaction_metadata_path)
        print(f"交互作用特徴量数: {len(self.metadata)}")
        
        # 特徴量の分類
        self.numeric_cols = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 明示的なカテゴリカル変数の指定（誤判定防止）
        explicit_cat_cols = [
            '都道府県コード', '路線コード', '地点コード', '市区町村コード',
            '昼夜', '天候', '地形', '路面状態', '道路形状', '信号機',
            '衝突地点', 'ゾーン規制', '中央分離帯施設等', '歩車道区分',
            '事故類型', '曜日(発生年月日)', '祝日(発生年月日)'
        ]
        explicit_cat_cols = [c for c in explicit_cat_cols if c in self.X.columns]
        self.categorical_cols = list(set(self.categorical_cols + explicit_cat_cols))
        self.numeric_cols = [c for c in self.numeric_cols if c not in self.categorical_cols]
        
        # カテゴリカル変数を文字列に統一
        for col in self.categorical_cols:
            if col in self.X.columns:
                self.X[col] = self.X[col].astype(str)
                
        print(f"特徴量: 数値 {len(self.numeric_cols)}個, カテゴリ {len(self.categorical_cols)}個")
        
    def evaluate_baseline(self):
        """
        ベースラインモデル（交互作用特徴量なし）の性能を評価
        
        Returns:
        --------
        dict
            各評価指標の平均値
        """
        print("\n" + "="*60)
        print("ベースラインモデルの評価（交互作用特徴量なし）")
        print("="*60)
        
        cv_scores = self._cross_validate(self.X, self.y)
        
        print(f"ベースライン PR-AUC: {cv_scores['pr_auc']:.6f}")
        print(f"ベースライン ROC-AUC: {cv_scores['roc_auc']:.6f}")
        print(f"ベースライン F1: {cv_scores['f1']:.6f}")
        
        return cv_scores
    
    
    def _prepare_data_for_lightgbm(self, X, is_train=False, encoder=None):
        """
        LightGBM用のデータ準備（カテゴリカル変数をエンコード）
        動的にカテゴリカル列を判定して処理
        """
        X_prepared = X.copy()
        
        # 数値型とカテゴリカル型を動的に判定
        numeric_cols = X_prepared.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X_prepared.select_dtypes(include=['object', 'category']).columns
        
        # 数値型の欠損値補完
        for col in numeric_cols:
            if X_prepared[col].isna().any():
                X_prepared[col].fillna(X_prepared[col].median(), inplace=True)
        
        # カテゴリカル型のエンコード
        if len(categorical_cols) > 0:
            # 欠損値を文字列として埋める
            X_prepared[categorical_cols] = X_prepared[categorical_cols].fillna('missing').astype(str)
            
            if is_train:
                encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                X_prepared[categorical_cols] = encoder.fit_transform(X_prepared[categorical_cols])
                return X_prepared, encoder
            else:
                if encoder is None:
                    # エンコーダーがない場合は新規作成（通常ありえないが安全策）
                    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    X_prepared[categorical_cols] = encoder.fit_transform(X_prepared[categorical_cols])
                else:
                    X_prepared[categorical_cols] = encoder.transform(X_prepared[categorical_cols])
                return X_prepared
        
        if is_train:
            return X_prepared, None
        else:
            return X_prepared

    def _cross_validate(self, X, y):
        """
        5-fold Stratified CVでモデルを評価
        
        Parameters:
        -----------
        X : pd.DataFrame
            特徴量
        y : pd.Series
            目的変数
            
        Returns:
        --------
        dict
            各評価指標の平均値
        """
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        scores = {
            'pr_auc': [],
            'roc_auc': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # データ前処理（エンコード）
            X_train_encoded, encoder = self._prepare_data_for_lightgbm(X_train, is_train=True)
            X_val_encoded = self._prepare_data_for_lightgbm(X_val, is_train=False, encoder=encoder)
            
            # モデル訓練
            model = lgb.LGBMClassifier(**self.best_params)
            model.fit(
                X_train_encoded, y_train,
                eval_set=[(X_val_encoded, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
            
            # 予測
            y_pred_proba = model.predict_proba(X_val_encoded)[:, 1]
            y_pred = model.predict(X_val_encoded)
            
            # 評価指標計算
            scores['pr_auc'].append(average_precision_score(y_val, y_pred_proba))
            scores['roc_auc'].append(roc_auc_score(y_val, y_pred_proba))
            scores['accuracy'].append(accuracy_score(y_val, y_pred))
            scores['precision'].append(precision_score(y_val, y_pred, zero_division=0))
            scores['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))
        
        # 平均を計算
        avg_scores = {k: np.mean(v) for k, v in scores.items()}
        
        return avg_scores
    
    def evaluate_all_interactions(self, baseline_scores):
        """
        すべての交互作用特徴量を評価
        
        Parameters:
        -----------
        baseline_scores : dict
            ベースラインモデルの評価指標
            
        Returns:
        --------
        pd.DataFrame
            評価結果（ランキング付き）
        """
        print("\n" + "="*60)
        print("交互作用特徴量の評価開始")
        print("="*60)
        print(f"評価対象: {len(self.metadata)} 個の交互作用特徴量")
        print(f"ベースライン PR-AUC: {baseline_scores['pr_auc']:.6f}")
        print("="*60)
        
        results = []
        
        # チェックポイントファイルのパス
        checkpoint_path = Path(self.interaction_dir).parent / 'results' / 'interaction_features_road_type' / 'checkpoint_results.csv'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 既存のチェックポイントがあれば読み込み
        processed_features = set()
        if checkpoint_path.exists():
            print(f"\n📂 チェックポイントを検出: {checkpoint_path}")
            existing_df = pd.read_csv(checkpoint_path)
            results = existing_df.to_dict('records')
            processed_features = set(existing_df['feature_name'].tolist())
            print(f"   既に処理済み: {len(processed_features)} 個")
            print(f"   残り: {len(self.metadata) - len(processed_features)} 個")
            print("="*60)
        
        # プログレスバー付きで評価
        remaining = len(self.metadata) - len(processed_features)
        pbar = tqdm(total=remaining, desc="交互作用特徴量評価")
        
        for idx, row in self.metadata.iterrows():
            feature_name = row['feature_name']
            
            # 既に処理済みならスキップ
            if feature_name in processed_features:
                continue
            
            feature_path = self.interaction_dir / f"{feature_name}.pkl"
            
            # 交互作用特徴量を読み込み
            with open(feature_path, 'rb') as f:
                interaction_feature = pickle.load(f)
            
            # 元の特徴量に追加
            X_with_interaction = self.X.copy()
            X_with_interaction[feature_name] = interaction_feature
            
            # 評価
            try:
                scores = self._cross_validate(X_with_interaction, self.y)
                
                # ベースラインとの差分を計算
                delta_pr_auc = scores['pr_auc'] - baseline_scores['pr_auc']
                delta_roc_auc = scores['roc_auc'] - baseline_scores['roc_auc']
                delta_f1 = scores['f1'] - baseline_scores['f1']
                
                # 結果を記録
                result = {
                    'feature_name': feature_name,
                    'feature1': row['feature1'],
                    'feature2': row['feature2'],
                    'interaction_type': row['interaction_type'],
                    'pr_auc': scores['pr_auc'],
                    'delta_pr_auc': delta_pr_auc,
                    'roc_auc': scores['roc_auc'],
                    'delta_roc_auc': delta_roc_auc,
                    'f1': scores['f1'],
                    'delta_f1': delta_f1,
                    'accuracy': scores['accuracy'],
                    'precision': scores['precision'],
                    'recall': scores['recall'],
                    'n_unique': row['n_unique'],
                    'missing_rate': row['missing_rate']
                }
                results.append(result)
                
                # チェックポイントを保存（10個ごと）
                if len(results) % 10 == 0:
                    checkpoint_df = pd.DataFrame(results)
                    checkpoint_df.to_csv(checkpoint_path, index=False, encoding='utf-8-sig')
                
            except Exception as e:
                print(f"\n警告: {feature_name} の評価中にエラー: {e}")
                continue
            
            pbar.update(1)
            
            # 進捗を定期的に表示（50個ごと）
            if len(results) % 50 == 0:
                pbar.set_postfix({
                    'Current': feature_name[:30],
                    'Best Delta': f"{max([r['delta_pr_auc'] for r in results]):.6f}"
                })
                # チェックポイントも保存
                checkpoint_df = pd.DataFrame(results)
                checkpoint_df.to_csv(checkpoint_path, index=False, encoding='utf-8-sig')
                print(f"\n💾 チェックポイント保存: {len(results)} 個完了")
        
        pbar.close()
        
        # 最終チェックポイント保存
        checkpoint_df = pd.DataFrame(results)
        checkpoint_df.to_csv(checkpoint_path, index=False, encoding='utf-8-sig')
        print(f"\n✅ 最終チェックポイント保存: {checkpoint_path}")
        
        # 結果をDataFrameに変換
        results_df = pd.DataFrame(results)
        
        # delta_pr_aucでソート（降順）
        results_df = results_df.sort_values('delta_pr_auc', ascending=False).reset_index(drop=True)
        
        # ランク付け
        results_df['rank'] = range(1, len(results_df) + 1)
        
        return results_df
    
    def save_results(self, results_df, output_dir='results/interaction_features'):
        """
        評価結果を保存
        
        Parameters:
        -----------
        results_df : pd.DataFrame
            評価結果
        output_dir : str
            保存先ディレクトリ
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 全結果をCSV保存
        full_csv_path = output_path / f'interaction_features_ranking_full_{timestamp}.csv'
        results_df.to_csv(full_csv_path, index=False, encoding='utf-8-sig')
        print(f"\n全結果を保存: {full_csv_path}")
        
        # Top 100をCSV保存
        top100_csv_path = output_path / f'interaction_features_ranking_top100_{timestamp}.csv'
        results_df.head(100).to_csv(top100_csv_path, index=False, encoding='utf-8-sig')
        print(f"Top 100を保存: {top100_csv_path}")
        
        # サマリー表示
        print("\n" + "="*60)
        print("評価完了サマリー")
        print("="*60)
        print(f"評価した交互作用特徴量数: {len(results_df)}")
        print(f"\nTop 5 交互作用特徴量:")
        for idx, row in results_df.head(5).iterrows():
            print(f"  {row['rank']}. {row['feature_name']}")
            print(f"     Delta PR-AUC: {row['delta_pr_auc']:+.6f} ({row['delta_pr_auc']*100:+.2f}%)")
            print(f"     PR-AUC: {row['pr_auc']:.6f}")
        
        print(f"\nPR-AUC向上した特徴量数: {(results_df['delta_pr_auc'] > 0).sum()}")
        print(f"PR-AUC低下した特徴量数: {(results_df['delta_pr_auc'] < 0).sum()}")
        
        return full_csv_path, top100_csv_path


def main():
    """メイン処理"""
    # 設定（実行時に最新のディレクトリに変更してください）
    DATA_PATH = 'data/processed/honhyo_clean_predictable_only.csv'
    INTERACTION_DIR = 'data/interaction_features_20251211_140000'  # generate_interaction_features.pyの出力ディレクトリ
    METADATA_PATH = f'{INTERACTION_DIR}/interaction_features_metadata.csv'
    TARGET_COLUMN = '死者数'
    OUTPUT_DIR = 'results/interaction_features'
    
    print("="*60)
    print("LightGBMによる交互作用特徴量重要度評価")
    print("="*60)
    print(f"データパス: {DATA_PATH}")
    print(f"交互作用特徴量ディレクトリ: {INTERACTION_DIR}")
    print(f"出力先: {OUTPUT_DIR}")
    print("="*60)
    
    # 評価器の初期化
    evaluator = InteractionFeatureEvaluator(
        data_path=DATA_PATH,
        interaction_metadata_path=METADATA_PATH,
        interaction_dir=INTERACTION_DIR,
        target_column=TARGET_COLUMN,
        n_folds=5,
        random_state=42
    )
    
    # ベースライン評価
    baseline_scores = evaluator.evaluate_baseline()
    
    # すべての交互作用特徴量を評価
    results_df = evaluator.evaluate_all_interactions(baseline_scores)
    
    # 結果を保存
    full_csv, top100_csv = evaluator.save_results(results_df, output_dir=OUTPUT_DIR)
    
    print("\n✅ すべての評価が完了しました！")
    print(f"次のステップ: generate_ranking_report.py を実行してレポートを生成してください")


if __name__ == '__main__':
    main()
