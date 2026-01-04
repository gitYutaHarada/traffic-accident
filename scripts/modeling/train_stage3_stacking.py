"""
Stage 3: Stacking Meta-Model (Enhanced Robust Version)
=======================================================
Two-Stage CatBoostとSingle-Stage TabNetの予測値を、
ロジスティック回帰メタモデルで統合する。

【特徴】
- ID-Based Alignment: 全てのマージは `original_index` カラムをキーとして実行
- Dynamic Feature Selection: 全モデルの予測値を候補として比較し、最適な組み合わせを選択
- Multicollinearity対策: 多重共線性の検出と強化された正則化
- Robust Missing Value Imputation: Easy SampleはSingle-Stageの予測値で補完
- Intel Extension for Scikit-learn (sklearnex) サポート
- Intel Core Ultra 9 285K 最適化

実行方法:
    python scripts/modeling/train_stage3_stacking.py

前提条件:
    - Single-Stage OOF (`spatio_temporal_ensemble/oof_predictions.csv`) に `original_index` が含まれること
    - Two-Stage OOF (`twostage_spatiotemporal_ensemble/oof_predictions.csv`) に `original_index` が含まれること
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

# Intel Extension for Scikit-learn (オプション高速化)
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    SKLEARNEX_AVAILABLE = True
    print("✅ Intel Extension for Scikit-learn が有効化されました")
except ImportError:
    SKLEARNEX_AVAILABLE = False

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

# ========================================
# 定数
# ========================================
RANDOM_SEED = 42
N_FOLDS = 5

# パス設定
DATA_DIR = Path("data")
SPATIO_TEMPORAL_DIR = DATA_DIR / "spatio_temporal"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = Path("results")

# 入力ファイル
STAGE1_OOF_PATH = PROCESSED_DIR / "stage1_oof_predictions.csv"
STAGE1_TEST_PATH = PROCESSED_DIR / "stage1_test_predictions.csv"
SINGLE_STAGE_OOF_PATH = RESULTS_DIR / "spatio_temporal_ensemble" / "oof_predictions.csv"
SINGLE_STAGE_TEST_PATH = RESULTS_DIR / "spatio_temporal_ensemble" / "test_predictions.csv"
TWO_STAGE_OOF_PATH = RESULTS_DIR / "twostage_spatiotemporal_ensemble" / "oof_predictions.csv"
TWO_STAGE_TEST_PATH = RESULTS_DIR / "twostage_spatiotemporal_ensemble" / "test_predictions.csv"

# 出力ディレクトリ
OUTPUT_DIR = RESULTS_DIR / "stage3_stacking"


class StackingMetaModel:
    """Stage 3 Stacking メタモデル（ID-Based Alignment + Dynamic Feature Selection + Robust）"""
    
    def __init__(
        self,
        output_dir: Path = OUTPUT_DIR,
        n_folds: int = N_FOLDS,
        random_state: int = RANDOM_SEED,
        use_all_models: bool = True,
        regularization_c: float = 0.1,  # 多重共線性対策のため強化（デフォルト1.0→0.1）
        use_single_stage_imputation: bool = True,  # Easy SampleをSingle-Stageで補完
    ):
        self.output_dir = output_dir
        self.n_folds = n_folds
        self.random_state = random_state
        self.use_all_models = use_all_models
        self.regularization_c = regularization_c
        self.use_single_stage_imputation = use_single_stage_imputation
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # データ格納
        self.df_train = None
        self.df_test = None
        self.feature_names = None
        self.all_feature_names = None
        
        # モデル性能比較用
        self.model_aucs = {}
        
        # 予測格納
        self.oof_predictions = None
        self.test_predictions = None
        
        print("=" * 70)
        print("🚀 Stage 3: Stacking Meta-Model (Enhanced Robust Version)")
        print(f"   Output: {self.output_dir}")
        print(f"   Folds: {n_folds}, Seed: {random_state}")
        print(f"   Use All Models: {use_all_models}")
        print(f"   Regularization C: {regularization_c} (低いほど正則化が強い)")
        print(f"   Use Single-Stage Imputation: {use_single_stage_imputation}")
        print(f"   Intel sklearnex: {'有効' if SKLEARNEX_AVAILABLE else '無効'}")
        print("=" * 70)
    
    def _check_unique(self, df: pd.DataFrame, name: str):
        """original_index の一意性を確認"""
        if df['original_index'].duplicated().any():
            dup_count = df['original_index'].duplicated().sum()
            raise ValueError(
                f"❌ {name} に original_index の重複が {dup_count} 件あります。\n"
                "マージ前に解消してください。"
            )
    
    def _ensure_index_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """original_index の型を統一 (int)"""
        if 'original_index' in df.columns:
            df['original_index'] = df['original_index'].astype(int)
        return df
    
    def load_data(self):
        """
        データ読み込みとID-Basedマージ（改善版）
        - 重複チェックの追加
        - Two-Stageの欠損補完戦略の改善
        - original_index の型統一
        """
        print("\n📂 データ読み込み中（ID-Basedマージ + 改善版）...")
        
        # ========================================
        # 1. Single-Stage OOF をベースにロード（全モデル取得）
        # ========================================
        print("\n   📊 Single-Stage OOFをロード（ベース）...")
        single_oof = pd.read_csv(SINGLE_STAGE_OOF_PATH)
        single_oof = self._ensure_index_type(single_oof)
        # 【Fix】Train+Valが含まれている場合、重複を削除（最初の行を保持）
        if single_oof['original_index'].duplicated().any():
            dup_count = single_oof['original_index'].duplicated().sum()
            print(f"      ⚠️ 重複 {dup_count:,} 件を検出（Train+Val形式）。重複を削除します...")
            single_oof = single_oof.drop_duplicates(subset='original_index', keep='first')
        self._check_unique(single_oof, "Single-Stage OOF")
        
        print(f"      Single-Stage OOF: {len(single_oof):,} 行")
        print(f"      カラム: {list(single_oof.columns)}")
        
        # 利用可能なモデルカラムを特定
        single_model_cols = [c for c in single_oof.columns 
                            if c not in ['original_index', 'target', 'ensemble']]
        print(f"      利用可能なモデル: {single_model_cols}")
        
        # ベースDataFrame作成
        base_cols = ['original_index', 'target'] + single_model_cols
        self.df_train = single_oof[[c for c in base_cols if c in single_oof.columns]].copy()
        
        # カラム名にプレフィックスを追加
        rename_dict = {col: f'single_{col}' for col in single_model_cols}
        self.df_train = self.df_train.rename(columns=rename_dict)
        
        # アンサンブルカラムの処理
        if 'ensemble' in single_oof.columns:
            self.df_train['single_ensemble'] = single_oof['ensemble'].values
        else:
            model_preds = single_oof[single_model_cols].values
            self.df_train['single_ensemble'] = model_preds.mean(axis=1)
        
        # ========================================
        # 2. Stage 1 OOF をマージ
        # ========================================
        print("\n   📊 Stage 1 OOFをマージ...")
        stage1_oof = pd.read_csv(STAGE1_OOF_PATH)
        stage1_oof = self._ensure_index_type(stage1_oof)
        self._check_unique(stage1_oof, "Stage 1 OOF")
        
        if 'ensemble_prob' in stage1_oof.columns:
            stage1_prob_col = 'ensemble_prob'
        elif 'prob_catboost' in stage1_oof.columns:
            stage1_oof['ensemble_prob'] = 0.85 * stage1_oof['prob_catboost'] + 0.15 * stage1_oof['prob_lgbm']
            stage1_prob_col = 'ensemble_prob'
        else:
            raise ValueError("Stage 1 OOFに確率カラムがありません")
        
        stage1_for_merge = stage1_oof[['original_index', stage1_prob_col]].copy()
        stage1_for_merge = stage1_for_merge.rename(columns={stage1_prob_col: 'stage1_prob'})
        
        self.df_train = self.df_train.merge(stage1_for_merge, on='original_index', how='left')
        
        n_merged_s1 = self.df_train['stage1_prob'].notna().sum()
        print(f"      マージ成功: {n_merged_s1:,} / {len(self.df_train):,}")
        self.df_train['stage1_prob'] = self.df_train['stage1_prob'].fillna(0)
        
        # ========================================
        # 3. Two-Stage OOF をマージ（全モデル取得）
        # ========================================
        print("\n   📊 Two-Stage OOFをマージ...")
        two_stage_oof = pd.read_csv(TWO_STAGE_OOF_PATH)
        two_stage_oof = self._ensure_index_type(two_stage_oof)
        # 【Fix】Train+Valが含まれている場合、重複を削除（最初の行を保持）
        if two_stage_oof['original_index'].duplicated().any():
            dup_count = two_stage_oof['original_index'].duplicated().sum()
            print(f"      ⚠️ 重複 {dup_count:,} 件を検出。重複を削除します...")
            two_stage_oof = two_stage_oof.drop_duplicates(subset='original_index', keep='first')
        self._check_unique(two_stage_oof, "Two-Stage OOF")
        
        print(f"      Two-Stage OOF: {len(two_stage_oof):,} 行 (Hard Samples)")
        
        two_stage_model_cols = [c for c in two_stage_oof.columns 
                               if c not in ['original_index', 'target', 'ensemble']]
        print(f"      利用可能なモデル: {two_stage_model_cols}")
        
        two_stage_for_merge = two_stage_oof[['original_index'] + two_stage_model_cols].copy()
        rename_dict_ts = {col: f'twostage_{col}' for col in two_stage_model_cols}
        two_stage_for_merge = two_stage_for_merge.rename(columns=rename_dict_ts)
        
        self.df_train = self.df_train.merge(two_stage_for_merge, on='original_index', how='left')
        
        n_merged_ts = self.df_train['twostage_catboost'].notna().sum() if 'twostage_catboost' in self.df_train.columns else 0
        print(f"      マージ成功: {n_merged_ts:,} / {len(self.df_train):,}")
        
        # ========================================
        # 【改善】Easy Sampleの欠損補完戦略
        # ========================================
        ts_first_col = f'twostage_{two_stage_model_cols[0]}' if two_stage_model_cols else None
        if ts_first_col:
            self.df_train['is_easy_sample'] = self.df_train[ts_first_col].isna().astype(int)
        else:
            self.df_train['is_easy_sample'] = 0
        
        n_easy = self.df_train['is_easy_sample'].sum()
        print(f"\n   📊 Easy Sample: {n_easy:,} / {len(self.df_train):,} ({n_easy/len(self.df_train)*100:.1f}%)")
        
        if self.use_single_stage_imputation:
            print("   🔧 Easy SampleをSingle-Stage予測値で補完...")
            for ts_col in [c for c in self.df_train.columns if c.startswith('twostage_')]:
                # 対応するSingle-Stageカラムを特定
                single_counterpart = ts_col.replace('twostage_', 'single_')
                if single_counterpart in self.df_train.columns:
                    # NaN部分をSingle-Stageの値で補完
                    mask = self.df_train[ts_col].isna()
                    self.df_train.loc[mask, ts_col] = self.df_train.loc[mask, single_counterpart]
                    filled_count = mask.sum()
                    if filled_count > 0:
                        print(f"      {ts_col}: {filled_count:,} 件を {single_counterpart} で補完")
                else:
                    # 対応がない場合は0埋め
                    self.df_train[ts_col] = self.df_train[ts_col].fillna(0)
        else:
            # 従来の0埋め
            for col in [c for c in self.df_train.columns if c.startswith('twostage_')]:
                self.df_train[col] = self.df_train[col].fillna(0)
        
        # ========================================
        # 4. 交互作用項を追加
        # ========================================
        if 'single_tabnet' in self.df_train.columns and 'twostage_catboost' in self.df_train.columns:
            self.df_train['tabnet_x_catboost'] = self.df_train['single_tabnet'] * self.df_train['twostage_catboost']
        
        # ========================================
        # 5. 特徴量候補を整理
        # ========================================
        self.all_feature_names = [c for c in self.df_train.columns 
                                 if c not in ['original_index', 'target'] 
                                 and self.df_train[c].dtype in ['float64', 'int64', 'float32', 'int32']]
        
        print(f"\n      全特徴量候補: {len(self.all_feature_names)} 個")
        print(f"      学習用DataFrame: {len(self.df_train):,} 行")
        
        # ========================================
        # 6. テストデータも同様にマージ
        # ========================================
        self._load_test_data(two_stage_model_cols)
        
        # ========================================
        # 7. 整合性検証
        # ========================================
        self._validate_data()
        
        print("\n   ✅ データ読み込み完了")
    
    def _load_test_data(self, two_stage_model_cols: List[str]):
        """テストデータの読み込み"""
        print("\n   📊 テストデータをマージ...")
        
        single_test = pd.read_csv(SINGLE_STAGE_TEST_PATH)
        single_test = self._ensure_index_type(single_test)
        self._check_unique(single_test, "Single-Stage Test")
        
        single_test_model_cols = [c for c in single_test.columns 
                                 if c not in ['original_index', 'target', 'ensemble']]
        
        base_cols_test = ['original_index'] + single_test_model_cols
        self.df_test = single_test[[c for c in base_cols_test if c in single_test.columns]].copy()
        rename_dict_test = {col: f'single_{col}' for col in single_test_model_cols}
        self.df_test = self.df_test.rename(columns=rename_dict_test)
        
        # アンサンブル
        if 'ensemble' in single_test.columns:
            self.df_test['single_ensemble'] = single_test['ensemble'].values
        else:
            model_preds_test = single_test[single_test_model_cols].values
            self.df_test['single_ensemble'] = model_preds_test.mean(axis=1)
        
        # Stage 1 Test
        stage1_test = pd.read_csv(STAGE1_TEST_PATH)
        stage1_test = self._ensure_index_type(stage1_test)
        self._check_unique(stage1_test, "Stage 1 Test")
        
        if 'ensemble_prob' in stage1_test.columns:
            s1_test_prob_col = 'ensemble_prob'
        elif 'prob_catboost' in stage1_test.columns:
            stage1_test['ensemble_prob'] = 0.85 * stage1_test['prob_catboost'] + 0.15 * stage1_test['prob_lgbm']
            s1_test_prob_col = 'ensemble_prob'
        else:
            s1_test_prob_col = None
        
        if s1_test_prob_col:
            s1_test_merge = stage1_test[['original_index', s1_test_prob_col]].copy()
            s1_test_merge = s1_test_merge.rename(columns={s1_test_prob_col: 'stage1_prob'})
            self.df_test = self.df_test.merge(s1_test_merge, on='original_index', how='left')
            self.df_test['stage1_prob'] = self.df_test['stage1_prob'].fillna(0)
        else:
            self.df_test['stage1_prob'] = 0
        
        # Two-Stage Test
        two_stage_test = pd.read_csv(TWO_STAGE_TEST_PATH)
        two_stage_test = self._ensure_index_type(two_stage_test)
        self._check_unique(two_stage_test, "Two-Stage Test")
        
        ts_test_model_cols = [c for c in two_stage_test.columns 
                             if c not in ['original_index', 'target', 'ensemble']]
        ts_test_merge = two_stage_test[['original_index'] + ts_test_model_cols].copy()
        rename_dict_ts_test = {col: f'twostage_{col}' for col in ts_test_model_cols}
        ts_test_merge = ts_test_merge.rename(columns=rename_dict_ts_test)
        self.df_test = self.df_test.merge(ts_test_merge, on='original_index', how='left')
        
        # Easy Sample フラグ
        ts_first_col = f'twostage_{ts_test_model_cols[0]}' if ts_test_model_cols else None
        if ts_first_col and ts_first_col in self.df_test.columns:
            self.df_test['is_easy_sample'] = self.df_test[ts_first_col].isna().astype(int)
        else:
            self.df_test['is_easy_sample'] = 0
        
        # Two-Stageの欠損補完
        if self.use_single_stage_imputation:
            for ts_col in [c for c in self.df_test.columns if c.startswith('twostage_')]:
                single_counterpart = ts_col.replace('twostage_', 'single_')
                if single_counterpart in self.df_test.columns:
                    mask = self.df_test[ts_col].isna()
                    self.df_test.loc[mask, ts_col] = self.df_test.loc[mask, single_counterpart]
                else:
                    self.df_test[ts_col] = self.df_test[ts_col].fillna(0)
        else:
            for col in [c for c in self.df_test.columns if c.startswith('twostage_')]:
                self.df_test[col] = self.df_test[col].fillna(0)
        
        # 交互作用項
        if 'single_tabnet' in self.df_test.columns and 'twostage_catboost' in self.df_test.columns:
            self.df_test['tabnet_x_catboost'] = self.df_test['single_tabnet'] * self.df_test['twostage_catboost']
        
        # テストのtarget
        raw_test = pd.read_parquet(SPATIO_TEMPORAL_DIR / "raw_test.parquet")
        if 'fatal' in raw_test.columns:
            raw_test['original_index'] = raw_test.index
            target_merge = raw_test[['original_index', 'fatal']].copy()
            target_merge = self._ensure_index_type(target_merge)
            self.df_test = self.df_test.merge(target_merge, on='original_index', how='left')
            self.df_test = self.df_test.rename(columns={'fatal': 'target'})
        
        print(f"      テスト用DataFrame: {len(self.df_test):,} 行")
    
    def _validate_data(self):
        """データ整合性検証"""
        print("\n   🔍 整合性検証...")
        
        # NaNチェック
        for col in self.all_feature_names:
            if col in self.df_train.columns:
                train_nan = self.df_train[col].isna().sum()
                test_nan = self.df_test[col].isna().sum() if col in self.df_test.columns else 0
                if train_nan > 0 or test_nan > 0:
                    print(f"      ⚠️ {col}: Train NaN={train_nan}, Test NaN={test_nan}")
        
        # マージ後の行数検証
        print(f"      Train行数: {len(self.df_train):,}")
        print(f"      Test行数: {len(self.df_test):,}")
    
    def evaluate_feature_sets(self):
        """
        複数の特徴量セットを評価し、最適なセットを選択する
        多重共線性が高いセットには警告を出す
        """
        print("\n🔬 特徴量セットの評価中...")
        
        y = self.df_train['target'].values
        results = []
        
        # 評価する特徴量セットの定義（多重共線性を考慮）
        feature_sets = {
            # 基本セット（多重共線性が低い組み合わせ）
            "baseline": ['stage1_prob', 'single_tabnet', 'twostage_catboost', 'tabnet_x_catboost', 'is_easy_sample'],
            
            # TabNet重視セット（最小限の特徴量）
            "tabnet_focus": ['single_tabnet', 'twostage_catboost', 'is_easy_sample'],
            
            # アンサンブル使用セット（多重共線性リスク低）
            "ensemble_based": ['stage1_prob', 'single_ensemble', 'twostage_catboost', 'is_easy_sample'],
            
            # 多様性重視セット（異なるタイプのモデルのみ）
            "diversity": ['single_tabnet', 'single_lgbm', 'twostage_catboost', 'is_easy_sample'],
        }
        
        # 存在する特徴量のみに絞る
        valid_feature_sets = {}
        for name, cols in feature_sets.items():
            valid_cols = [c for c in cols if c in self.df_train.columns]
            if len(valid_cols) >= 2:
                valid_feature_sets[name] = valid_cols
        
        print(f"   評価可能な特徴量セット: {list(valid_feature_sets.keys())}")
        
        for set_name, feature_cols in valid_feature_sets.items():
            X = self.df_train[feature_cols].values
            
            # 相関行列をチェック（多重共線性の警告）
            corr_matrix = self.df_train[feature_cols].corr().abs()
            high_corr_pairs = []
            for i in range(len(feature_cols)):
                for j in range(i+1, len(feature_cols)):
                    if corr_matrix.iloc[i, j] > 0.8:
                        high_corr_pairs.append((feature_cols[i], feature_cols[j], corr_matrix.iloc[i, j]))
            
            if high_corr_pairs:
                print(f"   ⚠️ {set_name}: 高相関ペア検出")
                for pair in high_corr_pairs:
                    print(f"      - {pair[0]} ↔ {pair[1]}: {pair[2]:.3f}")
            
            oof_preds = np.zeros(len(X))
            scaler = StandardScaler()
            skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]
                
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                
                model = LogisticRegression(
                    C=self.regularization_c,  # 強化された正則化
                    penalty='l2', 
                    solver='lbfgs', 
                    max_iter=1000, 
                    random_state=self.random_state
                )
                model.fit(X_train_scaled, y_train)
                
                oof_preds[val_idx] = model.predict_proba(X_val_scaled)[:, 1]
            
            oof_auc = roc_auc_score(y, oof_preds)
            oof_prauc = average_precision_score(y, oof_preds)
            
            results.append({
                'set_name': set_name,
                'features': feature_cols,
                'oof_auc': oof_auc,
                'oof_prauc': oof_prauc,
                'high_corr_pairs': len(high_corr_pairs),
            })
            
            print(f"   {set_name}: AUC={oof_auc:.4f}, PR-AUC={oof_prauc:.4f}, 高相関ペア={len(high_corr_pairs)}")
        
        # 最良のセットを選択（PR-AUC優先、高相関ペアが少ないものを好む）
        # PR-AUCが0.01以上差がなければ、高相関ペアが少ない方を選ぶ
        sorted_results = sorted(results, key=lambda x: (-x['oof_prauc'], x['high_corr_pairs']))
        best_result = sorted_results[0]
        
        self.feature_names = best_result['features']
        
        print(f"\n   📌 選択された特徴量セット: {best_result['set_name']}")
        print(f"      特徴量: {self.feature_names}")
        print(f"      OOF AUC: {best_result['oof_auc']:.4f}")
        print(f"      OOF PR-AUC: {best_result['oof_prauc']:.4f}")
        
        # 結果をJsonで保存
        with open(self.output_dir / "feature_selection_results.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return best_result
    
    def train(self):
        """メタモデルの学習（係数監視付き）"""
        print("\n🧠 Stacking メタモデル学習中...")
        print(f"   正則化強度 C={self.regularization_c} (低いほど正則化が強い)")
        
        X = self.df_train[self.feature_names].values
        y = self.df_train['target'].values
        X_test = self.df_test[self.feature_names].values
        
        self.oof_predictions = np.zeros(len(X))
        self.test_predictions = np.zeros(len(X_test))
        
        scaler = StandardScaler()
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        fold_aucs = []
        all_coefs = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"   Fold {fold+1}/{self.n_folds}...")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            X_test_scaled = scaler.transform(X_test)
            
            model = LogisticRegression(
                C=self.regularization_c,
                penalty='l2',
                solver='lbfgs',
                max_iter=1000,
                random_state=self.random_state,
            )
            model.fit(X_train_scaled, y_train)
            
            val_pred = model.predict_proba(X_val_scaled)[:, 1]
            test_pred = model.predict_proba(X_test_scaled)[:, 1]
            
            self.oof_predictions[val_idx] = val_pred
            self.test_predictions += test_pred / self.n_folds
            
            fold_auc = roc_auc_score(y_val, val_pred)
            fold_aucs.append(fold_auc)
            all_coefs.append(model.coef_[0])
            print(f"      Fold {fold+1} AUC: {fold_auc:.4f}")
        
        oof_auc = roc_auc_score(y, self.oof_predictions)
        oof_prauc = average_precision_score(y, self.oof_predictions)
        
        print(f"\n   📊 Stacking OOF AUC:    {oof_auc:.4f}")
        print(f"   📊 Stacking OOF PR-AUC: {oof_prauc:.4f}")
        
        # 係数の分析（多重共線性の警告付き）
        print("\n   📈 メタモデル係数確認 (負の係数に注意):")
        mean_coefs = np.mean(all_coefs, axis=0)
        std_coefs = np.std(all_coefs, axis=0)
        
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coeff_Mean': mean_coefs,
            'Coeff_Std': std_coefs,
        }).sort_values(by='Coeff_Mean', ascending=False)
        
        print(coef_df.to_string(index=False))
        print(f"      intercept: {model.intercept_[0]:.4f}")
        
        # 警告: 負の係数または不安定な係数
        neg_coefs = coef_df[coef_df['Coeff_Mean'] < 0]
        if not neg_coefs.empty:
            print(f"\n   ⚠️ 警告: 以下の特徴量の係数が負になっています。多重共線性の可能性があります:")
            print(neg_coefs.to_string(index=False))
        
        unstable_coefs = coef_df[coef_df['Coeff_Std'] > abs(coef_df['Coeff_Mean']) * 0.5]
        if not unstable_coefs.empty:
            print(f"\n   ⚠️ 警告: 以下の特徴量の係数がFold間で不安定です:")
            print(unstable_coefs.to_string(index=False))
        
        # 係数情報を保存
        coef_df.to_csv(self.output_dir / "model_coefficients.csv", index=False)
        
        return oof_auc, oof_prauc
    
    def save_results(self):
        """結果の保存"""
        print("\n📈 結果保存中...")
        
        y_train = self.df_train['target'].values
        y_test = self.df_test['target'].values if 'target' in self.df_test.columns and self.df_test['target'].notna().all() else None
        
        # OOF予測
        oof_df = pd.DataFrame({
            'original_index': self.df_train['original_index'].values,
            'stacking_prob': self.oof_predictions,
            'target': y_train,
        })
        oof_df.to_csv(self.output_dir / "oof_predictions.csv", index=False)
        
        # テスト予測
        test_df = pd.DataFrame({
            'original_index': self.df_test['original_index'].values,
            'stacking_prob': self.test_predictions,
        })
        if y_test is not None:
            test_df['target'] = y_test
        test_df.to_csv(self.output_dir / "test_predictions.csv", index=False)
        
        # 最終提出用ファイル
        submission_df = pd.DataFrame({
            'original_index': self.df_test['original_index'].values,
            'prob': self.test_predictions,
        })
        submission_df.to_csv(self.output_dir / "final_submission_stacking.csv", index=False)
        
        # スコアサマリー
        oof_auc = roc_auc_score(y_train, self.oof_predictions)
        oof_prauc = average_precision_score(y_train, self.oof_predictions)
        
        scores = {
            'oof_auc': float(oof_auc),
            'oof_prauc': float(oof_prauc),
            'selected_features': self.feature_names,
            'regularization_c': self.regularization_c,
            'use_single_stage_imputation': self.use_single_stage_imputation,
        }
        if y_test is not None:
            test_auc = roc_auc_score(y_test, self.test_predictions)
            test_prauc = average_precision_score(y_test, self.test_predictions)
            scores['test_auc'] = float(test_auc)
            scores['test_prauc'] = float(test_prauc)
            print(f"   Test AUC:    {test_auc:.4f}")
            print(f"   Test PR-AUC: {test_prauc:.4f}")
        
        with open(self.output_dir / "scores.json", 'w') as f:
            json.dump(scores, f, indent=2)
        
        print(f"   ✅ 完了: {self.output_dir}")
    
    def run(self):
        """全工程実行"""
        start_time = datetime.now()
        
        self.load_data()
        
        if self.use_all_models:
            self.evaluate_feature_sets()
        else:
            # 従来の固定特徴量
            self.feature_names = ['stage1_prob', 'single_tabnet', 'twostage_catboost', 'tabnet_x_catboost', 'is_easy_sample']
            self.feature_names = [c for c in self.feature_names if c in self.df_train.columns]
        
        self.train()
        self.save_results()
        
        elapsed = (datetime.now() - start_time).total_seconds() / 60
        
        print("\n" + "=" * 70)
        print(f"✅ 全工程完了！ 実行時間: {elapsed:.1f}分")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Stage 3 Stacking Meta-Model (Enhanced Robust Version)")
    parser.add_argument(
        '--use-all-models',
        action='store_true',
        default=True,
        help='全モデルの予測値を候補として動的に選択（デフォルト: True）'
    )
    parser.add_argument(
        '--regularization-c',
        type=float,
        default=0.1,
        help='ロジスティック回帰の正則化強度 C（低いほど強い正則化、デフォルト: 0.1）'
    )
    parser.add_argument(
        '--use-single-stage-imputation',
        action='store_true',
        default=True,
        help='Easy SampleをSingle-Stage予測値で補完（デフォルト: True）'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(OUTPUT_DIR),
        help='結果出力ディレクトリ'
    )
    args = parser.parse_args()
    
    stacking = StackingMetaModel(
        output_dir=Path(args.output_dir),
        n_folds=N_FOLDS,
        random_state=RANDOM_SEED,
        use_all_models=args.use_all_models,
        regularization_c=args.regularization_c,
        use_single_stage_imputation=args.use_single_stage_imputation,
    )
    stacking.run()


if __name__ == "__main__":
    main()
