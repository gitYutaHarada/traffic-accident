"""
Mixture of Experts (MoE) Stage 2: 3 Experts 版 (Leakage-Free)
==============================================================
3人の専門家モデルを学習し、Stacking (Stage 3) 用のOOF予測値を出力する。

【重要】Global Fold戦略
- 最初に全体でFoldを固定し、各Expertは同じFold番号で学習・予測を行う
- これにより、OOF予測時のData Leakageを防止

専門家の構成:
- Expert A (Urban): 市街地(地形=3) / 信号なし(信号機=7) / カーブ(道路形状=13)
- Expert B (Night): 夜間 (昼夜 = 21, 22, 23)
- Generalist: 上記以外の全データ

出力 (Stage 3 入力用):
- results/moe_stage2_3experts/oof_predictions.csv (学習データのOOF)
- results/moe_stage2_3experts/test_predictions.csv (テストデータ)

実行方法:
    python scripts/modeling/train_moe_stage2_3experts.py
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
import lightgbm as lgb
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings('ignore')


# =============================================================================
# ドメイン判定関数
# =============================================================================

def create_urban_mask(df):
    """Urban Domain判定 (Expert A担当領域)"""
    terrain = pd.to_numeric(df['地形'], errors='coerce').fillna(-1)
    signal = pd.to_numeric(df['信号機'], errors='coerce').fillna(-1)
    road = pd.to_numeric(df['道路形状'], errors='coerce').fillna(-1)
    return (terrain == 3) | (signal == 7) | (road == 13)

def create_night_mask(df):
    """Night Domain判定 (Expert B担当領域)"""
    daytime = pd.to_numeric(df['昼夜'], errors='coerce').fillna(-1)
    return daytime.isin([21, 22, 23])


# =============================================================================
# 3 Experts Pipeline (Leakage-Free)
# =============================================================================

class MoE3ExpertsPipeline:
    """
    Mixture of Experts Stage 2 Pipeline - 3 Experts Version
    
    【Leakage-Free設計】
    1. Global Fold: 全体で5-Foldを固定し、全Expertが同じ分割を使用
    2. OOF予測: 各サンプルは「自分を学習に使っていないFoldのモデル」で予測
    3. テスト予測: 全Foldのモデル平均で予測
    """

    def __init__(
        self,
        data_path: str = "data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv",
        target_col: str = "fatal",
        n_folds: int = 5,
        random_state: int = 42,
        stage1_threshold: float = 0.0400,
        test_size: float = 0.2,
        output_dir: str = "results/moe_stage2_3experts",
        stage1_ckpt_dir: str = "results/ensemble_stage2/checkpoints",
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.n_folds = n_folds
        self.random_state = random_state
        self.stage1_threshold = stage1_threshold
        self.test_size = test_size
        self.output_dir = output_dir
        self.n_seeds = 3
        self.stage1_ckpt_dir = stage1_ckpt_dir

        self.ckpt_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

        print("=" * 80)
        print("Mixture of Experts (MoE) Stage 2: 3 Experts Version (Leakage-Free)")
        print("  👮 Expert A: Urban Specialist (市街地/信号なし/カーブ)")
        print("  🌃 Expert B: Night Specialist (夜間)")
        print("  🧢 Generalist: 標準領域")
        print(f"  📊 Global Fold戦略: {n_folds}-Fold CV")
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

        # Stage 1 OOF読み込み
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_stage1 = np.zeros(len(self.y_train_full))
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train_full, self.y_train_full)):
            fold_dir = os.path.join(self.stage1_ckpt_dir, f"stage1_fold{fold}")
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

        print(f"\n   Stage 2 データ: {len(self.y_s2):,} (Pos: {self.y_s2.sum():,})")

    def create_domain_masks(self):
        """ドメインマスクを作成"""
        print("\n🏙️  ドメイン分割 (Urban / Night / Generalist)...")
        
        self.urban_mask = create_urban_mask(self.X_s2)
        self.night_mask = create_night_mask(self.X_s2)
        # Night - Urban の純粋な夜間領域
        self.pure_night_mask = self.night_mask & ~self.urban_mask
        # Generalist: Urban でも Night でもない
        self.generalist_mask = ~self.urban_mask & ~self.night_mask

        print(f"   👮 Urban Domain (Expert A): {self.urban_mask.sum():,}")
        print(f"   🌃 Night Domain (Expert B): {self.pure_night_mask.sum():,}")
        print(f"   🧢 Generalist Domain: {self.generalist_mask.sum():,}")

    def _train_single_fold(self, X_train, y_train, X_val, expert_name, fold):
        """単一Foldの学習と予測"""
        save_dir = os.path.join(self.ckpt_dir, f"{expert_name}_fold{fold}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Categorical変換
        X_train_cat = X_train.copy()
        X_val_cat = X_val.copy()
        for col in self.categorical_cols:
            if col in X_train_cat.columns:
                X_train_cat[col] = X_train_cat[col].astype('category')
                X_val_cat[col] = X_val_cat[col].astype('category')
        
        # LightGBM
        path_lgb = os.path.join(save_dir, "lgb_model.pkl")
        if os.path.exists(path_lgb):
            lgb_model = joblib.load(path_lgb)
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
            lgb_model = lgb.LGBMClassifier(**lgb_params, random_state=self.random_state + fold)
            lgb_model.fit(X_train_cat, y_train)
            joblib.dump(lgb_model, path_lgb)
        
        # CatBoost
        path_cat = os.path.join(save_dir, "cat_model.pkl")
        if os.path.exists(path_cat):
            cat_model = joblib.load(path_cat)
        else:
            cat_model = CatBoostClassifier(
                iterations=1000, learning_rate=0.05, depth=8, l2_leaf_reg=3,
                loss_function='Logloss', eval_metric='AUC', random_seed=self.random_state + fold,
                verbose=False, task_type='CPU', thread_count=4,
                cat_features=[c for c in self.categorical_cols if c in X_train_cat.columns]
            )
            cat_model.fit(X_train_cat, y_train, verbose=False)
            joblib.dump(cat_model, path_cat)
        
        # 予測
        pred_lgb = lgb_model.predict_proba(X_val_cat)[:, 1]
        pred_cat = cat_model.predict_proba(X_val_cat)[:, 1]
        pred_ens = (pred_lgb + pred_cat) / 2
        
        return lgb_model, cat_model, pred_ens

    def train_with_global_fold(self):
        """
        Global Fold戦略で全Expertを学習
        
        【Key Point】
        - 全体で5-Foldを固定
        - 各Expertは「自分の担当領域のデータ」のみで学習
        - OOF予測は「自分を学習に使っていないFoldのモデル」で行う
        """
        print("\n🔧 Global Fold戦略で学習中...")
        
        n_samples = len(self.y_s2)
        
        # OOF予測格納配列
        self.oof_expert_a = np.zeros(n_samples)
        self.oof_expert_b = np.zeros(n_samples)
        self.oof_generalist = np.zeros(n_samples)
        
        # モデル保存用
        self.models_expert_a = []
        self.models_expert_b = []
        self.models_generalist = []
        
        # Global Fold
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_s2, self.y_s2)):
            print(f"\n--- Fold {fold} ---")
            
            X_train_fold = self.X_s2.iloc[train_idx]
            y_train_fold = self.y_s2[train_idx]
            X_val_fold = self.X_s2.iloc[val_idx]
            
            # ドメインマスク (Train)
            urban_train = create_urban_mask(X_train_fold)
            night_train = create_night_mask(X_train_fold) & ~urban_train
            gen_train = ~urban_train & ~create_night_mask(X_train_fold)
            
            # === Expert A (Urban) ===
            X_train_a = X_train_fold[urban_train].reset_index(drop=True)
            y_train_a = y_train_fold[urban_train]
            if len(y_train_a) > 0 and y_train_a.sum() > 0:
                lgb_a, cat_a, pred_a = self._train_single_fold(X_train_a, y_train_a, X_val_fold, "ExpertA", fold)
                self.oof_expert_a[val_idx] = pred_a
                self.models_expert_a.append((lgb_a, cat_a))
            else:
                self.models_expert_a.append((None, None))
            
            # === Expert B (Night) ===
            X_train_b = X_train_fold[night_train].reset_index(drop=True)
            y_train_b = y_train_fold[night_train]
            if len(y_train_b) > 0 and y_train_b.sum() > 0:
                lgb_b, cat_b, pred_b = self._train_single_fold(X_train_b, y_train_b, X_val_fold, "ExpertB", fold)
                self.oof_expert_b[val_idx] = pred_b
                self.models_expert_b.append((lgb_b, cat_b))
            else:
                self.models_expert_b.append((None, None))
            
            # === Generalist ===
            X_train_g = X_train_fold[gen_train].reset_index(drop=True)
            y_train_g = y_train_fold[gen_train]
            if len(y_train_g) > 0 and y_train_g.sum() > 0:
                lgb_g, cat_g, pred_g = self._train_single_fold(X_train_g, y_train_g, X_val_fold, "Generalist", fold)
                self.oof_generalist[val_idx] = pred_g
                self.models_generalist.append((lgb_g, cat_g))
            else:
                self.models_generalist.append((None, None))
            
            print(f"   Fold {fold} 完了")
        
        print("\n✅ 全Fold学習完了")

    def predict_test_set(self):
        """テストデータに対して全Expertで予測（全Foldのモデル平均）"""
        print("\n📊 テストデータを予測中...")
        
        # Stage 1フィルタリング（テストデータ）
        X_test_cat = self.X_test.copy()
        for col in self.categorical_cols:
            if col in X_test_cat.columns:
                X_test_cat[col] = X_test_cat[col].astype('category')
        
        # Stage 1 モデルでフィルタリング
        test_proba_stage1 = np.zeros(len(self.y_test))
        for fold in range(self.n_folds):
            fold_dir = os.path.join(self.stage1_ckpt_dir, f"stage1_fold{fold}")
            for seed in range(self.n_seeds):
                model_path = os.path.join(fold_dir, f"seed{seed}_model.pkl")
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    test_proba_stage1 += model.predict_proba(X_test_cat)[:, 1]
        test_proba_stage1 /= (self.n_folds * self.n_seeds)
        
        test_stage2_mask = test_proba_stage1 >= self.stage1_threshold
        X_test_s2 = self.X_test[test_stage2_mask].reset_index(drop=True)
        y_test_s2 = self.y_test[test_stage2_mask]
        
        print(f"   Stage 1 フィルタリング後: {len(y_test_s2):,} / {len(self.y_test):,}")
        
        n_test = len(y_test_s2)
        test_pred_a = np.zeros(n_test)
        test_pred_b = np.zeros(n_test)
        test_pred_g = np.zeros(n_test)
        
        X_test_s2_cat = X_test_s2.copy()
        for col in self.categorical_cols:
            if col in X_test_s2_cat.columns:
                X_test_s2_cat[col] = X_test_s2_cat[col].astype('category')
        
        # 各Expertで予測（全Foldモデルの平均）
        n_valid_a, n_valid_b, n_valid_g = 0, 0, 0
        
        for lgb_m, cat_m in self.models_expert_a:
            if lgb_m is not None:
                test_pred_a += (lgb_m.predict_proba(X_test_s2_cat)[:, 1] + 
                               cat_m.predict_proba(X_test_s2_cat)[:, 1]) / 2
                n_valid_a += 1
        if n_valid_a > 0:
            test_pred_a /= n_valid_a
            
        for lgb_m, cat_m in self.models_expert_b:
            if lgb_m is not None:
                test_pred_b += (lgb_m.predict_proba(X_test_s2_cat)[:, 1] + 
                               cat_m.predict_proba(X_test_s2_cat)[:, 1]) / 2
                n_valid_b += 1
        if n_valid_b > 0:
            test_pred_b /= n_valid_b
            
        for lgb_m, cat_m in self.models_generalist:
            if lgb_m is not None:
                test_pred_g += (lgb_m.predict_proba(X_test_s2_cat)[:, 1] + 
                               cat_m.predict_proba(X_test_s2_cat)[:, 1]) / 2
                n_valid_g += 1
        if n_valid_g > 0:
            test_pred_g /= n_valid_g
        
        # メタ特徴量
        terrain = pd.to_numeric(X_test_s2['地形'], errors='coerce').fillna(-1)
        daytime = pd.to_numeric(X_test_s2['昼夜'], errors='coerce').fillna(-1)
        urban_mask = create_urban_mask(X_test_s2)
        night_mask = create_night_mask(X_test_s2) & ~urban_mask
        
        # テスト予測CSVを作成
        test_df = pd.DataFrame({
            'pred_expert_a': test_pred_a,
            'pred_expert_b': test_pred_b,
            'pred_generalist': test_pred_g,
            'is_urban': urban_mask.astype(int),
            'is_night': night_mask.astype(int),
            'y_true': y_test_s2
        })
        
        test_path = os.path.join(self.output_dir, "test_predictions.csv")
        test_df.to_csv(test_path, index=False)
        print(f"\n   📁 テスト予測CSVを保存: {test_path}")
        
        self.test_predictions = test_df
        return test_df

    def save_oof_predictions(self):
        """OOF予測CSVを保存"""
        print("\n💾 OOF予測CSVを保存中...")
        
        terrain = pd.to_numeric(self.X_s2['地形'], errors='coerce').fillna(-1)
        daytime = pd.to_numeric(self.X_s2['昼夜'], errors='coerce').fillna(-1)
        
        oof_df = pd.DataFrame({
            'pred_expert_a': self.oof_expert_a,
            'pred_expert_b': self.oof_expert_b,
            'pred_generalist': self.oof_generalist,
            'is_urban': self.urban_mask.astype(int),
            'is_night': self.pure_night_mask.astype(int),
            'y_true': self.y_s2
        })
        
        oof_path = os.path.join(self.output_dir, "oof_predictions.csv")
        oof_df.to_csv(oof_path, index=False)
        print(f"   📁 OOF予測CSVを保存: {oof_path}")
        
        self.oof_predictions = oof_df
        return oof_df

    def run(self):
        """パイプライン実行"""
        start = datetime.now()
        self.load_data()
        self.create_domain_masks()
        self.train_with_global_fold()
        self.save_oof_predictions()
        self.predict_test_set()
        
        elapsed_sec = (datetime.now() - start).total_seconds()

        print("\n" + "=" * 70)
        print("✅ MoE Stage 2: 3 Experts (Leakage-Free) 完了!")
        print(f"   総実行時間: {elapsed_sec:.1f}秒")
        print(f"   OOF予測: {self.output_dir}/oof_predictions.csv")
        print(f"   Test予測: {self.output_dir}/test_predictions.csv")
        print(f"   → 次のステップ: train_stage3_stacking.py を実行してください")
        print("=" * 70)

        return {'elapsed_sec': elapsed_sec}


if __name__ == "__main__":
    pipeline = MoE3ExpertsPipeline()
    pipeline.run()
