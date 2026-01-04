"""
MoEモデルにおけるUrbanバイアス改善確認スクリプト
==============================================
目的:
以前のモデルで「地形=3 (市街地)」がFalse Positive (FP) に占める割合が
全体より +43.7% も高い（過剰に誤検知している）という問題がありました。
MoEモデル導入後、このバイアスがどの程度改善したかを定量化します。

対象:
- MoEモデルの統合予測値
- 比較閾値:
    1. Max F1 Threshold (0.4859) - バランス設定
    2. High Precision Threshold (0.8247) - 確実性重視設定

実行方法:
    python scripts/analysis/check_moe_urban_improvement.py
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, train_test_split

class MoEUrbanBiasChecker:
    def __init__(
        self,
        data_path="data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv",
        target_col="fatal",
        moe_ckpt_dir="results/moe_stage2/checkpoints",
        n_folds=5,
        n_seeds=3
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.moe_ckpt_dir = moe_ckpt_dir
        self.n_folds = n_folds
        self.n_seeds = n_seeds
        self.random_state = 42

    def create_domain_mask(self, df):
        def to_float(x):
            try: return float(x)
            except: return -1
        terrain = df['地形'].apply(to_float)
        signal = df['信号機'].apply(to_float)
        road = df['道路形状'].apply(to_float)
        return (terrain == 3) | (signal == 7) | (road == 13)

    def run(self):
        print("📂 データを読み込み、MoE OOFを再構築中...")
        df = pd.read_csv(self.data_path)
        y_all = df[self.target_col].values
        X_all = df.drop(columns=[self.target_col])
        if '発生日時' in X_all.columns:
            X_all = X_all.drop(columns=['発生日時'])
            
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=self.random_state, stratify=y_all
        )
        self.X_train_full = X_train.reset_index(drop=True)
        self.y_train_full = y_train

        # --- Stage 1 OOF & Mask ---
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
            
        threshold_stage1 = 0.0400
        stage2_mask = oof_stage1 >= threshold_stage1
        
        self.X_s2 = self.X_train_full[stage2_mask].reset_index(drop=True)
        self.y_s2 = self.y_train_full[stage2_mask]

        # --- Domain Split & MoE Reconstruction ---
        urban_mask = self.create_domain_mask(self.X_s2)
        X_urban = self.X_s2[urban_mask].reset_index(drop=True)
        y_urban = self.y_s2[urban_mask]
        X_general = self.X_s2[~urban_mask].reset_index(drop=True)
        y_general = self.y_s2[~urban_mask]

        self.oof_moe_proba = np.zeros(len(self.y_s2))

        # Expert A
        skf_urban = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_urban_ens = np.zeros(len(y_urban))
        for fold, (train_idx, val_idx) in enumerate(skf_urban.split(X_urban, y_urban)):
            fold_dir = os.path.join(self.moe_ckpt_dir, f"ExpertA_Urban_fold{fold}")
            p_lgb = np.load(os.path.join(fold_dir, "lgb_pred.npy"))
            p_cat = np.load(os.path.join(fold_dir, "cat_pred.npy"))
            oof_urban_ens[val_idx] = 0.5 * p_lgb + 0.5 * p_cat
        
        # Generalist
        skf_general = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_general_ens = np.zeros(len(y_general))
        for fold, (train_idx, val_idx) in enumerate(skf_general.split(X_general, y_general)):
            fold_dir = os.path.join(self.moe_ckpt_dir, f"Generalist_NonUrban_fold{fold}")
            p_lgb = np.load(os.path.join(fold_dir, "lgb_pred.npy"))
            p_cat = np.load(os.path.join(fold_dir, "cat_pred.npy"))
            oof_general_ens[val_idx] = 0.5 * p_lgb + 0.5 * p_cat

        self.oof_moe_proba[urban_mask.values] = oof_urban_ens
        self.oof_moe_proba[~urban_mask.values] = oof_general_ens

        # --- Analysis ---
        print("\n📊 Urban Bias (地形=3) 分析")
        print(f"   比較基準 (前回): +43.7% (Over-represented)")
        
        # 全体分布 (地形=3 の割合)
        # 注意: 文字列型になっている可能性があるので変換しながら確認
        def is_terrain_3(x):
            return str(x) == '3.0' or str(x) == '3'

        overall_ratio = self.X_s2['地形'].apply(is_terrain_3).mean()
        print(f"   全体における地形=3の割合: {overall_ratio:.2%}")

        # Thresholds to check
        thresholds = {
            "Max F1 (0.4859)": 0.4859,
            "High Precision (0.8247)": 0.8247
        }

        for name, th in thresholds.items():
            pred_binary = (self.oof_moe_proba >= th).astype(int)
            is_fp = (self.y_s2 == 0) & (pred_binary == 1)
            
            n_fp = is_fp.sum()
            if n_fp == 0:
                print(f"\n   [{name}] FP数: 0 -> 計算不能")
                continue

            fp_ratio = self.X_s2[is_fp]['地形'].apply(is_terrain_3).mean()
            diff = fp_ratio - overall_ratio
            
            print(f"\n   [{name}]")
            print(f"      FP数: {n_fp:,}")
            print(f"      FP内の地形=3 割合: {fp_ratio:.2%}")
            print(f"      乖離 (Bias): {diff:+.2%} (前回差: {diff*100 - 43.7:+.1f}pt)")
            
            if diff < 0.437:
                print(f"      ✅ 改善しました！ (+43.7% -> {diff*100:+.1f}%)")
            else:
                print(f"      ⚠️ 悪化または変化なし")

if __name__ == "__main__":
    checker = MoEUrbanBiasChecker()
    checker.run()
