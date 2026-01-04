"""
Urbanドメイン（Expert A担当領域）のデータ量確認スクリプト
======================================================
目的:
1. MoE戦略で定義した「Urban Domain」に含まれるデータ数を確認する。
2. その中の「死亡事故（正例）」の件数を確認する。

Urban Domain定義:
- 地形コード = 3 (市街地)
- OR 信号機コード = 7 (信号なし・点滅)
- OR 道路形状コード = 13 (カーブ)

実行方法:
    python scripts/analysis/check_urban_stats.py
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, train_test_split

class UrbanStatsChecker:
    def __init__(
        self,
        data_path="data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv",
        target_col="fatal",
        ckpt_dir="results/ensemble_stage2/checkpoints",
        n_folds=5,
        n_seeds=3
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.ckpt_dir = ckpt_dir
        self.n_folds = n_folds
        self.n_seeds = n_seeds
        self.random_state = 42

    def run(self):
        print("📂 データを読み込み中...")
        df = pd.read_csv(self.data_path)
        y_all = df[self.target_col].values
        X_all = df.drop(columns=[self.target_col])
        if '発生日時' in X_all.columns:
            X_all = X_all.drop(columns=['発生日時'])

        # Data Split (Train/Test)
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=self.random_state, stratify=y_all
        )
        X_train = X_train.reset_index(drop=True)
        # y_train は numpy array なので index reset 不要だが、フィルタリング用にSeriesにしても便利
        # ここでは numpy のまま扱う

        # --- Stage 1 OOF 再構築 ---
        print("   Stage 1 OOF 再構築...")
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_stage1 = np.zeros(len(y_train))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            fold_dir = os.path.join(self.ckpt_dir, f"stage1_fold{fold}")
            fold_pred = np.zeros(len(val_idx))
            for seed in range(self.n_seeds):
                pred_path = os.path.join(fold_dir, f"seed{seed}_pred.npy")
                if os.path.exists(pred_path):
                    fold_pred += np.load(pred_path)
            oof_stage1[val_idx] = fold_pred / self.n_seeds

        # 閾値判定 (Recall 99%)
        threshold_stage1 = 0.0400
        stage2_mask = oof_stage1 >= threshold_stage1
        
        X_s2 = X_train[stage2_mask].reset_index(drop=True)
        y_s2 = y_train[stage2_mask]

        print(f"\n📊 Stage 2 データ総数: {len(y_s2):,} (Positive: {y_s2.sum():,})")

        # --- Urban Domain 定義によるフィルタリング ---
        # データ型が object (str) になっている可能性があるので注意
        # もともとのデータ読み込み時点で数値のカラムもあれば文字列のカラムもあるかもしれない
        # ここでは pandas のまま扱うので型変換を試みる
        
        # 定義: 地形=3 OR 信号機=7 OR 道路形状=13
        # カラム名は日本語
        
        def to_float(x):
            try: return float(x)
            except: return -1

        # 各カラムを数値化して判定
        terrain = X_s2['地形'].apply(to_float)
        signal = X_s2['信号機'].apply(to_float)
        road = X_s2['道路形状'].apply(to_float)

        is_terrain_3 = (terrain == 3)
        is_signal_7 = (signal == 7)
        is_road_13 = (road == 13)

        is_urban = is_terrain_3 | is_signal_7 | is_road_13

        n_urban = is_urban.sum()
        n_urban_pos = y_s2[is_urban].sum()

        print("\n🏙️  Urban Domain (Expert A担当) 集計")
        print(f"   データ数: {n_urban:,} ({n_urban/len(y_s2):.1%})")
        print(f"   死亡事故 (Positive): {n_urban_pos:,} ({n_urban_pos/y_s2.sum():.1%} of Stage 2 Positives)")
        print(f"   正例の割合 (Positive Rate): {n_urban_pos/n_urban:.2%}")

        print("\n   [参考] 内訳")
        print(f"   - 地形=3 (市街地): {is_terrain_3.sum():,} (Pos: {y_s2[is_terrain_3].sum():,})")
        print(f"   - 信号=7 (信号なし): {is_signal_7.sum():,} (Pos: {y_s2[is_signal_7].sum():,})")
        print(f"   - 道路=13 (カーブ): {is_road_13.sum():,} (Pos: {y_s2[is_road_13].sum():,})")

if __name__ == "__main__":
    checker = UrbanStatsChecker()
    checker.run()
