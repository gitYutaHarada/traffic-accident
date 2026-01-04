"""
「判定が難しい事故 (Hard Samples)」分析スクリプト
================================================
目的:
1. Stage 2 アンサンブルモデルにとって「難しい」サンプルを特定する。
   - Boundary Samples: 予測確率が 0.5 付近 (0.3 - 0.7) で迷っているデータ
   - High Variance Samples: モデル間 (LGBM, CatBoost, TabNet) で意見が割れているデータ
   - Error Samples: False Positive (誤検知) / False Negative (見逃し)

2. これらのサンプルの特徴（昼夜、地形、道路形状など）を集計し、
   「専門家モデル (Mixture of Experts)」の切り口を提案する。

実行方法:
    python scripts/analysis/analyze_hard_samples.py
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import recall_score

class HardSampleAnalyzer:
    def __init__(
        self,
        data_path="data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv",
        target_col="fatal",
        ckpt_dir="results/ensemble_stage2/checkpoints",
        output_dir="results/ensemble_stage2",
        random_state=42,
        n_folds=5,
        n_seeds=3
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.ckpt_dir = ckpt_dir
        self.output_dir = output_dir
        self.random_state = random_state
        self.n_folds = n_folds
        self.n_seeds = n_seeds

    def load_data_and_predictions(self):
        """データと予測値の読み込み (OOF再構築)"""
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
        self.X_train = X_train
        self.y_train = y_train

        # --- Stage 1 OOF & Mask ---
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
        threshold_stage1 = 0.0400 # 前回の実験値を使用
        stage2_mask = oof_stage1 >= threshold_stage1
        
        self.X_s2 = X_train[stage2_mask].reset_index(drop=True)
        self.y_s2 = y_train[stage2_mask]
        print(f"   Stage 2 Target Data: {len(self.y_s2):,}")

        # --- Stage 2 OOF (All Models) ---
        print("   Stage 2 OOF 再構築 (LGBM, CatBoost, TabNet)...")
        skf_s2 = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        self.oof_lgb = np.zeros(len(self.y_s2))
        self.oof_cat = np.zeros(len(self.y_s2))
        self.oof_tab = np.zeros(len(self.y_s2))
        
        for fold, (train_idx, val_idx) in enumerate(skf_s2.split(self.X_s2, self.y_s2)):
            fold_dir = os.path.join(self.ckpt_dir, f"stage2_fold{fold}")
            
            p_lgb = os.path.join(fold_dir, "lgb_pred.npy")
            p_cat = os.path.join(fold_dir, "cat_pred.npy")
            p_tab = os.path.join(fold_dir, "tab_pred.npy")
            
            if os.path.exists(p_lgb): self.oof_lgb[val_idx] = np.load(p_lgb)
            if os.path.exists(p_cat): self.oof_cat[val_idx] = np.load(p_cat)
            if os.path.exists(p_tab): self.oof_tab[val_idx] = np.load(p_tab)

        # Ensemble Prediction (Even weights for analysis simplicity or optimized)
        # Optimized weights from previous run: 0.333 each
        w = [1/3, 1/3, 1/3]
        self.oof_ens = w[0]*self.oof_lgb + w[1]*self.oof_cat + w[2]*self.oof_tab

    def analyze_hard_samples(self):
        """難しいサンプルの特定と分析"""
        print("\n🔍 判定難易度分析を実行中...")
        
        df_res = self.X_s2.copy()
        df_res['target'] = self.y_s2
        df_res['pred_ens'] = self.oof_ens
        df_res['pred_lgb'] = self.oof_lgb
        df_res['pred_cat'] = self.oof_cat
        df_res['pred_tab'] = self.oof_tab
        
        # 1. Uncertainty (迷い) : 0.4 < prob < 0.6
        df_res['is_uncertain'] = (df_res['pred_ens'] > 0.4) & (df_res['pred_ens'] < 0.6)
        
        # 2. Disagreement (意見割れ) : 標準偏差が大きい上位10%
        # 3モデルの予測値の分散を計算
        preds_stack = np.vstack([self.oof_lgb, self.oof_cat, self.oof_tab])
        df_res['model_std'] = np.std(preds_stack, axis=0)
        high_var_thresh = np.percentile(df_res['model_std'], 90)
        df_res['is_disagreement'] = df_res['model_std'] > high_var_thresh
        
        # 3. Error (間違ったもの) with standard threshold 0.5
        df_res['pred_binary'] = (df_res['pred_ens'] >= 0.5).astype(int)
        df_res['is_fp'] = (df_res['target'] == 0) & (df_res['pred_binary'] == 1)
        df_res['is_fn'] = (df_res['target'] == 1) & (df_res['pred_binary'] == 0)
        
        # カテゴリ作成: "Hard Sample"
        # 定義: 「迷っている」 または 「意見が割れている」 または 「間違えた(FP)」
        # FNは「見逃し」なので少し性質が違うが、今回は含める
        df_res['is_hard'] = df_res['is_uncertain'] | df_res['is_disagreement'] | df_res['is_fp'] | df_res['is_fn']

        n_hard = df_res['is_hard'].sum()
        print(f"   Hard Samples Identified: {n_hard:,} / {len(df_res):,} ({n_hard/len(df_res):.1%})")
        
        # 分析: Hard Sampleに特徴的なカテゴリは何か？
        self._analyze_categorical_bias(df_res, 'is_hard', "Hard Sample")
        self._analyze_categorical_bias(df_res, 'is_fp', "False Positive (誤検知)")
        self._analyze_categorical_bias(df_res, 'is_disagreement', "Disagreement (意見割れ)")

    def _analyze_categorical_bias(self, df, flag_col, title):
        """特定のフラグが立っているデータに偏っているカテゴリを探す"""
        print(f"\n📊 --- {title} の特徴分析 ---")
        target_cols = ['昼夜', '天候', '地形', '路面状態', '道路形状', '信号機', '事故類型']
        
        report_lines = []
        report_lines.append(f"### {title} の特徴的パターン\n")
        
        for col in target_cols:
            if col not in df.columns: continue
            
            # 全体分布
            overall_dist = df[col].value_counts(normalize=True)
            # ターゲット分布
            target_dist = df[df[flag_col] == True][col].value_counts(normalize=True)
            
            # 差分が大きいカテゴリを探す
            diff = target_dist - overall_dist
            
            # 重要度スコア (差分の絶対値の合計)
            importance = diff.abs().sum()
            
            if importance > 0.05: # ある程度差がある場合のみ表示
                print(f"   category: {col}")
                report_lines.append(f"#### {col}")
                # 特徴的な値トップ3
                top_diffs = diff.abs().sort_values(ascending=False).head(3)
                for val in top_diffs.index:
                    d = diff[val]
                    if abs(d) > 0.02: # 2%以上の乖離
                        direction = "多い (Over-represented)" if d > 0 else "少ない (Under-represented)"
                        msg = f"      - **{val}**: {target_dist.get(val, 0):.1%} (全体比 {d:+.1%}) -> {direction}"
                        print(msg)
                        report_lines.append(msg)
                report_lines.append("")

        # レポート保存
        file_name = f"hard_sample_analysis_{flag_col}.md"
        with open(os.path.join(self.output_dir, file_name), 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))

if __name__ == "__main__":
    analyzer = HardSampleAnalyzer()
    analyzer.load_data_and_predictions()
    analyzer.analyze_hard_samples()
