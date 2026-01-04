"""
MoEモデルの閾値分析スクリプト
===========================
目的:
1. MoEモデル (Expert A + Generalist) の統合スコアに基づくPR曲線を描画
2. F1 Scoreが最大になる「最適閾値」の特定
3. 3つのシナリオ (Max F1, High Recall, High Precision) に基づく閾値提案
4. 前回のアンサンブルモデルとの性能比較

実行方法:
    python scripts/analysis/analyze_moe_thresholds.py
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, auc, recall_score, precision_score

class MoEThresholdAnalyzer:
    def __init__(
        self,
        data_path="data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv",
        target_col="fatal",
        # ckpt_dir="results/ensemble_stage2/checkpoints", # Stage 1はこっち
        moe_ckpt_dir="results/moe_stage2/checkpoints", # Stage 2 MoEはこっち
        output_dir="results/moe_stage2",
        random_state=42,
        n_folds=5,
        n_seeds=3
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.output_dir = output_dir
        self.moe_ckpt_dir = moe_ckpt_dir
        self.random_state = random_state
        self.n_folds = n_folds
        self.n_seeds = n_seeds

    def create_domain_mask(self, df):
        """Urban判定 (MoEと同じロジック)"""
        def to_float(x):
            try: return float(x)
            except: return -1
        
        terrain = df['地形'].apply(to_float)
        signal = df['信号機'].apply(to_float)
        road = df['道路形状'].apply(to_float)
        
        return (terrain == 3) | (signal == 7) | (road == 13)

    def load_and_reconstruct_oof(self):
        """データ読み込みとMoE OOF再構築"""
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

        # --- Stage 1 OOF (Mask作成用) ---
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
        print(f"   Stage 2 Target Data: {len(self.y_s2):,} (Positive: {self.y_s2.sum()})")

        # --- Domain Split ---
        urban_mask = self.create_domain_mask(self.X_s2)
        X_urban = self.X_s2[urban_mask].reset_index(drop=True)
        y_urban = self.y_s2[urban_mask]
        X_general = self.X_s2[~urban_mask].reset_index(drop=True)
        y_general = self.y_s2[~urban_mask]

        # --- MoE Predictions Reconstruction ---
        # MoEの学習時と同じデータ分割順序で予測値を埋める必要がある
        # train_moe_stage2.pyでは X_urban, X_general それぞれに対して StratifiedKFold を適用している
        
        self.oof_moe_proba = np.zeros(len(self.y_s2)) # 統合された予測確率

        # Expert A (Urban)
        skf_urban = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_urban_ens = np.zeros(len(y_urban))

        for fold, (train_idx, val_idx) in enumerate(skf_urban.split(X_urban, y_urban)):
            fold_dir = os.path.join(self.moe_ckpt_dir, f"ExpertA_Urban_fold{fold}")
            p_lgb = np.load(os.path.join(fold_dir, "lgb_pred.npy"))
            p_cat = np.load(os.path.join(fold_dir, "cat_pred.npy"))
            # Ensemble (0.5:0.5)
            oof_urban_ens[val_idx] = 0.5 * p_lgb + 0.5 * p_cat
        
        # Generalist (Non-Urban)
        skf_general = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_general_ens = np.zeros(len(y_general))

        for fold, (train_idx, val_idx) in enumerate(skf_general.split(X_general, y_general)):
            fold_dir = os.path.join(self.moe_ckpt_dir, f"Generalist_NonUrban_fold{fold}")
            p_lgb = np.load(os.path.join(fold_dir, "lgb_pred.npy"))
            p_cat = np.load(os.path.join(fold_dir, "cat_pred.npy"))
            oof_general_ens[val_idx] = 0.5 * p_lgb + 0.5 * p_cat

        # 統合 (元のインデックスに対応させる)
        # self.X_s2 のインデックスと urban_mask を使う
        self.oof_moe_proba[urban_mask.values] = oof_urban_ens
        self.oof_moe_proba[~urban_mask.values] = oof_general_ens

    def analyze_thresholds(self):
        """3つのシナリオに基づく閾値分析"""
        print("\n📊 MoE 閾値分析を実行中...")
        
        precisions, recalls, thresholds = precision_recall_curve(self.y_s2, self.oof_moe_proba)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        
        # --- 1. Max F1 ---
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        best_thresh = thresholds[best_idx]
        best_prec = precisions[best_idx]
        best_rec = recalls[best_idx]
        
        # --- 2. High Recall (98%) ---
        def get_metrics_at_recall(target_recall):
            idx = np.where(recalls >= target_recall)[0]
            if len(idx) == 0: return 0, 0, 0, 0
            i = idx[-1]
            th = thresholds[i] if i < len(thresholds) else 1.0
            return th, precisions[i], recalls[i], f1_scores[i]

        rec_th, rec_pre, rec_rec, rec_f1 = get_metrics_at_recall(0.98)

        # --- 3. High Precision (80%) ---
        def get_metrics_at_precision(target_precision):
            idx = np.where(precisions >= target_precision)[0]
            if len(idx) == 0: return 0, 0, 0, 0
            i = idx[0]
            th = thresholds[i] if i < len(thresholds) else 1.0
            return th, precisions[i], recalls[i], f1_scores[i]

        prec_th, prec_pre, prec_rec, prec_f1 = get_metrics_at_precision(0.80) 
        if prec_th == 0: # 80%に届かない場合
            max_p_idx = np.argmax(precisions)
            prec_th = thresholds[max_p_idx] if max_p_idx < len(thresholds) else 1.0
            prec_pre = precisions[max_p_idx]
            prec_rec = recalls[max_p_idx]
            prec_f1 = f1_scores[max_p_idx]

        # 比較用データ (定数値として埋め込み: 前回step 628の結果)
        prev_res = {
            'Max F1': {'th': 0.3085, 'pre': 0.2407, 'rec': 0.2773, 'f1': 0.2577},
            'Hi Rec': {'th': 0.0100, 'pre': 0.0142, 'rec': 0.9801, 'f1': 0.0280},
            'Hi Pre': {'th': 0.5489, 'pre': 0.8011, 'rec': 0.0228, 'f1': 0.0444}
        }

        # グラフ描画
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, label='MoE Model', linewidth=2)
        # MoE Points
        plt.scatter(best_rec, best_prec, c='red', s=100, marker='*', label=f'MoE Max F1', zorder=5)
        plt.scatter(rec_rec, rec_pre, c='orange', s=100, marker='*', label=f'MoE Recall 98%', zorder=5)
        plt.scatter(prec_rec, prec_pre, c='green', s=100, marker='*', label=f'MoE Precision 80%', zorder=5)
        
        # Previous Points (参考としてプロット)
        plt.scatter(prev_res['Max F1']['rec'], prev_res['Max F1']['pre'], c='red', s=50, marker='o', alpha=0.5, label='Prev Max F1')
        plt.scatter(prev_res['Hi Rec']['rec'], prev_res['Hi Rec']['pre'], c='orange', s=50, marker='o', alpha=0.5, label='Prev Recall 98%')
        plt.scatter(prev_res['Hi Pre']['rec'], prev_res['Hi Pre']['pre'], c='green', s=50, marker='o', alpha=0.5, label='Prev Precision 80%')

        plt.title('Precision-Recall Curve: MoE vs Previous Ensemble')
        plt.xlabel('Recall (Detection Rate)')
        plt.ylabel('Precision (Hit Rate)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'moe_threshold_comparison_pr_curve.png'))
        plt.close()

        # レポート出力
        def diff_fmt(curr, prev):
            d = curr - prev
            icon = "🔺" if d > 0 else "🔻" if d < 0 else "➡️"
            return f"{curr:.4f} ({icon} {d:+.4f})"

        report = f"""
# MoEモデル 閾値最適化 & 比較レポート

## 前回のアンサンブルモデルとの比較

### ステップ1: 閾値の最適化 (Max F1 Score)
**バランス重視**設定における性能比較
| 指標 | 今回 (MoE) | 前回 (Ensemble) | 変化 |
| :--- | :--- | :--- | :--- |
| **Threshold** | {best_thresh:.4f} | {prev_res['Max F1']['th']:.4f} | - |
| **F1 Score** | {diff_fmt(best_f1, prev_res['Max F1']['f1'])} | {prev_res['Max F1']['f1']:.4f} | バランス性能 |
| **Precision** | {diff_fmt(best_prec, prev_res['Max F1']['pre'])} | {prev_res['Max F1']['pre']:.4f} | 適合率 |
| **Recall** | {diff_fmt(best_rec, prev_res['Max F1']['rec'])} | {prev_res['Max F1']['rec']:.4f} | 再現率 |

### ステップ2: 見逃しを減らしたい (High Recall: ~98%)
**警察パトロール重点箇所**設定
| 指標 | 今回 (MoE) | 前回 (Ensemble) | 変化 |
| :--- | :--- | :--- | :--- |
| **Threshold** | {rec_th:.4f} | {prev_res['Hi Rec']['th']:.4f} | - |
| **Precision** | {diff_fmt(rec_pre, prev_res['Hi Rec']['pre'])} | {prev_res['Hi Rec']['pre']:.4f} | 効率性 |
| **Recall** | {rec_rec:.4f} | {prev_res['Hi Rec']['rec']:.4f} | 目標達成 |

### ステップ3: 確実な場所だけ知りたい (High Precision: ~80%)
**予算限定・集中対策**設定
| 指標 | 今回 (MoE) | 前回 (Ensemble) | 変化 |
| :--- | :--- | :--- | :--- |
| **Threshold** | {prec_th:.4f} | {prev_res['Hi Pre']['th']:.4f} | - |
| **Precision** | {prec_pre:.4f} | {prev_res['Hi Pre']['pre']:.4f} | 目標達成 |
| **Recall** | {diff_fmt(prec_rec, prev_res['Hi Pre']['rec'])} | {prev_res['Hi Pre']['rec']:.4f} | 検知数 |

## 考察
*   **Precisionの壁**: Expert Aを分離したことで、High Precision領域での性能（Recallの伸び）がどう変化したかに注目してください。
*   **全体の底上げ**: Max F1の向上が見られれば、MoE戦略全体の成功と言えます。
"""
        print(report)
        with open(os.path.join(self.output_dir, 'moe_threshold_strategies.md'), 'w', encoding='utf-8') as f:
            f.write(report)

        return report

if __name__ == "__main__":
    analyzer = MoEThresholdAnalyzer()
    analyzer.load_and_reconstruct_oof()
    analyzer.analyze_thresholds()
