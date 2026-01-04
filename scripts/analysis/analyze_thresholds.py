"""
アンサンブルモデルの閾値分析スクリプト
====================================
目的:
1. Precision-Recall Curve (PR曲線) の描画
2. F1 Scoreが最大になる「最適閾値」の特定
3. 以下の3つのシナリオに基づく閾値提案
    - ステップ1: バランス重視 (Max F1)
    - ステップ2: 見逃し防止 (High Recall)
    - ステップ3: 確実性重視 (High Precision)

実行方法:
    python scripts/analysis/analyze_thresholds.py
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_curve, f1_score, auc, recall_score, precision_score
from scipy.optimize import minimize

# 日本語フォント設定（必要に応じて）
# sns.set(style="whitegrid")

class ThresholdAnalyzer:
    def __init__(
        self,
        data_path="data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv",
        target_col="fatal",
        ckpt_dir="results/ensemble_stage2/checkpoints",
        output_dir="results/ensemble_stage2",
        random_state=42,
        n_folds=5,
        n_seeds=3,
        undersample_ratio=2.0
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.ckpt_dir = ckpt_dir
        self.output_dir = output_dir
        self.random_state = random_state
        self.n_folds = n_folds
        self.n_seeds = n_seeds
        self.undersample_ratio = undersample_ratio
        
        self.tabnet_available = True  # 実験でTabNet使用済みと仮定

    def load_and_reconstruct_oof(self):
        """データ読み込みとOOF再構築"""
        print("📂 データを読み込み、OOFを再構築中...")
        
        # 1. データ読み込み & 分割
        df = pd.read_csv(self.data_path)
        y_all = df[self.target_col].values
        X_all = df.drop(columns=[self.target_col])
        if '発生日時' in X_all.columns:
            X_all = X_all.drop(columns=['発生日時'])
            
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, test_size=0.2, random_state=self.random_state, stratify=y_all
        )
        X_train = X_train.reset_index(drop=True)
        self.y_train = y_train  # CV用GT

        # 2. Stage 1 OOF 再構築
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
            
        # 3. Stage 1 Threshold & Recall
        # train_ensemble_stage2.py のロジックを再現 (Recall 99%ターゲット)
        stage1_recall_target = 0.99
        threshold_stage1 = 0.001
        for thresh in np.arange(0.50, 0.001, -0.005):
            y_pred = (oof_stage1 >= thresh).astype(int)
            recall = recall_score(y_train, y_pred)
            if recall >= stage1_recall_target:
                threshold_stage1 = thresh
                break
        
        stage2_mask = oof_stage1 >= threshold_stage1
        self.y_s2 = y_train[stage2_mask]
        print(f"   Stage 1 Threshold: {threshold_stage1:.4f}")
        print(f"   Stage 2 Data Count: {len(self.y_s2):,} (Positive: {self.y_s2.sum()})")

        # 4. Stage 2 OOF 再構築
        X_s2 = X_train[stage2_mask].reset_index(drop=True) # インデックスリセット重要
        # Stage 2のCV分割も再現が必要
        # train_ensemble_stage2.pyでは X_s2 に対して StratifiedKFold している
        skf_s2 = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        self.oof_lgb = np.zeros(len(self.y_s2))
        self.oof_cat = np.zeros(len(self.y_s2))
        self.oof_tab = np.zeros(len(self.y_s2))
        
        for fold, (train_idx, val_idx) in enumerate(skf_s2.split(X_s2, self.y_s2)):
            fold_dir = os.path.join(self.ckpt_dir, f"stage2_fold{fold}")
            
            p_lgb = os.path.join(fold_dir, "lgb_pred.npy")
            p_cat = os.path.join(fold_dir, "cat_pred.npy")
            p_tab = os.path.join(fold_dir, "tab_pred.npy")
            
            if os.path.exists(p_lgb): self.oof_lgb[val_idx] = np.load(p_lgb)
            if os.path.exists(p_cat): self.oof_cat[val_idx] = np.load(p_cat)
            if os.path.exists(p_tab): self.oof_tab[val_idx] = np.load(p_tab)

    def optimize_ensemble(self):
        """アンサンブル重みの再最適化"""
        print("⚖️ アンサンブル重み最適化中...")
        def loss_func(weights):
            weights = np.array(weights)
            weights = np.clip(weights, 0, 1)
            weights /= weights.sum() + 1e-8
            ens_proba = weights[0]*self.oof_lgb + weights[1]*self.oof_cat + weights[2]*self.oof_tab
            y_pred = (ens_proba >= 0.5).astype(int)
            return -f1_score(self.y_s2, y_pred) # F1最大化

        init_weights = [1/3, 1/3, 1/3]
        bounds = [(0.05, 0.9)] * 3
        constraints = {'type': 'eq', 'fun': lambda w: 1 - sum(w)}
        
        res = minimize(loss_func, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        self.weights = res.x / res.x.sum()
        print(f"   Optimal Weights: LGB={self.weights[0]:.3f}, Cat={self.weights[1]:.3f}, Tab={self.weights[2]:.3f}")
        
        self.oof_ensemble = (
            self.weights[0]*self.oof_lgb + 
            self.weights[1]*self.oof_cat + 
            self.weights[2]*self.oof_tab
        )

    def analyze_thresholds(self):
        """3つのシナリオに基づく閾値分析"""
        print("\n📊 閾値分析を実行中...")
        
        precisions, recalls, thresholds = precision_recall_curve(self.y_s2, self.oof_ensemble)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        
        # --- ステップ1: バランス重視 (Max F1) ---
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        best_thresh = thresholds[best_idx]
        best_prec = precisions[best_idx]
        best_rec = recalls[best_idx]
        
        # --- ステップ2: 見逃し防止 (High Recall) ---
        # Recall >= 0.98 を満たす中での最大Precision
        # Recallは降順になっていることが多いが、念のため検索
        # thresholdsは昇順, precisions/recallsはthresholdsに対応(最後の要素は1,0)
        # thresholdsの長さは len(precisions)-1
        
        # target_recall = 0.98
        # idx_recall = np.where(recalls[:-1] >= target_recall)[0]
        # if len(idx_recall) > 0:
        #     # Recall条件を満たす中で最大のPrecisionを持つインデックス（一般に閾値が高いほどPrecision高い）
        #     # thresholdsは昇順なので、条件を満たす最大のインデックスが最も高い閾値
        #     target_idx = idx_recall[-1] 
        # else:
        #     target_idx = 0
            
        # もっと単純に、RecallがX以上になるギリギリの閾値を探す
        def get_metrics_at_recall(target_recall):
            idx = np.where(recalls >= target_recall)[0]
            if len(idx) == 0: return 0, 0, 0, 0
            # idx[-1] が条件を満たす中で最も高い閾値（Precisionが高くなりやすい）
            i = idx[-1]
            # iがthresholdsの範囲外になる場合(最後)のケア
            th = thresholds[i] if i < len(thresholds) else 1.0
            return th, precisions[i], recalls[i], f1_scores[i]

        rec_th, rec_pre, rec_rec, rec_f1 = get_metrics_at_recall(0.98) # 98% Recall目標

        # --- ステップ3: 確実性重視 (High Precision) ---
        # Precision >= 0.80 などを狙う、あるいはF0.5スコア最大化など
        # ここでは「予算限定」→ Top 100件程度に絞るイメージだが、閾値としては
        # Precisionが急激に上がるポイント、またはPrecision 80%ライン
        
        def get_metrics_at_precision(target_precision):
            idx = np.where(precisions >= target_precision)[0]
            if len(idx) == 0: return 0, 0, 0, 0
            # idx[0] が条件を満たす最も低い閾値 (Recallが高くなりやすい)
            i = idx[0]
            th = thresholds[i] if i < len(thresholds) else 1.0
            return th, precisions[i], recalls[i], f1_scores[i]

        prec_th, prec_pre, prec_rec, prec_f1 = get_metrics_at_precision(0.80) 
        # もし80%に届かなければ最大Precision
        if prec_th == 0:
            max_p_idx = np.argmax(precisions)
            prec_th = thresholds[max_p_idx] if max_p_idx < len(thresholds) else 1.0
            prec_pre = precisions[max_p_idx]
            prec_rec = recalls[max_p_idx]
            prec_f1 = f1_scores[max_p_idx]

        # グラフ描画
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, label='Ensemble Model')
        plt.scatter(best_rec, best_prec, c='red', s=100, label=f'Max F1 (Th={best_thresh:.3f})', zorder=5)
        plt.scatter(rec_rec, rec_pre, c='orange', s=100, label=f'High Recall (Th={rec_th:.3f})', zorder=5)
        plt.scatter(prec_rec, prec_pre, c='green', s=100, label=f'High Precision (Th={prec_th:.3f})', zorder=5)
        
        plt.title('Precision-Recall Curve with Strategy Points')
        plt.xlabel('Recall (Detection Rate)')
        plt.ylabel('Precision (Hit Rate)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, 'threshold_analysis_pr_curve.png'))
        plt.close()

        # レポート出力
        report = f"""
# 閾値最適化 & 戦略分析レポート

## ステップ1: 閾値の最適化 (Max F1 Score)
**バランス重視**: 精度と検知率のバランスが最も良いポイント
- **Threshold**: {best_thresh:.4f}
- **F1 Score**: {best_f1:.4f}
- Precision: {best_prec:.4f}
- Recall: {best_rec:.4f}

## ステップ2: 見逃しを減らしたい (High Recall Strategy)
**警察パトロール重点箇所**: 「怪しい場所は全部検知」
- **Target Recall**: ~98%
- **Threshold**: {rec_th:.4f}
- **Precision**: {rec_pre:.4f}
- Recall: {rec_rec:.4f}
- F1 Score: {rec_f1:.4f}
*解説: Precisionが低い（{rec_pre:.2%}）ため、空振りが多いが、危険な場所の98%を網羅できる設定。*

## ステップ3: 確実な場所だけ知りたい (High Precision Strategy)
**予算限定・集中対策**: 「絶対に事故が起きる場所だけ」
- **Target Precision**: ~80% (または最大)
- **Threshold**: {prec_th:.4f}
- **Precision**: {prec_pre:.4f}
- Recall: {prec_rec:.4f}
- F1 Score: {prec_f1:.4f}
*解説: 検知数（Recall）は低い（{prec_rec:.2%}）が、警報が出た場所の{prec_pre:.2%}で実際に事故が発生している高確度設定。*
"""
        print(report)
        with open(os.path.join(self.output_dir, 'threshold_strategies.md'), 'w', encoding='utf-8') as f:
            f.write(report)

        return report

if __name__ == "__main__":
    analyzer = ThresholdAnalyzer()
    analyzer.load_and_reconstruct_oof()
    analyzer.optimize_ensemble()
    analyzer.analyze_thresholds()
