"""
Stage 2 アンサンブル評価スクリプト (TabNet + LightGBM)

評価項目:
- 重み付け最適化 (F1最大化)
- Recall 99%/98%/95% 時の Precision (安全要件)
- モデル間相関

注意: ここで計算される Precision は「Stage 2 単体」の値です。
      システム全体の Precision を計算するには Stage 1 の分母を考慮する必要があります。
"""

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, precision_score, recall_score
import os

def evaluate_ensemble():
    print("🌿 Stage 2: Ensemble Evaluation (TabNet + LightGBM)")
    print("   ※ 注意: Precision は Stage 2 単体の値です。システム全体の評価には Stage 1 を考慮してください。\n")
    
    path_tabnet = 'results/oof/oof_stage2_tabnet.csv'
    path_lgb = 'results/oof/oof_stage2_lightgbm.csv'
    
    if not os.path.exists(path_tabnet) or not os.path.exists(path_lgb):
        print(f"⚠️ OOF prediction files not found.")
        print(f"   TabNet: {os.path.exists(path_tabnet)}")
        print(f"   LightGBM: {os.path.exists(path_lgb)}")
        return
    
    # Load Data
    df_tab = pd.read_csv(path_tabnet)
    df_lgb = pd.read_csv(path_lgb)
    
    # Check consistency
    print(f"   TabNet OOF: {len(df_tab):,} rows")
    print(f"   LightGBM OOF: {len(df_lgb):,} rows")
    
    # [修正] indexのみでマージし、その後ラベルの一致を確認
    df = pd.merge(
        df_tab.rename(columns={'prob': 'prob_tab', 'true_label': 'label_tab'}),
        df_lgb.rename(columns={'prob': 'prob_lgb', 'true_label': 'label_lgb'}),
        on='index',
        how='inner'
    )
    print(f"   Aligned Data: {len(df):,} rows")
    
    # ラベル一致確認
    label_mismatch = (df['label_tab'] != df['label_lgb']).sum()
    if label_mismatch > 0:
        print(f"   ⚠️ 警告: ラベル不一致が {label_mismatch} 件あります！")
        return
    print(f"   ✅ ラベル一致確認完了")
    
    y_true = df['label_tab'].values
    prob_tab = df['prob_tab'].values
    prob_lgb = df['prob_lgb'].values
    
    # Check Correlation
    corr = np.corrcoef(prob_tab, prob_lgb)[0, 1]
    print(f"   📊 Correlation between models: {corr:.4f}")
    
    # Helper function: 特定のRecallを達成する閾値でのPrecisionを取得
    def get_precision_at_recall(prob, y_true, target_recall):
        precisions, recalls, thresholds = precision_recall_curve(y_true, prob)
        idx = np.where(recalls >= target_recall)[0]
        if len(idx) > 0:
            idx = idx[-1]  # 最も高いRecallを達成する最後のインデックス
            thresh = thresholds[idx] if idx < len(thresholds) else 0.0
            prec = precisions[idx]
            return thresh, prec
        return 0.0, 0.0
    
    # Search for Best Weight
    best_f1_score = -1
    best_f1_weight = -1
    best_f1_metrics = {}
    
    # 高Recall時の最適重みも追跡
    best_recall99_prec = -1
    best_recall99_weight = -1
    
    print("\n   🔍 Searching for best weight (w * LightGBM + (1-w) * TabNet)...")
    print(f"   {'Weight':<8} {'AUC':<8} {'F1':<8} {'Prec@R99':<10} {'Prec@R98':<10} {'Prec@R95':<10}")
    print("-" * 70)
    
    for w in np.arange(0.0, 1.01, 0.05):
        prob_ens = w * prob_lgb + (1 - w) * prob_tab
        
        # Calculate AUC
        auc = roc_auc_score(y_true, prob_ens)
        
        # Find Best F1 for this weight
        precisions, recalls, thresholds = precision_recall_curve(y_true, prob_ens)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-15)
        best_f1_idx = np.argmax(f1_scores)
        f1 = f1_scores[best_f1_idx]
        
        # [追加] Recall 99%/98%/95% 時の Precision
        _, prec_r99 = get_precision_at_recall(prob_ens, y_true, 0.99)
        _, prec_r98 = get_precision_at_recall(prob_ens, y_true, 0.98)
        _, prec_r95 = get_precision_at_recall(prob_ens, y_true, 0.95)
        
        print(f"   {w:.1f}      {auc:.4f}   {f1:.4f}   {prec_r99:.4f}      {prec_r98:.4f}      {prec_r95:.4f}")
        
        if f1 > best_f1_score:
            best_f1_score = f1
            best_f1_weight = w
            prec = precisions[best_f1_idx]
            rec = recalls[best_f1_idx]
            best_f1_metrics = {
                'auc': auc, 'f1': f1, 'precision': prec, 'recall': rec,
                'threshold': thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else 0.5
            }
        
        # 高Recall時の最適重みを追跡
        if prec_r99 > best_recall99_prec:
            best_recall99_prec = prec_r99
            best_recall99_weight = w
            
    print("-" * 70)
    
    # 最終結果表示
    print(f"\n🏆 Best F1 Ensemble (w_lgb={best_f1_weight:.1f})")
    print(f"   AUC: {best_f1_metrics['auc']:.4f}")
    print(f"   F1 Score: {best_f1_metrics['f1']:.4f}")
    print(f"   Precision: {best_f1_metrics['precision']:.4f}")
    print(f"   Recall: {best_f1_metrics['recall']:.4f}")
    print(f"   Threshold: {best_f1_metrics['threshold']:.4f}")
    
    # [追加] 高Recall時の評価
    print(f"\n🎯 High Recall Evaluation (w_lgb={best_recall99_weight:.1f})")
    prob_best = best_recall99_weight * prob_lgb + (1 - best_recall99_weight) * prob_tab
    
    for target_recall in [0.99, 0.98, 0.95]:
        thresh, prec = get_precision_at_recall(prob_best, y_true, target_recall)
        print(f"   Recall ≥ {target_recall:.0%}: Threshold={thresh:.4f}, Precision={prec:.4f}")
    
    # 単体モデルとの比較
    print("\n📊 単体モデルとの比較 (Recall ≥ 99%):")
    _, prec_tab_r99 = get_precision_at_recall(prob_tab, y_true, 0.99)
    _, prec_lgb_r99 = get_precision_at_recall(prob_lgb, y_true, 0.99)
    print(f"   TabNet:   Precision@R99 = {prec_tab_r99:.4f}")
    print(f"   LightGBM: Precision@R99 = {prec_lgb_r99:.4f}")
    print(f"   Ensemble: Precision@R99 = {best_recall99_prec:.4f}")

if __name__ == "__main__":
    evaluate_ensemble()
