import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import os

def analyze_thresholds():
    print("# Threshold Analysis Report\n")
    
    # Paths
    path_tabnet = 'results/oof/oof_stage2_tabnet.csv'
    path_lgb = 'results/oof/oof_stage2_lightgbm.csv'
    
    if not os.path.exists(path_tabnet) or not os.path.exists(path_lgb):
        print("Error: OOF files not found.")
        return

    # Load Data
    df_tab = pd.read_csv(path_tabnet)
    df_lgb = pd.read_csv(path_lgb)
    
    # Merge
    df = pd.merge(
        df_tab.rename(columns={'prob': 'prob_tab', 'true_label': 'label_tab'}),
        df_lgb.rename(columns={'prob': 'prob_lgb', 'true_label': 'label_lgb'}),
        on='index',
        how='inner'
    )
    
    y_true = df['label_tab'].values
    prob_tab = df['prob_tab'].values
    prob_lgb = df['prob_lgb'].values
    
    # Best Output Ensemble: TabNet 90% + LGB 10%
    w_lgb = 0.1
    prob_ens = w_lgb * prob_lgb + (1 - w_lgb) * prob_tab
    
    print(f"**Target Model**: Ensemble (TabNet * 0.9 + LightGBM * 0.1)")
    print(f"**Data Count**: {len(df):,}\n")

    # Threshold Table
    print("| Threshold | F1 Score | Precision | Recall | TP | FP | FN | TN |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    
    best_f1 = -1
    best_thresh = -1
    
    # Broad search
    thresholds = np.arange(0.05, 1.00, 0.05)
    
    for th in thresholds:
        pred = (prob_ens >= th).astype(int)
        
        f1 = f1_score(y_true, pred)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred)
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        
        is_best = ""
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th
        
        print(f"| {th:.2f} | {f1:.4f} | {prec:.4f} | {rec:.4f} | {tp} | {fp} | {fn} | {tn} |")

    print(f"\n**Best F1**: {best_f1:.4f} at Threshold **{best_thresh:.2f}**")
    
    # Detailed check around best
    if best_thresh > 0:
        print(f"\n### Fine-tuning around {best_thresh:.2f}")
        print("| Threshold | F1 Score | Precision | Recall |")
        print("| :--- | :--- | :--- | :--- |")
        start = max(0, best_thresh - 0.04)
        end = min(1, best_thresh + 0.05)
        for th in np.arange(start, end, 0.01):
            pred = (prob_ens >= th).astype(int)
            f1 = f1_score(y_true, pred)
            prec = precision_score(y_true, pred, zero_division=0)
            rec = recall_score(y_true, pred)
            print(f"| {th:.2f} | {f1:.4f} | {prec:.4f} | {rec:.4f} |")

if __name__ == "__main__":
    analyze_thresholds()
