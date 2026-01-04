
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def verify_on_filtered():
    # 1. Load Stage 1 Test Predictions (Baseline Filter Source)
    print("Loading Stage 1 Predictions...")
    stage1_preds = pd.read_csv('data/processed/stage1_test_predictions.csv')
    
    # Calculate Stage 1 Ensemble Prob
    # Weights from experiment_report: 0.85 * CatBoost + 0.15 * LightGBM
    stage1_prob = 0.85 * stage1_preds['prob_catboost'] + 0.15 * stage1_preds['prob_lgbm']
    
    # Filter Indices (Threshold 0.0645)
    threshold = 0.0645
    filtered_indices = stage1_preds[stage1_prob >= threshold]['original_index'].values
    print(f"Stage 1 Filtered Count: {len(filtered_indices):,} / {len(stage1_preds):,}")
    
    # 2. Load New Spatio-Temporal Test Predictions & Data
    print("\nLoading New Spatio-Temporal Data...")
    new_preds = pd.read_csv('results/spatio_temporal_ensemble/test_predictions.csv')
    new_raw_data = pd.read_parquet('data/spatio_temporal/raw_test.parquet')
    
    # Verify alignment
    if len(new_preds) != len(new_raw_data):
        raise ValueError(f"Length mismatch: Preds {len(new_preds)} vs Raw {len(new_raw_data)}")
        
    # Assign predictions to raw data to use its index
    new_raw_data['pred_ensemble'] = new_preds['ensemble'].values
    new_raw_data['pred_tabnet'] = new_preds['tabnet'].values
    new_raw_data['pred_catboost'] = new_preds['catboost'].values
    new_raw_data['target'] = (new_raw_data['fatal'] > 0).astype(int) # Ensure target exists
    
    # 3. Filter New Data by Intersection of Indices
    print(f"New Data Index Range: {new_raw_data.index.min()} - {new_raw_data.index.max()}")
    print(f"Filtered Indices Range: {filtered_indices.min()} - {filtered_indices.max()}")
    
    subset_df = new_raw_data[new_raw_data.index.isin(filtered_indices)]
    print(f"\nMatched Subset Count: {len(subset_df):,}")
    
    if len(subset_df) == 0:
        print("CRITICAL ERROR: No indices matched! Indices might have been reset.")
        return

    # 4. Compute Metrics
    print("\n=== Verification Results on Filtered Data (Same Test Set as Baseline) ===")
    
    metrics = {}
    for model in ['pred_ensemble', 'pred_tabnet', 'pred_catboost']:
        auc = roc_auc_score(subset_df['target'], subset_df[model])
        prauc = average_precision_score(subset_df['target'], subset_df[model])
        metrics[model] = {'AUC': auc, 'PR-AUC': prauc}
        print(f"{model}: PR-AUC = {prauc:.4f}, ROC-AUC = {auc:.4f}")
        
    return metrics

if __name__ == "__main__":
    verify_on_filtered()
