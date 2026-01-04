import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path

def main():
    base_dir = Path("results/twostage_spatiotemporal_ensemble")
    data_dir = Path("data/spatio_temporal")
    
    # Load predictions
    pred_path = base_dir / "final_submission_full.csv"
    if not pred_path.exists():
        print(f"File not found: {pred_path}")
        return

    print("Loading predictions...")
    preds = pd.read_csv(pred_path)
    
    # Load ground truth
    print("Loading ground truth...")
    # Use preprocessed_test.parquet as it guarantees alignment with the training script
    gt_path = data_dir / "preprocessed_test.parquet"
    # if not gt_path.exists():
    #      gt_path = data_dir / "test.parquet"

    gt = pd.read_parquet(gt_path)
    
    if 'fatal' not in gt.columns:
        print("Target 'fatal' not found in ground truth file.")
        return

    # Check alignment
    # preds should have 'original_index'
    if 'original_index' in preds.columns:
        # Merge to ensure alignment
        merged = gt.merge(preds, left_index=True, right_on='original_index', how='inner')
        y_true = merged['fatal'].values
        y_pred = merged['prob_ensemble'].values
    else:
        # Assume aligned if lengths match
        if len(preds) != len(gt):
            print(f"Length mismatch: Preds {len(preds)}, GT {len(gt)}")
            return
        y_true = gt['fatal'].values
        y_pred = preds['prob_ensemble'].values

    print(f"Evaluating on {len(y_true)} samples...")
    
    auc = roc_auc_score(y_true, y_pred)
    prauc = average_precision_score(y_true, y_pred)
    
    print(f"Full Test AUC: {auc:.4f}")
    print(f"Full Test PR-AUC: {prauc:.4f}")

if __name__ == "__main__":
    main()
