import pandas as pd
import argparse
import os
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Export FP and TP rows from OOF predictions based on Recall threshold.")
    parser.add_argument("--target_recall", type=float, default=0.99, help="Target Recall to determine threshold (e.g. 0.99)")
    parser.add_argument("--oof_path", type=str, default=r"results/oof/oof_stage2_lightgbm.csv", help="Path to OOF CSV")
    parser.add_argument("--raw_path", type=str, default=r"data/raw/honhyo_all_shishasuu_binary.csv", help="Path to raw data CSV")
    parser.add_argument("--output_dir", type=str, default=r"results/analysis/fp_tp_export", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading OOF from {args.oof_path}...")
    if not os.path.exists(args.oof_path):
        print(f"Error: OOF file not found at {args.oof_path}")
        return

    df_oof = pd.read_csv(args.oof_path)
    
    if 'index' not in df_oof.columns or 'prob' not in df_oof.columns or 'true_label' not in df_oof.columns:
        raise ValueError("OOF file must have 'index', 'prob', and 'true_label' columns.")

    # Calculate probabilities statistics
    print("Probability Distribution (All):")
    print(df_oof['prob'].describe())

    # --- Threshold Calculation Logic ---
    y_true = df_oof['true_label'].values
    y_prob = df_oof['prob'].values
    
    # Extract probabilities for positive class
    pos_probs = y_prob[y_true == 1]
    print(f"\nPositive samples (Fatal Accidents): {len(pos_probs)}")
    
    if len(pos_probs) == 0:
        print("Error: No positive samples found in OOF data.")
        return

    # To capture X% of positives, we need the threshold to be at the (1-X) quantile of positive probabilities
    # e.g. Recall 0.99 -> We want to keep top 99% values -> threshold is at 1% quantile
    threshold_quantile = 1.0 - args.target_recall
    calculated_threshold = np.quantile(pos_probs, threshold_quantile)
    
    print(f"Target Recall: {args.target_recall}")
    print(f"Calculated Threshold: {calculated_threshold:.6f}")
    
    # Check actual recall with this threshold
    pred_label_at_thresh = (y_prob >= calculated_threshold).astype(int)
    tp_count = ((pred_label_at_thresh == 1) & (y_true == 1)).sum()
    actual_recall = tp_count / len(pos_probs)
    print(f"Actual Recall achieved: {actual_recall:.4f} ({tp_count}/{len(pos_probs)})")

    
    # --- Loading Raw Data for Merging ---
    print(f"\nLoading Raw Data from {args.raw_path}...")
    if not os.path.exists(args.raw_path):
        print(f"Error: Raw data file not found at {args.raw_path}")
        return

    df_raw = pd.read_csv(args.raw_path)

    print("Merging data...")
    # Set index to 'index' column for OOF to match raw data index
    df_oof = df_oof.set_index('index')
    
    # Validating indices
    common_indices = df_oof.index.intersection(df_raw.index)
    print(f"Found {len(common_indices)} common indices out of {len(df_oof)} OOF rows.")
    
    df_subset = df_raw.loc[common_indices].copy()
    
    # Add prediction features from OOF
    df_subset['pred_prob'] = df_oof.loc[common_indices, 'prob']
    df_subset['true_label'] = df_oof.loc[common_indices, 'true_label']
    
    # Calculate predictions with the determined threshold
    df_subset['pred_label'] = (df_subset['pred_prob'] >= calculated_threshold).astype(int)
    
    # Extract TP and FP
    # TP: Predicted Positive (1) AND Actual Positive (1)
    df_tp = df_subset[(df_subset['pred_label'] == 1) & (df_subset['true_label'] == 1)]
    
    # FP: Predicted Positive (1) AND Actual Negative (0)
    df_fp = df_subset[(df_subset['pred_label'] == 1) & (df_subset['true_label'] == 0)]
    
    print("-" * 30)
    print(f"Threshold used: {calculated_threshold:.6f}")
    print(f"True Positives (TP) count: {len(df_tp)}")
    print(f"False Positives (FP) count: {len(df_fp)}")
    if len(df_fp) > 0:
        precision = len(df_tp) / (len(df_tp) + len(df_fp))
        print(f"Precision at this threshold: {precision:.4f}")
    print("-" * 30)
    
    thresh_str = f"{calculated_threshold:.4f}".replace('.', '_')
    tp_file = os.path.join(args.output_dir, f"tp_recall_{args.target_recall}_thresh_{thresh_str}.csv")
    fp_file = os.path.join(args.output_dir, f"fp_recall_{args.target_recall}_thresh_{thresh_str}.csv")
    
    # Sort by probability descending
    df_tp = df_tp.sort_values('pred_prob', ascending=False)
    df_fp = df_fp.sort_values('pred_prob', ascending=False)

    print("Saving files...")
    df_tp.to_csv(tp_file, index=True)
    df_fp.to_csv(fp_file, index=True)
    
    print(f"✅ Saved TP list to: {tp_file}")
    print(f"✅ Saved FP list to: {fp_file}")

if __name__ == "__main__":
    main()
