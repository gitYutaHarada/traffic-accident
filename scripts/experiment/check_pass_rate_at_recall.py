import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, recall_score

def check_pass_rates():
    print("Loading data...")
    # Load data
    df = pd.read_csv("data/processed/stage1_oof_predictions.csv")
    
    # Check target column name
    target_col = 'fatal'
    if 'fatal' not in df.columns and 'target' in df.columns:
        target_col = 'target'
    elif '死者数' in df.columns:
        target_col = '死者数'
        
    y = df[target_col].values
    
    # Weighted Ensemble
    prob = 0.85 * df['prob_catboost'] + 0.15 * df['prob_lgbm']
    
    precision, recall, thresholds = precision_recall_curve(y, prob)
    
    targets = [0.999, 0.995, 0.990, 0.980, 0.950, 0.900]
    total_positives = y.sum()
    total_samples = len(y)
    
    print(f"\nTotal Samples: {total_samples:,}")
    print(f"Total Fatal Accidents: {total_positives:,}")
    print("-" * 80)
    print(f"{'Target Recall':<15} {'Act Recall':<12} {'Threshold':<10} {'Pass Rate':<10} {'Filtered':<10} {'Missed Fatal':<12}")
    print("-" * 80)
    
    for target in targets:
        # Find threshold
        valid_indices = np.where(recall[:-1] >= target)[0]
        if len(valid_indices) > 0:
            best_idx = valid_indices[-1]
            thresh = thresholds[best_idx]
            actual_rec = recall[best_idx]
        else:
            thresh = 0.0
            actual_rec = 1.0
            
        # Calculate stats
        mask = prob >= thresh
        pass_rate = mask.mean()
        filter_rate = 1 - pass_rate
        missed = int(total_positives - (mask * y).sum()) # More accurate missed count
        
        print(f"{target*100:>5.1f}%          {actual_rec*100:>6.2f}%      {thresh:.4f}     {pass_rate*100:>5.1f}%      {filter_rate*100:>5.1f}%      {missed:>4,}")

if __name__ == "__main__":
    check_pass_rates()
