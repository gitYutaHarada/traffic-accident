"""
Top-K Precision Calculation Script for Accident Prediction Models
"""
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score

def calc_top_k_precision(y_true, y_pred, k_list=[50, 100, 200, 500, 1000]):
    results = {}
    df = pd.DataFrame({'target': y_true, 'pred': y_pred})
    df = df.sort_values('pred', ascending=False)
    
    for k in k_list:
        top_k = df.head(k)
        precision = top_k['target'].mean() # 正解率
        hits = top_k['target'].sum()
        results[f'Top-{k}'] = f"{precision:.2%} ({int(hits)}/{k})"
    
    return results

def main():
    # Load OOF predictions
    oof_path = "results/tabnet_optimized/oof_predictions.csv"
    print(f"📂 Loading: {oof_path}")
    df = pd.read_csv(oof_path)
    
    models = ['lgbm', 'catboost', 'mlp', 'tabnet_optimized', 'ensemble']
    k_list = [50, 100, 200, 500, 1000]
    
    print(f"\n📊 Top-K Precision Analysis (N={len(df):,})")
    print("-" * 80)
    
    # Header
    header = f"{'Model':<20}" + "".join([f"{f'Top-{k}':<15}" for k in k_list])
    print(header)
    print("-" * 80)
    
    for model in models:
        if model not in df.columns:
            continue
            
        metrics = calc_top_k_precision(df['target'], df[model], k_list)
        
        row = f"{model:<20}"
        for k in k_list:
            row += f"{metrics[f'Top-{k}']:<15}"
        print(row)
    
    print("-" * 80)

if __name__ == "__main__":
    main()
