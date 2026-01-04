import pandas as pd

# Load a small sample of the processed data to inspect integration results
data_path = "data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv"

try:
    # Read a few rows
    df = pd.read_csv(data_path, nrows=100)
    
    # Check valid columns
    print("Available columns:", df.columns.tolist())
    
    # Define columns of interest for the example (check if they exist)
    traffic_candidates = ['traffic_24h', 'average_travel_speed', 'congestion_degree']
    hospital_candidates = ['distance_to_hospital_km', 'nearest_hospital_beds', 'nearest_hospital_disaster']
    
    traffic_cols = [c for c in traffic_candidates if c in df.columns]
    hospital_cols = [c for c in hospital_candidates if c in df.columns]
    
    print("--- Data Sample for Report Examples ---")
    
    if not traffic_cols or not hospital_cols:
        print("Missing target columns.")
    else:
        # Find a row with non-null values for these columns to use as an example
        sample_row = df.dropna(subset=traffic_cols + hospital_cols).iloc[0]
        
        print("Traffic Integration Example:")
        print(f"Location: ({sample_row.get('地点　緯度（北緯）', 'N/A')}, {sample_row.get('地点　経度（東経）', 'N/A')})")
        for col in traffic_cols:
            print(f"  - {col}: {sample_row[col]}")
            
        print("\nHospital Integration Example:")
        print(f"Location: ({sample_row.get('地点　緯度（北緯）', 'N/A')}, {sample_row.get('地点　経度（東経）', 'N/A')})")
        for col in hospital_cols:
            print(f"  - {col}: {sample_row[col]}")

except Exception as e:
    print(f"Error reading data: {e}")
