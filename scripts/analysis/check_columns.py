import pandas as pd
from pathlib import Path

# CSVファイルを読み込む
csv_path = Path("data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv")
df = pd.read_csv(csv_path)

print("Columns:", df.columns.tolist())
print("-" * 20)
print(df.head(2))
print("-" * 20)
print("Shape:", df.shape)
if 'fatal' in df.columns:
    print("Fatal counts:", df['fatal'].value_counts())
elif '事故内容' in df.columns:
    print("事故内容 counts:", df['事故内容'].value_counts())
