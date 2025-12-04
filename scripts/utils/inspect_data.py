import pandas as pd
import io

file_path = r'c:\Users\socce\software-lab\traffic-accident\honhyo_all\csv\honhyo_all_shishasuu_binary.csv'

try:
    df = pd.read_csv(file_path, encoding='cp932') # Try cp932 first for Japanese Windows
except UnicodeDecodeError:
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        print("Could not read file with cp932 or utf-8")
        exit()

print("Columns:")
print(df.columns.tolist())
print("\nInfo:")
print(df.info())
print("\nHead:")
print(df.head())
print("\nMissing Values:")
print(df.isnull().sum())
