import pandas as pd

try:
    df = pd.read_csv('data/raw/honhyo_all_shishasuu_binary.csv', nrows=1, encoding='cp932')
    print("Columns (cp932):", df.columns.tolist())
except Exception as e:
    print("cp932 failed:", e)
    try:
        df = pd.read_csv('data/raw/honhyo_all_shishasuu_binary.csv', nrows=1, encoding='utf-8')
        print("Columns (utf-8):", df.columns.tolist())
    except Exception as e2:
        print("utf-8 failed:", e2)
