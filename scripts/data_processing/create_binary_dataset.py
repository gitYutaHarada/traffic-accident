"""
二値分類用データセット作成スクリプト

死者数列を以下のように変換:
- 死者数 = 0 → 0 (非死亡事故)
- 死者数 >= 1 → 1 (死亡事故)
"""
import pandas as pd
from pathlib import Path

# パス設定
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = BASE_DIR / "data" / "processed" / "honhyo_clean_no_leakage.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "honhyo_clean_binary.csv"

print("=" * 80)
print("二値分類用データセット作成")
print("=" * 80)

# データ読み込み
print(f"\n[1/4] データ読み込み中: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"[OK] {len(df):,}件, {len(df.columns)}列")

# 死者数の分布確認（変換前）
print("\n[2/4] 変換前の死者数分布:")
print(df['死者数'].value_counts().sort_index())

# 二値分類用に変換
print("\n[3/4] 死者数列を二値に変換中...")
df['死者数'] = (df['死者数'] >= 1).astype(int)

# 変換後の分布確認
print("\n[変換後] 死者数分布:")
value_counts = df['死者数'].value_counts().sort_index()
print(value_counts)
print(f"\n  0 (非死亡事故): {value_counts[0]:,}件 ({value_counts[0]/len(df)*100:.2f}%)")
print(f"  1 (死亡事故):   {value_counts[1]:,}件 ({value_counts[1]/len(df)*100:.2f}%)")
print(f"  不均衡比: {value_counts[0]/value_counts[1]:.1f}:1")

# 保存
print(f"\n[4/4] 保存中: {OUTPUT_FILE}")
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print("[OK] 保存完了")

print("\n" + "=" * 80)
print("完了")
print("=" * 80)
print(f"入力: {INPUT_FILE}")
print(f"出力: {OUTPUT_FILE}")
print(f"行数: {len(df):,}")
print(f"列数: {len(df.columns)}")
print("=" * 80)
