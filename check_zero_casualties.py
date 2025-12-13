import pandas as pd

# CSVファイルを読み込む
csv_path = r'honhyo\honhyo_2024_UTF-8.csv'

encodings = ['utf-8', 'shift-jis', 'cp932', 'utf-8-sig']
df = None

for encoding in encodings:
    try:
        df = pd.read_csv(csv_path, encoding=encoding)
        print(f"✓ エンコード '{encoding}' で読み込み成功\n")
        break
    except UnicodeDecodeError:
        continue

if df is None:
    raise Exception("CSVファイルを読み込めませんでした")

# 死者数と負傷者数がともにゼロの事故を検索
zero_casualties = df[(df['死者数'] == 0) & (df['負傷者数'] == 0)]

print("=" * 80)
print("【死者数・負傷者数がともにゼロの事故の分析】")
print("=" * 80)

print(f"\n■ 全体統計")
print(f"  ・総レコード数: {len(df):,}行")
print(f"  ・総事故数（本票番号でユニーク）: {df['本票番号'].nunique():,}件")

print(f"\n■ 死傷者ゼロの事故")
print(f"  ・死者数=0 かつ 負傷者数=0 のレコード数: {len(zero_casualties):,}行")

if len(zero_casualties) > 0:
    # 本票番号でユニークな事故数を計算
    unique_zero_casualty_accidents = zero_casualties['本票番号'].nunique()
    print(f"  ・死傷者ゼロのユニークな事故数: {unique_zero_casualty_accidents:,}件")
    
    # パーセンテージを計算
    percentage = (unique_zero_casualty_accidents / df['本票番号'].nunique()) * 100
    print(f"  ・全事故に占める割合: {percentage:.2f}%")
    
    # サンプルデータを表示
    print(f"\n■ サンプルデータ（最初の5件）")
    print(f"\n")
    sample_cols = ['本票番号', '死者数', '負傷者数', '事故内容', '発生日時　　年', '発生日時　　月', '発生日時　　日']
    available_cols = [col for col in sample_cols if col in zero_casualties.columns]
    print(zero_casualties[available_cols].head(5).to_string(index=False))
    
    # 事故内容の内訳
    if '事故内容' in zero_casualties.columns:
        print(f"\n■ 事故内容の内訳")
        accident_type_counts = zero_casualties['事故内容'].value_counts()
        for accident_type, count in accident_type_counts.items():
            print(f"  ・事故内容コード {accident_type}: {count:,}件")
else:
    print(f"\n  → 死者数・負傷者数がともにゼロの事故は存在しません。")

# 参考：死傷者数の分布
print(f"\n■ 参考：死傷者数の分布")
print(f"  ・死者数=0 のレコード数: {len(df[df['死者数'] == 0]):,}行")
print(f"  ・負傷者数=0 のレコード数: {len(df[df['負傷者数'] == 0]):,}行")
print(f"  ・死者数>0 または 負傷者数>0 のレコード数: {len(df[(df['死者数'] > 0) | (df['負傷者数'] > 0)]):,}行")

print("\n" + "=" * 80)
