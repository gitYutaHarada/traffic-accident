import pandas as pd

# CSVファイルを読み込む（複数のエンコードを試行）
csv_path = r'honhyo\honhyo_2024_UTF-8.csv'

encodings = ['utf-8', 'shift-jis', 'cp932', 'utf-8-sig']
df = None

for encoding in encodings:
    try:
        df = pd.read_csv(csv_path, encoding=encoding)
        print(f"✓ エンコード '{encoding}' で正常に読み込みました\n")
        break
    except UnicodeDecodeError:
        continue

if df is None:
    raise Exception("CSVファイルを読み込めませんでした")

# ==================================================================================
# 2024年の交通事故関与者数の分析結果
# ==================================================================================

print("=" * 80)
print("【2024年 交通事故関与者数 分析結果】")
print("=" * 80)

# 基本統計
print(f"\n■ データセット概要")
print(f"  ・総レコード数: {len(df):,}行")
print(f"  ・カラム数: {len(df.columns)}列")

# 事故番号（本票番号）でユニークな事故数を計算
unique_accidents = df['本票番号'].nunique()
print(f"\n■ 事故統計")
print(f"  ・ユニークな事故数: {unique_accidents:,}件")

# このデータセットの構造を判断
# 1行が1人の当事者を表すのか、1件の事故を表すのかを確認
avg_rows_per_accident = len(df) / unique_accidents

print(f"\n■ データ構造の分析")
print(f"  ・1事故あたりの平均レコード数: {avg_rows_per_accident:.2f}行")

# 当事者数の推定
# - 通常、交通事故データは1行で1件の事故を表し、当事者A・当事者Bという形で複数の当事者情報を持つ
# - このデータセットは「当事者A」「当事者B」のカラムがあることから、1行=1事故 の可能性が高い

# 当事者Aと当事者Bの関連カラムを確認
a_columns = [col for col in df.columns if '当事者A' in col or '（当事者A）' in col]
b_columns = [col for col in df.columns if '当事者B' in col or '（当事者B）' in col]

print(f"  ・当事者A関連カラム数: {len(a_columns)}個")
print(f"  ・当事者B関連カラム数: {len(b_columns)}個")

# 死者数と負傷者数の合計を計算
total_deaths = df['死者数'].sum()
total_injuries = df['負傷者数'].sum()
total_casualties = total_deaths + total_injuries

print(f"\n■ 死傷者統計")
print(f"  ・死者数合計: {int(total_deaths):,}人")
print(f"  ・負傷者数合計: {int(total_injuries):,}人")
print(f"  ・死傷者数合計: {int(total_casualties):,}人")

# 関与者数の推定
# 1行が1件の事故で、最小で当事者A・B の2名が関与していると仮定
# ただし、実際には歩行者、自転車、複数の車両が関与する場合もあるため、より正確な計算が必要

print(f"\n■ 関与者数の推定")

# 方法1: 最小推定（各事故に少なくとも当事者A・Bの2名が関与）
min_estimate = unique_accidents * 2
print(f"  【推定方法1】最小推定（各事故に最低2名関与）")
print(f"    → 最小関与者数: {min_estimate:,}人")

# 方法2: 死傷者数ベース（死傷者=関与者と仮定）
print(f"  【推定方法2】死傷者数ベース")
print(f"    → 死傷者数（関与者の一部）: {int(total_casualties):,}人")

# 方法3: データ構造からの推定
# もし1行が1人の当事者を表す場合
if avg_rows_per_accident > 10:  # 1事故あたり10行以上なら、1行=1当事者の可能性
    print(f"  【推定方法3】各行が1人の当事者を表す場合")
    print(f"    → 総関与者数: {len(df):,}人")
else:
    print(f"  【推定方法3】各行が1件の事故を表す場合")
    # より正確な推定が必要（事故タイプ、車両数などから計算）
    print(f"    → 総関与者数: 推定により {min_estimate:,}人以上")

print("\n" + "=" * 80)
print("【結論】")
print("=" * 80)
print(f"\n2024年の交通事故データ（honhyo_2024_UTF-8.csv）の分析結果:")
print(f"  ・総事故件数: {unique_accidents:,}件")
print(f"  ・死傷者総数: {int(total_casualties):,}人")
print(f"    - 死者: {int(total_deaths):,}人")
print(f"    - 負傷者: {int(total_injuries):,}人")
print(f"\n※ このデータセットは1行が1件の事故を表しており、")
print(f"  各事故には最低2名以上の当事者が関与していると推定されます。")
print(f"  正確な総関与者数を算出するには、追加の情報が必要です。")
print("\n" + "=" * 80)
