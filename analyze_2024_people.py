import pandas as pd

# CSVファイルを読み込む（複数のエンコードを試行）
csv_path = r'honhyo\honhyo_2024_UTF-8.csv'

encodings = ['utf-8', 'shift-jis', 'cp932', 'utf-8-sig']
df = None

for encoding in encodings:
    try:
        df = pd.read_csv(csv_path, encoding=encoding)
        print(f"✓ エンコード '{encoding}' で正常に読み込みました")
        break
    except UnicodeDecodeError:
        print(f"✗ エンコード '{encoding}' で読み込み失敗")
        continue

if df is None:
    raise Exception("CSVファイルを読み込めませんでした")

# データの基本情報を表示
print("=" * 80)
print("データセットの基本情報")
print("=" * 80)
print(f"総レコード数: {len(df):,}")
print(f"\nカラム数: {len(df.columns)}")
print(f"\nカラム名:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

# 2024年の交通事故に関与している人数を計算
# 各行が1件の事故を表し、関与者数に関するカラムを探す
print("\n" + "=" * 80)
print("2024年の交通事故に関与している人数の分析")
print("=" * 80)

# 可能性のある関与者数カラムを探す
person_related_columns = [col for col in df.columns if '当事者' in col or '人数' in col or '人員' in col]
print(f"\n関与者・人数関連のカラム:")
for col in person_related_columns:
    print(f"  - {col}")

# データの最初の数行を表示して構造を確認
print("\n" + "=" * 80)
print("データの先頭5行（関連カラムのみ）")
print("=" * 80)
if person_related_columns:
    print(df[person_related_columns].head(10))
else:
    # 関連カラムが見つからない場合は全カラムの最初の数行を表示
    print("関連カラムが見つからなかったため、すべてのカラムを表示:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.head(3))

# 一般的な解釈：各行が1人の関与者を表す場合
print("\n" + "=" * 80)
print("分析結果")
print("=" * 80)
print(f"\n【方法1】各行が1人の関与者を表す場合")
print(f"  → 2024年の交通事故関与者総数: {len(df):,}人")

# もし死傷者数などの合計カラムがあれば計算
if '死傷者数' in df.columns:
    total_casualties = df['死傷者数'].sum()
    print(f"\n【方法2】死傷者数の合計")
    print(f"  → 死傷者総数: {total_casualties:,}人")

# ユニークな事故番号がある場合
accident_id_columns = [col for col in df.columns if '本番号' in col or '事故番号' in col or '番号' in col]
if accident_id_columns:
    print(f"\n【参考】事故番号関連のカラム: {accident_id_columns}")
    unique_accidents = df[accident_id_columns[0]].nunique()
    print(f"  → ユニークな事故数: {unique_accidents:,}件")
    print(f"  → 1事故あたりの平均関与者数: {len(df) / unique_accidents:.2f}人")

print("\n" + "=" * 80)
