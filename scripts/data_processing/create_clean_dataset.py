"""
データリーク防止のためのクリーンデータセット作成スクリプト

事後情報（事故発生後にのみわかる情報）を完全に除外し、
事前に観測可能な特徴量のみを含むデータセットを作成する。
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# パス設定
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = BASE_DIR / "honhyo_all" / "csv" / "honhyo_all_with_datetime.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "honhyo_clean_no_leakage.csv"

# 出力ディレクトリ作成
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("クリーンデータセット作成スクリプト")
print("=" * 80)

# データ読み込み
print(f"\n[1/5] データ読み込み中: {INPUT_FILE}")
try:
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    print(f"[OK] 読み込み完了: {len(df):,}件, {len(df.columns)}列")
except Exception as e:
    print(f"[ERROR] 読み込み失敗: {e}")
    sys.exit(1)

# 除外する列の完全リスト
print("\n[2/5] 事後情報列の定義")

# 事後情報列（事故発生後にのみ分かる情報）
POST_ACCIDENT_COLS = [
    # 最も致命的なリーク
    "事故内容",  # 死亡/負傷の区分そのもの（100%リーク）
    
    # 人的損傷（事故の結果）
    "人身損傷程度（当事者A）",  # 死亡・重傷・軽傷
    "人身損傷程度（当事者B）",
    
    # 負傷者・死者数（ターゲット変数に直接関連）
    "負傷者数",
    # "死者数" は目的変数なので別途処理
    
    # 車両の損壊（事故の重大性を示す事後情報）
    "車両の損壊程度（当事者A）",
    "車両の損壊程度（当事者B）",
    
    # 衝突部位（事故発生後にわかる情報）
    "車両の衝突部位（当事者A）",
    "車両の衝突部位（当事者B）",
    
    # エアバッグ（装備そのものは事前情報だが、作動状況は事後）
    # 安全のため除外
    "エアバッグの装備（当事者A）",
    "エアバッグの装備（当事者B）",
    "サイドエアバッグの装備（当事者A）",
    "サイドエアバッグの装備（当事者B）",
]

# 管理情報（分析に不要）
MANAGEMENT_COLS = [
    "資料区分",
    "本票番号",
]

# 除外する列の統合
DROP_COLS = POST_ACCIDENT_COLS + MANAGEMENT_COLS

print(f"除外する列: {len(DROP_COLS)}個")
for i, col in enumerate(DROP_COLS, 1):
    print(f"  {i:2d}. {col}")

# 除外前の検証
print("\n[3/5] 除外前の検証")
cols_in_data = [col for col in DROP_COLS if col in df.columns]
cols_not_in_data = [col for col in DROP_COLS if col not in df.columns]

print(f"[INFO] データに存在する除外対象列: {len(cols_in_data)}/{len(DROP_COLS)}")
if cols_not_in_data:
    print(f"[WARNING] データに存在しない列 ({len(cols_not_in_data)}個):")
    for col in cols_not_in_data:
        print(f"  - {col}")

# 列の除外
print("\n[4/5] 事後情報列の除外")
df_clean = df.drop(columns=cols_in_data)
print(f"[OK] 除外完了: {len(df_clean.columns)}列が残存")

# 除外後の検証（重要）
print("\n[5/5] 除外後の検証")
leaked_cols_remaining = [col for col in POST_ACCIDENT_COLS if col in df_clean.columns]

if leaked_cols_remaining:
    print("[ERROR] 除外に失敗した列が残っています:")
    for col in leaked_cols_remaining:
        print(f"  - {col}")
    print("\n[CRITICAL] データリークの可能性があります。処理を中止します。")
    sys.exit(1)
else:
    print("[OK] すべての事後情報列が正常に除外されました")

# データの基本情報
print("\n" + "=" * 80)
print("クリーンデータセットの概要")
print("=" * 80)
print(f"行数: {len(df_clean):,}")
print(f"列数: {len(df_clean.columns)}")
print(f"\n目的変数 '死者数' の分布:")
if "死者数" in df_clean.columns:
    print(df_clean["死者数"].value_counts().sort_index())
else:
    print("[WARNING] 目的変数 '死者数' が見つかりません")

# 保存
print(f"\n[SAVE] クリーンデータを保存中: {OUTPUT_FILE}")
df_clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"[OK] 保存完了")

# 最終確認
print("\n" + "=" * 80)
print("最終確認")
print("=" * 80)

# サンプルデータで検証
verification_df = pd.read_csv(OUTPUT_FILE, nrows=5)
leaked_check = [col for col in POST_ACCIDENT_COLS if col in verification_df.columns]

if leaked_check:
    print("[ERROR] 保存されたデータに事後情報列が含まれています:")
    print(leaked_check)
    sys.exit(1)
else:
    print("[OK] ✓ 保存されたデータに事後情報列は含まれていません")
    print("[OK] ✓ クリーンデータセットの作成が完了しました")

print("\n" + "=" * 80)
print("データセット情報")
print("=" * 80)
print(f"入力: {INPUT_FILE}")
print(f"出力: {OUTPUT_FILE}")
print(f"除外した列: {len(cols_in_data)}個")
print(f"残った列: {len(df_clean.columns)}個")
print(f"データサイズ: {len(df_clean):,}行")
print("=" * 80)
