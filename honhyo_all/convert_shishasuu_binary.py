import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / './honhyo/honhyo_all.csv'
OUTPUT_PATH = BASE_DIR / 'honhyo_all_shishasuu_binary.csv'

# 死者数列の候補名（必要に応じて追加してください）
CANDIDATE_COLS = ['死者数', '死者計', '死者']

def find_shishasuu_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col in CANDIDATE_COLS:
            return col
    # 大文字小文字や前後空白の違いを吸収
    normalized = {str(col).strip().lower(): col for col in df.columns}
    for key, orig in normalized.items():
        for cand in CANDIDATE_COLS:
            if key == cand.lower():
                return orig
    raise KeyError('死者数列（候補: ' + ', '.join(CANDIDATE_COLS) + '）が見つかりませんでした。')


def main() -> None:
    print(f'入力ファイル: {INPUT_PATH}')
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f'入力ファイルが見つかりません: {INPUT_PATH}')

    # メモリ節約のため、まずは先頭数行だけ読んで列名を確認
    preview = pd.read_csv(INPUT_PATH, nrows=10)
    target_col = find_shishasuu_column(preview)
    print(f'検出された死者数列名: {target_col}')

    # 本読み込み & 変換
    # ファイルが大きい場合は chunksize で分割処理
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(INPUT_PATH, chunksize=100_000):
        # 0以外を1に変換（NaNも1扱いにしたくない場合は fillna(0) を入れるなど調整）
        chunk[target_col] = chunk[target_col].apply(lambda x: 0 if pd.isna(x) or x == 0 else 1)
        chunks.append(chunk)

    result = pd.concat(chunks, ignore_index=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f'出力ファイルを作成しました: {OUTPUT_PATH}')


def check_hassei_nichiji_type():
    print(f'入力ファイル: {INPUT_PATH}')
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f'入力ファイルが見つかりません: {INPUT_PATH}')

    # 先頭数行を読み込み
    preview = pd.read_csv(INPUT_PATH, nrows=10)
    if '発生日時' in preview.columns:
        print(f"発生日時列の型: {preview['発生日時'].dtype}")
    else:
        print("発生日時列が見つかりませんでした。")


if __name__ == '__main__':
    main()
    check_hassei_nichiji_type()
