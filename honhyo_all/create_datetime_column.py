"""honhyo_all.csv の発生日時関連カラムを単一の datetime カラムへ統合するスクリプト。"""
from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Dict

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
INPUT_PATH = PROJECT_ROOT / "honhyo" / "honhyo_all.csv"
OUTPUT_PATH = BASE_DIR / "honhyo_all_with_datetime.csv"
NEW_COLUMN_NAME = "発生日時"
CHUNK_SIZE = 200_000

DATETIME_SOURCE_COLUMNS: Dict[str, str] = {
    "year": "発生日時　　年",
    "month": "発生日時　　月",
    "day": "発生日時　　日",
    "hour": "発生日時　　時",
    "minute": "発生日時　　分",
}


def build_datetime_series(chunk: pd.DataFrame) -> pd.Series:
    """必要なカラムから pandas の datetime64[ns] Series を作成する。"""
    components = {
        part: pd.to_numeric(chunk[column], errors="coerce")
        for part, column in DATETIME_SOURCE_COLUMNS.items()
    }
    components["second"] = 0
    return pd.to_datetime(pd.DataFrame(components), errors="coerce")


def transform_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.copy()
    chunk[NEW_COLUMN_NAME] = build_datetime_series(chunk)
    chunk.drop(columns=DATETIME_SOURCE_COLUMNS.values(), inplace=True)
    return chunk


def ensure_required_columns() -> None:
    """入力ファイルに必要なカラムが存在するか確認する。"""
    preview = pd.read_csv(INPUT_PATH, nrows=1)
    missing = [col for col in DATETIME_SOURCE_COLUMNS.values() if col not in preview.columns]
    if missing:
        raise KeyError(f"必要なカラムが見つかりません: {missing}")


def main() -> None:
    print(f"入力ファイル: {INPUT_PATH}")
    print(f"出力ファイル: {OUTPUT_PATH}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"入力ファイルが存在しません: {INPUT_PATH}")

    ensure_required_columns()

    first_chunk = True
    for chunk in pd.read_csv(INPUT_PATH, chunksize=CHUNK_SIZE, low_memory=False):
        transformed = transform_chunk(chunk)
        transformed.to_csv(
            OUTPUT_PATH,
            mode="w" if first_chunk else "a",
            index=False,
            header=first_chunk,
            encoding="utf-8-sig",
        )
        first_chunk = False

    print("発生日時カラムの統合が完了しました。")


if __name__ == "__main__":
    main()
