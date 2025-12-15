"""
アソシエーション分析用データ前処理スクリプト

交通事故データをアソシエーション分析(Apriori)用のトランザクション形式に変換します。
死亡事故に繋がる要因を発見することを目的としています。
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def load_data(file_path: str) -> pd.DataFrame:
    """データを読み込む"""
    print(f"データを読み込み中: {file_path}")
    df = pd.read_csv(file_path)
    print(f"読み込み完了: {len(df):,} 件のレコード")
    return df


def create_fatal_accident_flag(df: pd.DataFrame) -> pd.DataFrame:
    """死亡事故フラグを作成"""
    df = df.copy()
    df['fatal_accident'] = (df['死者数'] > 0).astype(int)
    
    fatal_count = df['fatal_accident'].sum()
    fatal_rate = fatal_count / len(df) * 100
    print(f"\n死亡事故: {fatal_count:,} 件 ({fatal_rate:.2f}%)")
    
    return df


def categorize_age(age: float) -> str:
    """年齢を年代に変換"""
    if pd.isna(age):
        return "年齢不明"
    elif age < 20:
        return "10代以下"
    elif age < 30:
        return "20代"
    elif age < 40:
        return "30代"
    elif age < 50:
        return "40代"
    elif age < 60:
        return "50代"
    elif age < 70:
        return "60代"
    else:
        return "70代以上"


def categorize_time(datetime_str: str) -> str:
    """時刻を時間帯に変換"""
    try:
        dt = pd.to_datetime(datetime_str)
        hour = dt.hour
        
        if 0 <= hour < 6:
            return "深夜"
        elif 6 <= hour < 9:
            return "早朝"
        elif 9 <= hour < 12:
            return "午前"
        elif 12 <= hour < 15:
            return "昼"
        elif 15 <= hour < 18:
            return "夕方"
        elif 18 <= hour < 21:
            return "夜"
        else:
            return "夜間"
    except:
        return "時刻不明"


def categorize_weather(weather_code: int) -> str:
    """天候コードを天候名に変換"""
    weather_map = {
        1: "晴れ",
        2: "曇り",
        3: "雨",
        4: "霧",
        5: "雪",
    }
    return weather_map.get(weather_code, "その他天候")


def categorize_road_shape(road_shape_code: int) -> str:
    """道路形状コードを道路形状名に変換"""
    road_shape_map = {
        1: "交差点",
        7: "カーブ",
        11: "トンネル",
        12: "橋",
        13: "踏切",
        14: "直線",
    }
    return road_shape_map.get(road_shape_code, "その他道路形状")


def categorize_day_night(day_night_code: int) -> str:
    """昼夜コードを昼夜名に変換"""
    day_night_map = {
        11: "昼",
        12: "昼",
        21: "夜",
        22: "夜",
        23: "夜",
    }
    return day_night_map.get(day_night_code, "不明")


def categorize_road_surface(road_surface_code: int) -> str:
    """路面状態コードを路面状態名に変換"""
    road_surface_map = {
        1: "乾燥",
        2: "湿潤",
        3: "積雪",
        4: "凍結",
    }
    return road_surface_map.get(road_surface_code, "その他路面")


def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量を前処理してカテゴリ化"""
    df = df.copy()
    
    print("\n特徴量を前処理中...")
    
    # 年代
    df['年代'] = df['年齢（当事者A）'].apply(categorize_age)
    
    # 時間帯
    df['時間帯'] = df['発生日時'].apply(categorize_time)
    
    # 天候
    df['天候名'] = df['天候'].apply(categorize_weather)
    
    # 道路形状
    df['道路形状名'] = df['道路形状'].apply(categorize_road_shape)
    
    # 昼夜
    df['昼夜名'] = df['昼夜'].apply(categorize_day_night)
    
    # 路面状態
    df['路面状態名'] = df['路面状態'].apply(categorize_road_surface)
    
    # 曜日マッピング
    weekday_map = {
        1: "月曜",
        2: "火曜",
        3: "水曜",
        4: "木曜",
        5: "金曜",
        6: "土曜",
        7: "日曜",
    }
    df['曜日名'] = df['曜日(発生年月日)'].map(weekday_map).fillna("不明")
    
    # 死亡事故フラグ
    df['死亡事故'] = df['fatal_accident'].apply(lambda x: "死亡事故あり" if x == 1 else "死亡事故なし")
    
    print("前処理完了")
    
    return df


def create_transaction_data(df: pd.DataFrame, target_columns: list) -> list:
    """トランザクション形式のデータを作成"""
    print("\nトランザクション形式に変換中...")
    
    transactions = []
    for idx, row in df.iterrows():
        transaction = []
        for col in target_columns:
            value = row[col]
            if pd.notna(value) and value != "不明" and value != "その他":
                # カラム名と値を組み合わせてアイテム名を作成
                item = f"{col}={value}"
                transaction.append(item)
        
        if transaction:  # 空でないトランザクションのみ追加
            transactions.append(transaction)
        
        if (idx + 1) % 100000 == 0:
            print(f"  処理中: {idx + 1:,} / {len(df):,} 件")
    
    print(f"トランザクション作成完了: {len(transactions):,} 件")
    
    return transactions


def save_transactions(transactions: list, output_path: str):
    """トランザクションデータを保存"""
    print(f"\nトランザクションデータを保存中: {output_path}")
    
    # CSV形式で保存(各行がカンマ区切りのアイテムリスト)
    with open(output_path, 'w', encoding='utf-8') as f:
        for transaction in transactions:
            f.write(','.join(transaction) + '\n')
    
    print("保存完了")


def main():
    """メイン処理"""
    # パス設定
    data_path = project_root / "data" / "processed" / "honhyo_clean_road_type.csv"
    output_dir = project_root / "results" / "association_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # データ読み込み
    df = load_data(data_path)
    
    # 死亡事故フラグ作成
    df = create_fatal_accident_flag(df)
    
    # 特徴量前処理
    df = preprocess_features(df)
    
    # トランザクション作成に使用するカラム
    target_columns = [
        '年代',
        '時間帯',
        '天候名',
        '道路形状名',
        '昼夜名',
        '路面状態名',
        '曜日名',
        '死亡事故',
    ]
    
    # 全データのトランザクション作成
    print("\n=== 全データのトランザクション作成 ===")
    all_transactions = create_transaction_data(df, target_columns)
    save_transactions(all_transactions, output_dir / "transactions_all.csv")
    
    # 死亡事故のみのトランザクション作成
    print("\n=== 死亡事故のみのトランザクション作成 ===")
    df_fatal = df[df['fatal_accident'] == 1].copy()
    print(f"死亡事故データ: {len(df_fatal):,} 件")
    
    # 死亡事故データからは死亡事故フラグを除外(すべて同じ値なので)
    target_columns_fatal = [col for col in target_columns if col != '死亡事故']
    
    fatal_transactions = create_transaction_data(df_fatal, target_columns_fatal)
    save_transactions(fatal_transactions, output_dir / "transactions_fatal.csv")
    
    # 統計情報を保存
    stats = {
        '全レコード数': len(df),
        '死亡事故数': len(df_fatal),
        '死亡事故率': f"{len(df_fatal) / len(df) * 100:.2f}%",
        '全トランザクション数': len(all_transactions),
        '死亡事故トランザクション数': len(fatal_transactions),
    }
    
    stats_df = pd.DataFrame([stats]).T
    stats_df.columns = ['値']
    stats_path = output_dir / "preprocessing_stats.csv"
    stats_df.to_csv(stats_path, encoding='utf-8-sig')
    print(f"\n統計情報を保存: {stats_path}")
    print(stats_df)
    
    print("\n前処理が完了しました!")


if __name__ == "__main__":
    main()
