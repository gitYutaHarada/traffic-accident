"""
Spatio-Temporal Stage2 前処理スクリプト
======================================
- ジオハッシュ生成
- 過去ウィンドウの事故件数集計（リーク防止）
- 時系列特徴量生成
- カテゴリエンコーディング
- 時間ベース分割
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import warnings
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import KFold
import geohash2 as geohash

warnings.filterwarnings('ignore')

# ランダムシード固定
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class SpatioTemporalPreprocessor:
    """空間・時系列特徴量の前処理クラス"""
    
    def __init__(
        self,
        data_path: str = "data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv",
        output_dir: str = "data/spatio_temporal",
        target_col: str = "fatal",
        train_years: Tuple[int, int] = (2018, 2019),
        val_years: Tuple[int, int] = (2020, 2020),
        test_years: Tuple[int, int] = (2021, 2024),
        geohash_precision: int = 6,
        past_windows: List[int] = [30, 365],
        high_cardinality_threshold: int = 20,
    ):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.target_col = target_col
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.geohash_precision = geohash_precision
        self.past_windows = past_windows
        self.high_cardinality_threshold = high_cardinality_threshold
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # エンコーダ類
        self.scaler = StandardScaler()
        self.ohe = None
        self.target_encoders = {}
        
        print("=" * 70)
        print("Spatio-Temporal 前処理パイプライン")
        print(f"Train: {train_years[0]}-{train_years[1]}")
        print(f"Val:   {val_years[0]}-{val_years[1]}")
        print(f"Test:  {test_years[0]}-{test_years[1]}")
        print("=" * 70)
    
    def load_data(self) -> pd.DataFrame:
        """データ読み込み"""
        print("\n📂 データ読み込み中...")
        df = pd.read_csv(self.data_path)
        print(f"   元データ: {len(df):,} 行, {len(df.columns)} 列")
        return df
    
    def convert_lat_lon(self, df: pd.DataFrame) -> pd.DataFrame:
        """緯度経度を度数法に変換（ベクトル処理、ロバスト版）"""
        print("\n🌍 緯度経度変換中...")
        
        lat_col = '地点　緯度（北緯）'
        lon_col = '地点　経度（東経）'
        
        def convert_coord_vectorized(series):
            """ベクトル化された座標変換（混在データ対応）"""
            result = pd.Series(index=series.index, dtype=float)
            
            # 欠損値を除外
            valid_mask = series.notna()
            valid_vals = series[valid_mask].astype(float)
            
            # 整数形式（>1000000）と度数法（<1000）を判別
            is_integer_format = valid_vals > 1000000
            
            # 整数形式の変換 (dddmmssss)
            int_vals = valid_vals[is_integer_format].astype(int)
            deg = int_vals // 10000000
            remainder = int_vals % 10000000
            minutes = remainder // 100000
            seconds = (remainder % 100000) / 1000
            result.loc[valid_vals[is_integer_format].index] = deg + minutes / 60 + seconds / 3600
            
            # 既に度数法のもの
            result.loc[valid_vals[~is_integer_format].index] = valid_vals[~is_integer_format]
            
            return result
        
        df['lat'] = convert_coord_vectorized(df[lat_col])
        df['lon'] = convert_coord_vectorized(df[lon_col])
        
        print(f"   緯度範囲: {df['lat'].min():.4f} - {df['lat'].max():.4f}")
        print(f"   経度範囲: {df['lon'].min():.4f} - {df['lon'].max():.4f}")
        
        return df
    
    def filter_invalid_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        """無効な座標を除去（日本領域外）"""
        print("\n🔍 座標外れ値除去中...")
        
        original_len = len(df)
        
        # 日本の緯度経度範囲
        lat_min, lat_max = 24.0, 46.0
        lon_min, lon_max = 122.0, 146.0
        
        df = df[
            (df['lat'] >= lat_min) & (df['lat'] <= lat_max) &
            (df['lon'] >= lon_min) & (df['lon'] <= lon_max) &
            (df['lat'].notna()) & (df['lon'].notna())
        ].copy()
        
        removed = original_len - len(df)
        print(f"   除去: {removed:,} 行 ({removed/original_len*100:.2f}%)")
        
        return df
    
    def generate_geohash(self, df: pd.DataFrame) -> pd.DataFrame:
        """ジオハッシュ生成"""
        print(f"\n📍 ジオハッシュ生成中 (precision={self.geohash_precision})...")
        
        def get_geohash(row):
            try:
                return geohash.encode(row['lat'], row['lon'], precision=self.geohash_precision)
            except:
                return None
        
        df['geohash'] = df.apply(get_geohash, axis=1)
        
        # 高精度版も生成
        def get_geohash_fine(row):
            try:
                return geohash.encode(row['lat'], row['lon'], precision=7)
            except:
                return None
        
        df['geohash_fine'] = df.apply(get_geohash_fine, axis=1)
        
        n_unique = df['geohash'].nunique()
        print(f"   ユニークジオハッシュ数: {n_unique:,}")
        
        return df
    
    def create_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """日付列の作成"""
        print("\n📅 日付列作成中...")
        
        df['date'] = pd.to_datetime(
            df['year'].astype(str) + '-' + 
            df['month'].astype(str).str.zfill(2) + '-' + 
            df['day'].astype(str).str.zfill(2),
            errors='coerce'
        )
        
        # 無効な日付を除去
        df = df[df['date'].notna()].copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"   日付範囲: {df['date'].min()} - {df['date'].max()}")
        
        return df
    
    def generate_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """時系列特徴量生成"""
        print("\n⏰ 時系列特徴量生成中...")
        
        # hour を sin/cos に変換
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # 曜日を sin/cos に変換
        if '曜日(発生年月日)' in df.columns:
            # 曜日を数値に変換（カテゴリの場合）
            weekday_map = {'日': 0, '月': 1, '火': 2, '水': 3, '木': 4, '金': 5, '土': 6,
                           '日曜': 0, '月曜': 1, '火曜': 2, '水曜': 3, '木曜': 4, '金曜': 5, '土曜': 6,
                           0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
            df['weekday_num'] = df['曜日(発生年月日)'].map(weekday_map).fillna(0).astype(int)
            df['weekday_sin'] = np.sin(2 * np.pi * df['weekday_num'] / 7)
            df['weekday_cos'] = np.cos(2 * np.pi * df['weekday_num'] / 7)
        
        # month を sin/cos に変換
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # 祝日・昼夜のバイナリ特徴量
        if '祝日(発生年月日)' in df.columns:
            df['is_holiday'] = (df['祝日(発生年月日)'] == 1).astype(int)
        
        if '昼夜' in df.columns:
            # 昼夜コードを数値に（1: 昼, 2: 夜 など）
            df['is_night'] = (df['昼夜'].isin([2, 3, 4, 5, 6, 7, 8, 9, 10])).astype(int)
        
        print("   生成完了: hour_sin/cos, weekday_sin/cos, month_sin/cos, is_holiday, is_night")
        
        return df
    
    def generate_spatial_temporal_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """過去ウィンドウでのジオハッシュごとの事故集計（カレンダーベース・高速版）"""
        print("\n📊 空間・時系列集約特徴量生成中 (カレンダーベース)...")
        
        # 日付順にソート
        df = df.sort_values('date').reset_index(drop=True)
        
        # 全期間の日付レンジを生成
        all_dates = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
        all_geohashes = df['geohash'].dropna().unique()
        
        print(f"   日付範囲: {all_dates.min()} - {all_dates.max()} ({len(all_dates)}日間)")
        print(f"   ユニークGeohash: {len(all_geohashes):,}")
        
        # ========================================
        # 全事故のピボットテーブル（行: 日付, 列: Geohash）
        # ========================================
        daily_counts = df.groupby(['date', 'geohash']).size().unstack(fill_value=0)
        
        # 全日付でreindex（事故がない日を0で埋める）
        daily_counts = daily_counts.reindex(all_dates, fill_value=0)
        
        # 各ウィンドウでローリング集計
        for window in self.past_windows:
            print(f"   過去{window}日ウィンドウ (全事故)...")
            
            # shift(1)で未来情報を除外し、rolling sum
            rolled = daily_counts.shift(1).rolling(window=window, min_periods=1).sum()
            
            # 縦持ち（Long形式）に戻す
            rolled_long = rolled.stack().reset_index()
            rolled_long.columns = ['date', 'geohash', f'geohash_accidents_past_{window}d']
            
            # 元データにマージ
            df = df.merge(rolled_long, on=['date', 'geohash'], how='left')
            df[f'geohash_accidents_past_{window}d'] = df[f'geohash_accidents_past_{window}d'].fillna(0)
        
        # ========================================
        # 死亡事故のピボットテーブル
        # ========================================
        fatal_df = df[df[self.target_col] == 1]
        if len(fatal_df) > 0:
            daily_fatal = fatal_df.groupby(['date', 'geohash']).size().unstack(fill_value=0)
            daily_fatal = daily_fatal.reindex(all_dates, fill_value=0)
            
            # 存在しないGeohashを0で埋める
            missing_geohashes = [g for g in all_geohashes if g not in daily_fatal.columns]
            for g in missing_geohashes:
                daily_fatal[g] = 0
            
            for window in self.past_windows:
                print(f"   過去{window}日ウィンドウ (死亡事故)...")
                
                rolled_fatal = daily_fatal.shift(1).rolling(window=window, min_periods=1).sum()
                rolled_fatal_long = rolled_fatal.stack().reset_index()
                rolled_fatal_long.columns = ['date', 'geohash', f'geohash_fatal_past_{window}d']
                
                df = df.merge(rolled_fatal_long, on=['date', 'geohash'], how='left')
                df[f'geohash_fatal_past_{window}d'] = df[f'geohash_fatal_past_{window}d'].fillna(0)
        else:
            for window in self.past_windows:
                df[f'geohash_fatal_past_{window}d'] = 0
        
        print("   集約特徴量生成完了 ✅")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """欠損値処理"""
        print("\n🔧 欠損値処理中...")
        
        for col in df.columns:
            if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                # 数値列: 中央値で補完
                if df[col].isna().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
            elif df[col].dtype == 'object':
                # カテゴリ列: "_missing" で補完
                if df[col].isna().any():
                    df[col] = df[col].fillna("_missing")
        
        print("   欠損値処理完了")
        
        return df
    
    def identify_column_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
        """カラムタイプの識別"""
        
        # 除外する列
        exclude_cols = [
            self.target_col, 'date', 'lat', 'lon', 
            '地点　緯度（北緯）', '地点　経度（東経）',
            'geohash', 'geohash_fine', 'year'
        ]
        
        # 既知のカテゴリ列
        known_categoricals = [
            '都道府県コード', '市区町村コード', 'ゾーン規制', '信号機', '地形',
            '天候', '路面状態', '道路形状', '道路線形', '衝突地点',
            '中央分離帯施設等', '歩車道区分', '昼夜', '曜日(発生年月日)',
            '祝日(発生年月日)', 'area_id', 'road_type', 'terrain_id',
            '一時停止規制　標識（当事者A）', '一時停止規制　標識（当事者B）',
            '一時停止規制　表示（当事者A）', '一時停止規制　表示（当事者B）',
            '当事者種別（当事者A）', '用途別（当事者A）'
        ]
        
        low_cardinality_cats = []
        high_cardinality_cats = []
        numerical_cols = []
        
        for col in df.columns:
            if col in exclude_cols:
                continue
            
            if col in known_categoricals or df[col].dtype == 'object':
                nunique = df[col].nunique()
                if nunique < self.high_cardinality_threshold:
                    low_cardinality_cats.append(col)
                else:
                    high_cardinality_cats.append(col)
            else:
                numerical_cols.append(col)
        
        print(f"\n📋 カラムタイプ:")
        print(f"   数値列: {len(numerical_cols)}")
        print(f"   低カーディナリティ: {len(low_cardinality_cats)}")
        print(f"   高カーディナリティ: {len(high_cardinality_cats)}")
        
        return numerical_cols, low_cardinality_cats, high_cardinality_cats
    
    def split_by_time(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """時間ベースでの分割"""
        print("\n✂️ 時間ベースでデータ分割中...")
        
        train_mask = (df['year'] >= self.train_years[0]) & (df['year'] <= self.train_years[1])
        val_mask = (df['year'] >= self.val_years[0]) & (df['year'] <= self.val_years[1])
        test_mask = (df['year'] >= self.test_years[0]) & (df['year'] <= self.test_years[1])
        
        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()
        
        print(f"   Train: {len(train_df):,} (Fatal: {train_df[self.target_col].sum():,})")
        print(f"   Val:   {len(val_df):,} (Fatal: {val_df[self.target_col].sum():,})")
        print(f"   Test:  {len(test_df):,} (Fatal: {test_df[self.target_col].sum():,})")
        
        return train_df, val_df, test_df
    
    def fit_encoders(self, train_df: pd.DataFrame, numerical_cols: List[str], 
                     low_cardinality_cats: List[str], high_cardinality_cats: List[str]):
        """エンコーダの学習"""
        print("\n🎓 エンコーダ学習中...")
        
        # StandardScaler
        if numerical_cols:
            self.scaler.fit(train_df[numerical_cols])
            print(f"   StandardScaler: {len(numerical_cols)} 列")
        
        # One-Hot Encoder
        if low_cardinality_cats:
            # カテゴリ列を文字列に変換
            train_cats = train_df[low_cardinality_cats].astype(str)
            self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.ohe.fit(train_cats)
            print(f"   OneHotEncoder: {len(low_cardinality_cats)} 列 → {len(self.ohe.get_feature_names_out())} 特徴量")
        
        # Target Encoder (K-Fold方式でリーク防止)
        if high_cardinality_cats:
            self.target_encoders = {}  # 初期化
            n_te_folds = 5
            
            for col in high_cardinality_cats:
                global_mean = train_df[self.target_col].mean()
                
                # 学習データ全体の平均（テスト用）
                category_means = train_df.groupby(col)[self.target_col].mean().to_dict()
                
                # K-Fold用のエンコード値を計算（学習データ内での使用）
                kfold_encoded = pd.Series(index=train_df.index, dtype=float)
                kf = KFold(n_splits=n_te_folds, shuffle=True, random_state=RANDOM_SEED)
                
                for tr_idx, val_idx in kf.split(train_df):
                    tr_data = train_df.iloc[tr_idx]
                    val_data = train_df.iloc[val_idx]
                    
                    # Fold内の平均を計算
                    fold_means = tr_data.groupby(col)[self.target_col].mean()
                    
                    # Val部分にマッピング（未知カテゴリはグローバル平均）
                    kfold_encoded.iloc[val_idx] = val_data[col].map(fold_means).fillna(global_mean)
                
                self.target_encoders[col] = {
                    'global_mean': global_mean,
                    'category_means': category_means,
                    'kfold_encoded': kfold_encoded,  # 学習データ用
                }
            
            print(f"   TargetEncoder (K-Fold): {len(high_cardinality_cats)} 列")
    
    def transform_data(self, df: pd.DataFrame, numerical_cols: List[str],
                       low_cardinality_cats: List[str], high_cardinality_cats: List[str],
                       is_train: bool = False) -> pd.DataFrame:
        """データ変換"""
        
        result_dfs = []
        
        # 基本情報を保持
        meta_cols = ['lat', 'lon', 'geohash', 'geohash_fine', 'date', self.target_col, 'year']
        meta_df = df[[c for c in meta_cols if c in df.columns]].copy()
        result_dfs.append(meta_df)
        
        # 数値列のスケーリング
        if numerical_cols:
            scaled = self.scaler.transform(df[numerical_cols])
            scaled_df = pd.DataFrame(scaled, columns=[f"{c}_scaled" for c in numerical_cols], index=df.index)
            result_dfs.append(scaled_df)
        
        # One-Hot Encoding
        if low_cardinality_cats and self.ohe is not None:
            cats = df[low_cardinality_cats].astype(str)
            ohe_transformed = self.ohe.transform(cats)
            ohe_df = pd.DataFrame(
                ohe_transformed, 
                columns=self.ohe.get_feature_names_out(),
                index=df.index
            )
            result_dfs.append(ohe_df)
        
        # Target Encoding (K-Fold方式: 学習データはKFoldエンコード値、テスト/Valは全体平均)
        if high_cardinality_cats:
            for col in high_cardinality_cats:
                encoder = self.target_encoders.get(col)
                if encoder:
                    if is_train and 'kfold_encoded' in encoder:
                        # 学習データ: K-Foldでエンコードした値を使用（リーク防止）
                        df[f"{col}_te"] = encoder['kfold_encoded'].reindex(df.index).fillna(encoder['global_mean'])
                    else:
                        # テスト/Valデータ: 学習データ全体の平均値を使用
                        df[f"{col}_te"] = df[col].map(encoder['category_means']).fillna(encoder['global_mean'])
                    result_dfs.append(df[[f"{col}_te"]])
        
        # 時系列特徴量を含める
        temporal_cols = [c for c in df.columns if any(x in c for x in 
                        ['_sin', '_cos', 'is_holiday', 'is_night', 'past_'])]
        if temporal_cols:
            result_dfs.append(df[temporal_cols])
        
        result = pd.concat(result_dfs, axis=1)
        
        # 重複列を削除
        result = result.loc[:, ~result.columns.duplicated()]
        
        return result
    
    def save_outputs(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                     test_df: pd.DataFrame, numerical_cols: List[str],
                     low_cardinality_cats: List[str], high_cardinality_cats: List[str]):
        """出力ファイルの保存"""
        print("\n💾 ファイル保存中...")
        
        # Parquet形式で保存
        train_df.to_parquet(self.output_dir / "preprocessed_train.parquet", index=False)
        val_df.to_parquet(self.output_dir / "preprocessed_val.parquet", index=False)
        test_df.to_parquet(self.output_dir / "preprocessed_test.parquet", index=False)
        
        # エンコーダの保存
        joblib.dump(self.scaler, self.output_dir / "scaler.joblib")
        if self.ohe is not None:
            joblib.dump(self.ohe, self.output_dir / "ohe.joblib")
        joblib.dump(self.target_encoders, self.output_dir / "target_encoder.joblib")
        
        # カラム情報の保存
        column_info = {
            'numerical_cols': numerical_cols,
            'low_cardinality_cats': low_cardinality_cats,
            'high_cardinality_cats': high_cardinality_cats,
            'target_col': self.target_col,
        }
        joblib.dump(column_info, self.output_dir / "column_info.joblib")
        
        print(f"   保存先: {self.output_dir}")
        print(f"   - preprocessed_train.parquet ({len(train_df):,} 行)")
        print(f"   - preprocessed_val.parquet ({len(val_df):,} 行)")
        print(f"   - preprocessed_test.parquet ({len(test_df):,} 行)")
        print(f"   - scaler.joblib, ohe.joblib, target_encoder.joblib")
    
    def save_raw_outputs(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                         test_df: pd.DataFrame):
        """GBDT用の生データ（エンコーディング前）を保存"""
        print("\n💾 GBDT用生データ保存中...")
        
        # カテゴリ変数はそのまま、時空間特徴量は付与
        meta_cols = ['lat', 'lon', 'geohash', 'geohash_fine', 'date', self.target_col, 'year']
        
        # 時系列特徴量と空間集約特徴量を含める
        temporal_cols = [c for c in train_df.columns if any(x in c for x in 
                        ['_sin', '_cos', 'is_holiday', 'is_night', 'past_'])]
        
        # 除外する列
        exclude_cols = ['date', '地点　緯度（北緯）', '地点　経度（東経）']
        
        # 保存する列を決定
        keep_cols = [c for c in train_df.columns if c not in exclude_cols]
        
        # Parquet形式で保存
        train_df[keep_cols].to_parquet(self.output_dir / "raw_train.parquet", index=False)
        val_df[keep_cols].to_parquet(self.output_dir / "raw_val.parquet", index=False)
        test_df[keep_cols].to_parquet(self.output_dir / "raw_test.parquet", index=False)
        
        print(f"   保存先: {self.output_dir}")
        print(f"   - raw_train.parquet ({len(train_df):,} 行, {len(keep_cols)} 列)")
        print(f"   - raw_val.parquet ({len(val_df):,} 行)")
        print(f"   - raw_test.parquet ({len(test_df):,} 行)")
    
    def run(self, output_raw: bool = True) -> Dict:
        """前処理パイプラインの実行"""
        start_time = datetime.now()
        
        # 1. データ読み込み
        df = self.load_data()
        
        # 2. 緯度経度変換
        df = self.convert_lat_lon(df)
        
        # 3. 無効座標除去
        df = self.filter_invalid_coords(df)
        
        # 4. ジオハッシュ生成
        df = self.generate_geohash(df)
        
        # 5. 日付列作成
        df = self.create_date_column(df)
        
        # 6. 時系列特徴量
        df = self.generate_temporal_features(df)
        
        # 7. 空間・時系列集約
        df = self.generate_spatial_temporal_aggregates(df)
        
        # 8. 欠損値処理
        df = self.handle_missing_values(df)
        
        # 9. カラムタイプ識別
        numerical_cols, low_cardinality_cats, high_cardinality_cats = self.identify_column_types(df)
        
        # 10. 時間ベース分割
        train_df, val_df, test_df = self.split_by_time(df)
        
        # 11. エンコーダ学習
        self.fit_encoders(train_df, numerical_cols, low_cardinality_cats, high_cardinality_cats)
        
        # 12. データ変換
        train_transformed = self.transform_data(train_df, numerical_cols, low_cardinality_cats, high_cardinality_cats, is_train=True)
        val_transformed = self.transform_data(val_df, numerical_cols, low_cardinality_cats, high_cardinality_cats)
        test_transformed = self.transform_data(test_df, numerical_cols, low_cardinality_cats, high_cardinality_cats)
        
        # 13. 保存
        self.save_outputs(train_transformed, val_transformed, test_transformed,
                         numerical_cols, low_cardinality_cats, high_cardinality_cats)
        
        # GBDT用の生データも保存
        if output_raw:
            self.save_raw_outputs(train_df, val_df, test_df)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        print("\n" + "=" * 70)
        print(f"✅ 前処理完了！ (所要時間: {elapsed:.1f}秒)")
        print("=" * 70)
        
        return {
            'train_size': len(train_transformed),
            'val_size': len(val_transformed),
            'test_size': len(test_transformed),
            'n_features': len(train_transformed.columns),
            'elapsed_seconds': elapsed,
        }


def main():
    parser = argparse.ArgumentParser(description="Spatio-Temporal Preprocessing")
    parser.add_argument('--data-path', type=str, 
                        default="data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv")
    parser.add_argument('--output-dir', type=str, default="data/spatio_temporal")
    parser.add_argument('--train-years', type=str, default="2018,2019")
    parser.add_argument('--val-years', type=str, default="2020,2020")
    parser.add_argument('--test-years', type=str, default="2021,2024")
    parser.add_argument('--geohash-precision', type=int, default=6)
    parser.add_argument('--test', action='store_true', help="テストモード（小規模サブセットで実行）")
    
    args = parser.parse_args()
    
    train_years = tuple(map(int, args.train_years.split(',')))
    val_years = tuple(map(int, args.val_years.split(',')))
    test_years = tuple(map(int, args.test_years.split(',')))
    
    preprocessor = SpatioTemporalPreprocessor(
        data_path=args.data_path,
        output_dir=args.output_dir,
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        geohash_precision=args.geohash_precision,
    )
    
    result = preprocessor.run()
    print(f"\n結果: {result}")


if __name__ == "__main__":
    main()
