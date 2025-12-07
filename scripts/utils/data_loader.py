import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder

class DataLoader:
    def __init__(self, raw_data_path, cache_dir='data/processed'):
        self.raw_data_path = raw_data_path
        self.cache_dir = cache_dir
        self.cache_path = os.path.join(cache_dir, 'processed_data.pkl')
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def load_data(self, force_reload=False):
        """
        データを読み込み、前処理を行って返す。
        キャッシュが存在する場合はキャッシュから読み込む。
        """
        if not force_reload and os.path.exists(self.cache_path):
            print(f"📦 キャッシュからデータを読み込み中: {self.cache_path}")
            try:
                data = joblib.load(self.cache_path)
                print("✓ 読み込み完了")
                return data['X'], data['y']
            except Exception as e:
                print(f"⚠️ キャッシュの読み込みに失敗しました: {e}")
                print("🔄 生データから再構築します...")

        print(f"📂 生データを読み込み中: {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)
        
        # 前処理
        X, y = self._preprocess(df)
        
        # キャッシュ保存
        print(f"💾 データをキャッシュに保存中: {self.cache_path}")
        joblib.dump({'X': X, 'y': y}, self.cache_path)
        
        return X, y

    def _preprocess(self, df):
        print("🔧 データ前処理中...")
        
        target_col = '死者数'
        
        # 除外する列
        drop_cols = [
            '資料区分', '本票番号',
            '人身損傷程度（当事者A）', '人身損傷程度（当事者B）',
            '車両の損壊程度（当事者A）', '車両の損壊程度（当事者B）',
            '負傷者数',
            '車両の衝突部位（当事者A）', '車両の衝突部位（当事者B）',
            'エアバッグの装備（当事者A）', 'エアバッグの装備（当事者B）',
            'サイドエアバッグの装備（当事者A）', 'サイドエアバッグの装備（当事者B）',
            '事故内容'  # データリーク原因
        ]
        
        df_clean = df.drop(columns=drop_cols, errors='ignore')
        
        X = df_clean.drop(columns=[target_col])
        y = df_clean[target_col]
        
        # 欠損値処理
        num_cols = X.select_dtypes(include=[np.number]).columns
        X[num_cols] = X[num_cols].fillna(X[num_cols].median())
        
        cat_cols = X.select_dtypes(include=['object']).columns
        for col in cat_cols:
            X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
        
        # エンコーディング
        le = LabelEncoder()
        for col in cat_cols:
            X[col] = le.fit_transform(X[col].astype(str))
            
        print(f"✓ 前処理完了 - 特徴量数: {X.shape[1]}")
        return X, y
