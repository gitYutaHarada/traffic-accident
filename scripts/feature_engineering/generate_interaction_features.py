"""
交互作用特徴量の生成スクリプト

すべての2つの特徴量の組み合わせで交互作用特徴量を生成します。
- カテゴリ型 × カテゴリ型: 文字列結合でエンコード
- 数値型 × 数値型: 乗算
- 数値型 × カテゴリ型: カテゴリごとにグループ化した数値

出力:
- 各組み合わせの交互作用特徴量を個別に保存
- メタデータ（特徴量名、タイプ、組み合わせ）をCSVで保存
"""

import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path
from datetime import datetime
import pickle
from tqdm import tqdm


class InteractionFeatureGenerator:
    """交互作用特徴量を生成するクラス"""
    
    def __init__(self, data_path, target_column='死者数', output_dir='data/interaction_features'):
        """
        Parameters:
        -----------
        data_path : str
            元データのパス
        target_column : str
            目的変数のカラム名
        output_dir : str
            交互作用特徴量の保存先ディレクトリ
        """
        self.data_path = data_path
        self.target_column = target_column
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # データ読み込み
        print(f"データを読み込み中: {data_path}")
        self.df = pd.read_csv(data_path)
        print(f"データ形状: {self.df.shape}")
        
        # 目的変数を分離
        if target_column in self.df.columns:
            self.y = self.df[target_column]
            self.feature_columns = [col for col in self.df.columns if col != target_column]
        else:
            raise ValueError(f"目的変数 '{target_column}' が見つかりません")
        
        self.X = self.df[self.feature_columns].copy()
        
        # 特徴量タイプの分類
        self._classify_feature_types()
        
    def _classify_feature_types(self):
        """特徴量を数値型とカテゴリ型に分類"""
        self.numeric_features = []
        self.categorical_features = []
        
        for col in self.feature_columns:
            # 日時型は除外
            if self.X[col].dtype == 'object' and col != '発生日時':
                # 文字列型はスキップ（カテゴリ化が必要）
                continue
            elif col == '発生日時':
                # 日時型は別途処理が必要なのでスキップ
                continue
            elif self.X[col].dtype in ['int64', 'float64']:
                # ユニーク数が少ない場合はカテゴリ型として扱う
                n_unique = self.X[col].nunique()
                if n_unique <= 50:  # 閾値: 50種類以下はカテゴリ
                    self.categorical_features.append(col)
                else:
                    self.numeric_features.append(col)
        
        print(f"\n数値型特徴量: {len(self.numeric_features)}個")
        print(f"カテゴリ型特徴量: {len(self.categorical_features)}個")
        
    def generate_categorical_x_categorical(self, feat1, feat2):
        """
        カテゴリ型 × カテゴリ型の交互作用特徴量
        
        Parameters:
        -----------
        feat1, feat2 : str
            カテゴリ型特徴量名
            
        Returns:
        --------
        pd.Series
            交互作用特徴量
        """
        # 文字列結合でエンコード
        interaction = (
            self.X[feat1].astype(str) + "_" + self.X[feat2].astype(str)
        )
        
        # カテゴリコードに変換（数値化）
        interaction_encoded = pd.Categorical(interaction).codes
        
        return pd.Series(interaction_encoded, name=f"{feat1}_x_{feat2}")
    
    def generate_numeric_x_numeric(self, feat1, feat2):
        """
        数値型 × 数値型の交互作用特徴量
        
        Parameters:
        -----------
        feat1, feat2 : str
            数値型特徴量名
            
        Returns:
        --------
        pd.Series
            交互作用特徴量（乗算）
        """
        interaction = self.X[feat1] * self.X[feat2]
        return pd.Series(interaction, name=f"{feat1}_x_{feat2}")
    
    def generate_numeric_x_categorical(self, numeric_feat, cat_feat):
        """
        数値型 × カテゴリ型の交互作用特徴量
        
        Parameters:
        -----------
        numeric_feat : str
            数値型特徴量名
        cat_feat : str
            カテゴリ型特徴量名
            
        Returns:
        --------
        pd.Series
            交互作用特徴量（カテゴリごとの数値変換）
        """
        # カテゴリごとに数値特徴量の平均で置換
        cat_mean = self.X.groupby(cat_feat)[numeric_feat].transform('mean')
        
        # 元の数値との乗算
        interaction = self.X[numeric_feat] * cat_mean
        
        return pd.Series(interaction, name=f"{numeric_feat}_x_{cat_feat}")
    
    def generate_all_interactions(self):
        """
        すべての2つの組み合わせで交互作用特徴量を生成
        
        Returns:
        --------
        pd.DataFrame
            メタデータ（特徴量名、タイプ、組み合わせ情報）
        """
        metadata = []
        all_features = self.categorical_features + self.numeric_features
        total_combinations = len(list(combinations(all_features, 2)))
        
        print(f"\n合計 {total_combinations} 通りの組み合わせを生成します...")
        
        # プログレスバー付きで生成
        pbar = tqdm(total=total_combinations, desc="交互作用特徴量生成")
        
        for feat1, feat2 in combinations(all_features, 2):
            # タイプ判定
            feat1_type = 'categorical' if feat1 in self.categorical_features else 'numeric'
            feat2_type = 'categorical' if feat2 in self.categorical_features else 'numeric'
            
            # 組み合わせパターンに応じて生成
            if feat1_type == 'categorical' and feat2_type == 'categorical':
                interaction_feature = self.generate_categorical_x_categorical(feat1, feat2)
                interaction_type = 'cat_x_cat'
            elif feat1_type == 'numeric' and feat2_type == 'numeric':
                interaction_feature = self.generate_numeric_x_numeric(feat1, feat2)
                interaction_type = 'num_x_num'
            else:
                # 数値型を先に持ってくる
                if feat1_type == 'numeric':
                    interaction_feature = self.generate_numeric_x_categorical(feat1, feat2)
                else:
                    interaction_feature = self.generate_numeric_x_categorical(feat2, feat1)
                interaction_type = 'num_x_cat'
            
            # 特徴量を保存
            feature_name = interaction_feature.name
            output_path = self.output_dir / f"{feature_name}.pkl"
            
            with open(output_path, 'wb') as f:
                pickle.dump(interaction_feature.values, f)
            
            # メタデータを記録
            metadata.append({
                'feature_name': feature_name,
                'feature1': feat1,
                'feature2': feat2,
                'interaction_type': interaction_type,
                'n_unique': interaction_feature.nunique(),
                'missing_rate': interaction_feature.isna().mean(),
                'file_path': str(output_path)
            })
            
            pbar.update(1)
        
        pbar.close()
        
        # メタデータをDataFrameに変換
        metadata_df = pd.DataFrame(metadata)
        
        # メタデータを保存
        metadata_path = self.output_dir / 'interaction_features_metadata.csv'
        metadata_df.to_csv(metadata_path, index=False, encoding='utf-8-sig')
        print(f"\nメタデータを保存: {metadata_path}")
        
        # サマリー表示
        print("\n" + "="*60)
        print("生成完了サマリー")
        print("="*60)
        print(f"総組み合わせ数: {len(metadata_df)}")
        print(f"\n交互作用タイプ別:")
        print(metadata_df['interaction_type'].value_counts())
        print(f"\n平均欠損率: {metadata_df['missing_rate'].mean():.4f}")
        print(f"保存先ディレクトリ: {self.output_dir}")
        
        return metadata_df


def main():
    """メイン処理"""
    # 設定
    DATA_PATH = 'data/processed/honhyo_clean_predictable_only.csv'
    TARGET_COLUMN = '死者数'
    OUTPUT_DIR = 'data/interaction_features'
    
    # タイムスタンプ付きの出力ディレクトリ
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"{OUTPUT_DIR}_{timestamp}"
    
    print("="*60)
    print("交互作用特徴量生成スクリプト")
    print("="*60)
    print(f"データパス: {DATA_PATH}")
    print(f"出力先: {output_dir}")
    print("="*60)
    
    # 生成器の初期化
    generator = InteractionFeatureGenerator(
        data_path=DATA_PATH,
        target_column=TARGET_COLUMN,
        output_dir=output_dir
    )
    
    # すべての交互作用特徴量を生成
    metadata_df = generator.generate_all_interactions()
    
    print("\n✅ すべての処理が完了しました！")
    print(f"次のステップ: evaluate_interaction_importance.py を実行してください")


if __name__ == '__main__':
    main()
