import pandas as pd
import numpy as np
from sklearn.utils import resample
import os

def main():
    """
    クラス不均衡を解消するためのアップサンプリングスクリプト
    少数派クラス（死亡事故）を多数派クラスと同数になるまで複製する
    """
    
    # データファイルのパス
    input_file = 'data/raw/honhyo_all_shishasuu_binary.csv'
    output_file = 'data/processed/honhyo_all_upsampled.csv'
    
    print("=" * 70)
    print("クラス不均衡対処: アップサンプリング")
    print("=" * 70)
    
    # データの読み込み
    print(f"\n📂 データ読み込み中: {input_file}")
    try:
        df = pd.read_csv(input_file)
        print(f"✓ データ読み込み完了: {len(df):,} 件")
    except FileNotFoundError:
        print(f"❌ エラー: ファイルが見つかりません - {input_file}")
        return
    except Exception as e:
        print(f"❌ エラー: {e}")
        return
    
    # 死者数の列を確認
    if '死者数' not in df.columns:
        print("❌ エラー: '死者数' 列が見つかりません")
        return
    
    # クラス分布の確認（元データ）
    print("\n" + "=" * 70)
    print("【アップサンプリング前】クラス分布")
    print("=" * 70)
    
    class_counts = df['死者数'].value_counts().sort_index()
    total = len(df)
    
    print(f"\n総件数: {total:,} 件")
    print("\nクラス別件数:")
    for class_val, count in class_counts.items():
        percentage = (count / total) * 100
        print(f"  死者数={class_val}: {count:,} 件 ({percentage:.2f}%)")
    
    # クラスごとにデータを分離
    df_majority = df[df['死者数'] == 0]  # 非死亡事故（多数派）
    df_minority = df[df['死者数'] == 1]  # 死亡事故（少数派）
    
    majority_count = len(df_majority)
    minority_count = len(df_minority)
    
    print(f"\n多数派クラス（死者数=0）: {majority_count:,} 件")
    print(f"少数派クラス（死者数=1）: {minority_count:,} 件")
    print(f"クラス比: 1:{majority_count/minority_count:.1f}")
    
    # アップサンプリング実行
    print("\n" + "=" * 70)
    print("🔄 アップサンプリング実行中...")
    print("=" * 70)
    
    # 少数派クラスをアップサンプリング（多数派と同数まで）
    df_minority_upsampled = resample(
        df_minority,
        replace=True,              # 復元抽出を許可（同じサンプルを複数回選択可能）
        n_samples=majority_count,  # 多数派と同じ件数まで増やす
        random_state=42            # 再現性のため乱数シードを固定
    )
    
    print(f"✓ 少数派クラスを {minority_count:,} 件 → {len(df_minority_upsampled):,} 件に増加")
    
    # アップサンプリング後のデータを結合
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    
    # データをシャッフル（順序をランダムに）
    df_upsampled = df_upsampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"✓ データを結合・シャッフル完了")
    
    # アップサンプリング後のクラス分布を確認
    print("\n" + "=" * 70)
    print("【アップサンプリング後】クラス分布")
    print("=" * 70)
    
    class_counts_after = df_upsampled['死者数'].value_counts().sort_index()
    total_after = len(df_upsampled)
    
    print(f"\n総件数: {total_after:,} 件 (元データの {total_after/total:.2f}倍)")
    print("\nクラス別件数:")
    for class_val, count in class_counts_after.items():
        percentage = (count / total_after) * 100
        print(f"  死者数={class_val}: {count:,} 件 ({percentage:.2f}%)")
    
    print(f"\n✓ クラス比: 1:1 (完全にバランス調整)")
    
    # データの保存
    print("\n" + "=" * 70)
    print(f"💾 データ保存中: {output_file}")
    print("=" * 70)
    
    # 出力ディレクトリが存在しない場合は作成
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ ディレクトリ作成: {output_dir}")
    
    try:
        df_upsampled.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✓ データ保存完了: {output_file}")
        print(f"  ファイルサイズ: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"❌ 保存エラー: {e}")
        return
    
    # サマリー
    print("\n" + "=" * 70)
    print("📊 サマリー")
    print("=" * 70)
    print(f"・元データ件数: {total:,} 件")
    print(f"・新データ件数: {total_after:,} 件")
    print(f"・増加件数: {total_after - total:,} 件")
    print(f"・少数派クラスの増加: {len(df_minority_upsampled) - minority_count:,} 件")
    print(f"・クラスバランス: 50% : 50%")
    print("=" * 70)
    print("✅ アップサンプリング処理が正常に完了しました")
    print("=" * 70)

if __name__ == "__main__":
    main()
