"""
相関行列の作成と可視化

このスクリプトは、honhyo_all_shishasuu_binary.csvファイルから相関行列を計算し、
ヒートマップとして可視化します。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 日本語フォントの設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

# ファイルパスの設定
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'honhyo_all_shishasuu_binary.csv')
output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'analysis')

# 出力ディレクトリが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)

print(f"データを読み込んでいます: {data_path}")
# データの読み込み
df = pd.read_csv(data_path)

print(f"データ形状: {df.shape}")
print(f"カラム数: {len(df.columns)}")

# 数値型のカラムのみを選択
numeric_df = df.select_dtypes(include=[np.number])
print(f"数値型カラム数: {len(numeric_df.columns)}")

# 相関行列の計算
print("\n相関行列を計算しています...")
correlation_matrix = numeric_df.corr()

# 相関行列をCSVとして保存
correlation_csv_path = os.path.join(output_dir, 'correlation_matrix.csv')
correlation_matrix.to_csv(correlation_csv_path)
print(f"相関行列を保存しました: {correlation_csv_path}")

# 可視化: フルサイズのヒートマップ
print("\nヒートマップを作成しています...")
fig, ax = plt.subplots(figsize=(20, 18))
sns.heatmap(correlation_matrix, 
            annot=False,  # カラムが多い場合は数値を表示しない
            cmap='coolwarm',
            center=0,
            vmin=-1, 
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8})

plt.title('相関行列ヒートマップ', fontsize=16, pad=20)
plt.tight_layout()

# 画像として保存
heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
print(f"ヒートマップを保存しました: {heatmap_path}")
plt.close()

# 高い相関を持つペアを抽出（絶対値0.7以上）
print("\n高い相関を持つ変数ペアを抽出しています（|r| >= 0.7）...")
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) >= 0.7:
            high_corr_pairs.append({
                '変数1': correlation_matrix.columns[i],
                '変数2': correlation_matrix.columns[j],
                '相関係数': corr_value
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs)
    high_corr_df = high_corr_df.sort_values('相関係数', key=abs, ascending=False)
    
    high_corr_csv_path = os.path.join(output_dir, 'high_correlation_pairs.csv')
    high_corr_df.to_csv(high_corr_csv_path, index=False, encoding='utf-8-sig')
    print(f"高い相関を持つペア（{len(high_corr_pairs)}組）を保存しました: {high_corr_csv_path}")
    
    # 上位10件を表示
    print("\n上位10件の高い相関を持つペア:")
    print(high_corr_df.head(10).to_string(index=False))
else:
    print("相関係数の絶対値が0.7以上のペアは見つかりませんでした。")

# 目的変数との相関（死者数がある場合）
if 'shishasuu' in df.columns or '死者数' in df.columns:
    target_col = 'shishasuu' if 'shishasuu' in df.columns else '死者数'
    print(f"\n目的変数（{target_col}）との相関を確認しています...")
    
    target_corr = correlation_matrix[target_col].sort_values(ascending=False)
    
    # 自己相関以外のトップ20を表示
    print(f"\n{target_col}との相関（上位20件）:")
    print(target_corr.head(21).to_string())  # 自己相関を含めて21件表示
    
    # 目的変数との相関をCSVとして保存
    target_corr_path = os.path.join(output_dir, f'{target_col}_correlation.csv')
    target_corr.to_csv(target_corr_path, header=['相関係数'], encoding='utf-8-sig')
    print(f"\n{target_col}との相関を保存しました: {target_corr_path}")
    
    # 目的変数との相関の可視化（上位30件）
    fig, ax = plt.subplots(figsize=(12, 10))
    top_features = target_corr.head(31).drop(target_col, errors='ignore').head(30)
    
    colors = ['red' if x < 0 else 'blue' for x in top_features.values]
    top_features.plot(kind='barh', ax=ax, color=colors, alpha=0.7)
    
    plt.title(f'{target_col}との相関（上位30変数）', fontsize=14, pad=20)
    plt.xlabel('相関係数', fontsize=12)
    plt.ylabel('変数名', fontsize=12)
    plt.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
    plt.tight_layout()
    
    target_corr_plot_path = os.path.join(output_dir, f'{target_col}_correlation_plot.png')
    plt.savefig(target_corr_plot_path, dpi=150, bbox_inches='tight')
    print(f"{target_col}との相関プロットを保存しました: {target_corr_plot_path}")
    plt.close()

print("\n処理が完了しました！")
print(f"全ての結果は {output_dir} に保存されています。")
