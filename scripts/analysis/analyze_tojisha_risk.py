import pandas as pd
import numpy as np

def main():
    # データの読み込み
    file_path = r'c:\Users\socce\software-lab\traffic-accident\honhyo_all\csv\honhyo_all_shishasuu_binary.csv'
    print("データを読み込んでいます...")
    try:
        df = pd.read_csv(file_path, encoding='cp932')
    except UnicodeDecodeError:
        print("cp932での読み込みに失敗しました。utf-8で試行します。")
        df = pd.read_csv(file_path, encoding='utf-8')
    
    print(f"総データ数: {len(df):,}件")
    print(f"死者数=1の件数: {(df['死者数'] == 1).sum():,}件")
    print()
    
    # 当事者種別(当事者B)の列名を確認
    target_col = '当事者種別（当事者B）'
    
    if target_col not in df.columns:
        print(f"エラー: '{target_col}' 列が見つかりません")
        print("利用可能な列:", [col for col in df.columns if '当事者' in col])
        return
    
    # 当事者種別ごとの死亡率を計算
    print("=== 当事者種別(当事者B)別の死亡事故率 ===\n")
    
    # グループ化して集計
    grouped = df.groupby(target_col).agg({
        '死者数': ['count', 'sum', 'mean']
    }).round(4)
    
    grouped.columns = ['総事故数', '死亡事故数', '死亡率']
    grouped['死亡率(%)'] = (grouped['死亡率'] * 100).round(2)
    
    # 死亡率で降順ソート
    grouped_sorted = grouped.sort_values('死亡率', ascending=False)
    
    # 最低事故数のフィルタ（統計的信頼性のため、100件以上のみ）
    grouped_filtered = grouped_sorted[grouped_sorted['総事故数'] >= 100].copy()
    
    print("【死亡率ランキング Top 20】（総事故数100件以上）\n")
    print(f"{'順位':<4} {'コード':<6} {'総事故数':>12} {'死亡事故数':>12} {'死亡率(%)':>10}")
    print("-" * 55)
    
    for i, (code, row) in enumerate(grouped_filtered.head(20).iterrows(), 1):
        print(f"{i:<4} {int(code):<6} {int(row['総事故数']):>12,} {int(row['死亡事故数']):>12,} {row['死亡率(%)']:>10}")
    
    print("\n\n【全体統計】\n")
    print(grouped_sorted.head(30).to_string())
    
    # CSVに保存
    output_path = r'c:\Users\socce\software-lab\traffic-accident\tojisha_shubetsu_risk_ranking.csv'
    grouped_sorted.to_csv(output_path, encoding='utf-8-sig')
    print(f"\n\n詳細データを保存しました: {output_path}")

if __name__ == "__main__":
    main()


