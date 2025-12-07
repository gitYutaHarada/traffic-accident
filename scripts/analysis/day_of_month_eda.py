"""
日別事故傾向の探索的データ分析 (EDA)
- 1〜31日それぞれの死亡事故率を可視化
- 月初・給料日周辺・月末のグループ別比較
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

# 日本語フォント設定
mpl.rcParams['font.family'] = 'MS Gothic'

def main():
    print("=" * 80)
    print("日別事故傾向の探索的データ分析 (EDA)")
    print("=" * 80)
    
    # データ読み込み
    df = pd.read_csv('data/processed/honhyo_model_ready.csv')
    print(f"データ件数: {len(df):,} 件")
    
    # 日カラムの確認
    day_col = '発生日時　　日'
    target_col = '死者数'
    
    if day_col not in df.columns:
        print(f"❌ エラー: '{day_col}' カラムが見つかりません")
        return
    
    # ---------------------------------------------------------
    # Step 1: 日別の死亡事故率を計算
    # ---------------------------------------------------------
    print("\n📊 Step 1: 日別死亡事故率の計算")
    
    # 日ごとの集計
    daily_stats = df.groupby(day_col).agg(
        total_accidents=(target_col, 'count'),
        fatal_accidents=(target_col, 'sum')
    ).reset_index()
    
    daily_stats['fatality_rate'] = daily_stats['fatal_accidents'] / daily_stats['total_accidents'] * 100
    
    print(daily_stats.to_string())
    
    # ---------------------------------------------------------
    # Step 2: 日別死亡率の可視化
    # ---------------------------------------------------------
    print("\n📈 Step 2: グラフ作成")
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    bars = ax.bar(daily_stats[day_col], daily_stats['fatality_rate'], color='steelblue', alpha=0.8)
    
    # 特定日のハイライト
    highlight_days = {
        '月初': [1, 2, 3],
        '給料日': [24, 25, 26],
        '月末': [28, 29, 30, 31]
    }
    colors = {'月初': 'orange', '給料日': 'red', '月末': 'purple'}
    
    for label, days in highlight_days.items():
        for d in days:
            if d <= len(bars):
                bars[d-1].set_color(colors[label])
    
    ax.axhline(y=daily_stats['fatality_rate'].mean(), color='gray', linestyle='--', label=f"平均: {daily_stats['fatality_rate'].mean():.2f}%")
    
    ax.set_xlabel('日 (1〜31日)')
    ax.set_ylabel('死亡事故率 (%)')
    ax.set_title('日別 死亡事故率')
    ax.set_xticks(range(1, 32))
    ax.legend()
    
    # 凡例用のダミー
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='orange', label='月初 (1-3日)'),
        Patch(facecolor='red', label='給料日周辺 (24-26日)'),
        Patch(facecolor='purple', label='月末 (28-31日)'),
        Patch(facecolor='steelblue', label='その他')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('results/visualizations/day_fatality_rate.png', dpi=150)
    print("✓ グラフ保存: results/visualizations/day_fatality_rate.png")
    
    # ---------------------------------------------------------
    # Step 3: グループ別比較
    # ---------------------------------------------------------
    print("\n📊 Step 3: グループ別死亡事故率の比較")
    
    def assign_group(day):
        if day in [1, 2, 3]:
            return '月初'
        elif day in [24, 25, 26]:
            return '給料日周辺'
        elif day in [28, 29, 30, 31]:
            return '月末'
        else:
            return 'その他'
    
    df['day_group'] = df[day_col].apply(assign_group)
    
    group_stats = df.groupby('day_group').agg(
        total_accidents=(target_col, 'count'),
        fatal_accidents=(target_col, 'sum')
    )
    group_stats['fatality_rate'] = group_stats['fatal_accidents'] / group_stats['total_accidents'] * 100
    
    # 順序を指定
    group_order = ['月初', '給料日周辺', '月末', 'その他']
    group_stats = group_stats.reindex(group_order)
    
    print("\n【グループ別統計】")
    print(group_stats.to_string())
    
    # ---------------------------------------------------------
    # Step 4: 統計的検定 (カイ二乗検定)
    # ---------------------------------------------------------
    print("\n📐 Step 4: 統計的検定 (カイ二乗検定)")
    
    # クロス集計表の作成
    contingency_table = pd.crosstab(df['day_group'], df[target_col])
    print("\n【クロス集計表】")
    print(contingency_table)
    
    # カイ二乗検定
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    print(f"\nカイ二乗統計量: {chi2:.4f}")
    print(f"p値: {p_value:.6f}")
    print(f"自由度: {dof}")
    
    if p_value < 0.05:
        print("→ 有意水準5%で、グループ間に統計的に有意な差があります ✓")
    else:
        print("→ 有意水準5%で、グループ間に統計的に有意な差はありません")
    
    # ---------------------------------------------------------
    # 結果サマリー
    # ---------------------------------------------------------
    print("\n" + "=" * 80)
    print("📝 結果サマリー")
    print("=" * 80)
    
    baseline = group_stats.loc['その他', 'fatality_rate']
    for grp in ['月初', '給料日周辺', '月末']:
        rate = group_stats.loc[grp, 'fatality_rate']
        diff = rate - baseline
        print(f"  {grp}: {rate:.3f}% (基準との差: {diff:+.3f}%)")
    
    print(f"  その他 (基準): {baseline:.3f}%")
    print("\n✅ EDA完了")

if __name__ == "__main__":
    main()
