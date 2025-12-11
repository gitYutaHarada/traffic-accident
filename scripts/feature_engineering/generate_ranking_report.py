"""
交互作用特徴量ランキングレポート生成スクリプト

評価結果からMarkdownレポートと可視化を生成します。

出力内容:
- Top 100のランキング表
- 重要度の棒グラフ
- 交互作用タイプ別の分布
- ヒートマップ（上位特徴量）
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class RankingReportGenerator:
    """ランキングレポートを生成するクラス"""
    
    def __init__(self, ranking_csv_path, output_dir='results/interaction_features'):
        """
        Parameters:
        -----------
        ranking_csv_path : str
            評価結果のCSVパス
        output_dir : str
            出力先ディレクトリ
        """
        self.ranking_csv_path = ranking_csv_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # データ読み込み
        print(f"ランキングデータを読み込み中: {ranking_csv_path}")
        self.results_df = pd.read_csv(ranking_csv_path)
        print(f"データ読み込み完了: {len(self.results_df)} 件")
        
    def generate_bar_chart(self, top_n=20):
        """
        Top Nの重要度棒グラフを生成
        
        Parameters:
        -----------
        top_n : int
            表示する上位N個
            
        Returns:
        --------
        str
            保存したファイルパス
        """
        top_df = self.results_df.head(top_n).copy()
        
        # 図の作成
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 棒グラフ
        bars = ax.barh(
            range(top_n),
            top_df['delta_pr_auc'],
            color=['green' if x > 0 else 'red' for x in top_df['delta_pr_auc']]
        )
        
        # ラベル設定
        ax.set_yticks(range(top_n))
        ax.set_yticklabels([
            f"{row['rank']}. {row['feature1'][:15]} × {row['feature2'][:15]}"
            for _, row in top_df.iterrows()
        ], fontsize=9)
        
        ax.set_xlabel('Delta PR-AUC', fontsize=12)
        ax.set_title(f'Top {top_n} 交互作用特徴量の重要度', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # 値をバーに表示
        for i, (_, row) in enumerate(top_df.iterrows()):
            value = row['delta_pr_auc']
            ax.text(
                value + 0.00001 if value > 0 else value - 0.00001,
                i,
                f'{value:+.6f}',
                va='center',
                ha='left' if value > 0 else 'right',
                fontsize=8
            )
        
        plt.tight_layout()
        
        # 保存
        output_path = self.output_dir / f'top{top_n}_bar_chart.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"棒グラフを保存: {output_path}")
        return str(output_path)
    
    def generate_heatmap(self, top_n=30):
        """
        上位N個の交互作用特徴量のヒートマップを生成
        
        Parameters:
        -----------
        top_n : int
            表示する上位N個
            
        Returns:
        --------
        str
            保存したファイルパス
        """
        top_df = self.results_df.head(top_n).copy()
        
        # 特徴量名を短縮
        top_df['short_name'] = top_df.apply(
            lambda row: f"{row['feature1'][:10]}×{row['feature2'][:10]}", 
            axis=1
        )
        
        # ヒートマップ用のデータ作成
        heatmap_data = top_df[['short_name', 'delta_pr_auc', 'delta_roc_auc', 'delta_f1']].set_index('short_name')
        
        # 図の作成
        fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.3)))
        
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.6f',
            cmap='RdYlGn',
            center=0,
            cbar_kws={'label': 'Delta Score'},
            linewidths=0.5,
            ax=ax
        )
        
        ax.set_title(f'Top {top_n} 交互作用特徴量の評価指標', fontsize=14, fontweight='bold')
        ax.set_xlabel('評価指標', fontsize=12)
        ax.set_ylabel('交互作用特徴量', fontsize=12)
        
        plt.tight_layout()
        
        # 保存
        output_path = self.output_dir / f'top{top_n}_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ヒートマップを保存: {output_path}")
        return str(output_path)
    
    def generate_distribution_plot(self):
        """
        Delta PR-AUCの分布プロットを生成
        
        Returns:
        --------
        str
            保存したファイルパス
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Delta PR-AUCのヒストグラム
        ax1 = axes[0, 0]
        ax1.hist(self.results_df['delta_pr_auc'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='ベースライン')
        ax1.set_xlabel('Delta PR-AUC', fontsize=11)
        ax1.set_ylabel('頻度', fontsize=11)
        ax1.set_title('Delta PR-AUCの分布', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # 2. 交互作用タイプ別の平均Delta PR-AUC
        ax2 = axes[0, 1]
        type_avg = self.results_df.groupby('interaction_type')['delta_pr_auc'].mean().sort_values()
        type_avg.plot(kind='barh', ax=ax2, color='coral')
        ax2.set_xlabel('平均 Delta PR-AUC', fontsize=11)
        ax2.set_ylabel('交互作用タイプ', fontsize=11)
        ax2.set_title('交互作用タイプ別の平均重要度', fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax2.grid(axis='x', alpha=0.3)
        
        # 3. 累積分布
        ax3 = axes[1, 0]
        sorted_delta = np.sort(self.results_df['delta_pr_auc'].values)[::-1]
        cumsum = np.cumsum(sorted_delta)
        ax3.plot(range(len(cumsum)), cumsum, linewidth=2, color='green')
        ax3.set_xlabel('ランク', fontsize=11)
        ax3.set_ylabel('累積 Delta PR-AUC', fontsize=11)
        ax3.set_title('累積重要度', fontsize=12, fontweight='bold')
        ax3.grid(alpha=0.3)
        
        # 4. PR-AUC vs Delta PR-AUC
        ax4 = axes[1, 1]
        scatter = ax4.scatter(
            self.results_df['pr_auc'],
            self.results_df['delta_pr_auc'],
            c=self.results_df['delta_pr_auc'],
            cmap='RdYlGn',
            alpha=0.6,
            s=30
        )
        ax4.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax4.set_xlabel('PR-AUC（交互作用特徴量追加後）', fontsize=11)
        ax4.set_ylabel('Delta PR-AUC', fontsize=11)
        ax4.set_title('PR-AUC vs Delta PR-AUC', fontsize=12, fontweight='bold')
        ax4.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax4, label='Delta PR-AUC')
        
        plt.tight_layout()
        
        # 保存
        output_path = self.output_dir / 'distribution_plots.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"分布プロットを保存: {output_path}")
        return str(output_path)
    
    def generate_markdown_report(self, top_n=100):
        """
        Markdownレポートを生成
        
        Parameters:
        -----------
        top_n : int
            レポートに含める上位N個
            
        Returns:
        --------
        str
            保存したファイルパス
        """
        top_df = self.results_df.head(top_n)
        
        # 統計情報
        total_count = len(self.results_df)
        positive_count = (self.results_df['delta_pr_auc'] > 0).sum()
        negative_count = (self.results_df['delta_pr_auc'] < 0).sum()
        zero_count = (self.results_df['delta_pr_auc'] == 0).sum()
        
        best_feature = self.results_df.iloc[0]
        worst_feature = self.results_df.iloc[-1]
        
        # Markdown生成
        timestamp = datetime.now().strftime('%Y年%m月%d日 %H:%M')
        
        md_content = f"""# 交互作用特徴量 重要度ランキングレポート

**作成日時**: {timestamp}  
**評価対象**: {total_count} 個の交互作用特徴量  
**評価手法**: LightGBM（5-fold Stratified CV、PR-AUC最大化）

---

## エグゼクティブサマリー

### 評価結果サマリー

| 項目 | 値 |
|------|-----|
| **総評価数** | {total_count} 個 |
| **PR-AUC向上** | {positive_count} 個 ({positive_count/total_count*100:.1f}%) |
| **PR-AUC低下** | {negative_count} 個 ({negative_count/total_count*100:.1f}%) |
| **変化なし** | {zero_count} 個 ({zero_count/total_count*100:.1f}%) |

### 最良の交互作用特徴量

**{best_feature['feature_name']}**

- **組み合わせ**: `{best_feature['feature1']}` × `{best_feature['feature2']}`
- **タイプ**: {best_feature['interaction_type']}
- **Delta PR-AUC**: {best_feature['delta_pr_auc']:+.6f} ({best_feature['delta_pr_auc']*100:+.2f}%)
- **PR-AUC**: {best_feature['pr_auc']:.6f}
- **ROC-AUC**: {best_feature['roc_auc']:.6f}
- **F1 Score**: {best_feature['f1']:.6f}

---

## Top {min(top_n, len(top_df))} 交互作用特徴量ランキング

| ランク | 特徴量名 | 組み合わせ | タイプ | Delta PR-AUC | PR-AUC | Delta ROC-AUC | Delta F1 |
|--------|---------|-----------|--------|--------------|--------|---------------|----------|
"""
        
        for _, row in top_df.iterrows():
            md_content += f"| {row['rank']} | `{row['feature_name']}` | `{row['feature1']}` × `{row['feature2']}` | {row['interaction_type']} | {row['delta_pr_auc']:+.6f} | {row['pr_auc']:.6f} | {row['delta_roc_auc']:+.6f} | {row['delta_f1']:+.6f} |\n"
        
        md_content += f"""

---

## 詳細分析

### 1. 交互作用タイプ別の統計

"""
        
        # タイプ別の統計
        type_stats = self.results_df.groupby('interaction_type').agg({
            'delta_pr_auc': ['count', 'mean', 'std', 'min', 'max']
        }).round(6)
        
        md_content += "| タイプ | 個数 | 平均 Delta PR-AUC | 標準偏差 | 最小値 | 最大値 |\n"
        md_content += "|--------|------|------------------|----------|--------|--------|\n"
        
        for interaction_type in type_stats.index:
            stats = type_stats.loc[interaction_type, 'delta_pr_auc']
            md_content += f"| {interaction_type} | {int(stats['count'])} | {stats['mean']:+.6f} | {stats['std']:.6f} | {stats['min']:+.6f} | {stats['max']:+.6f} |\n"
        
        md_content += f"""

### 2. Top 10 の詳細

"""
        
        for idx, row in self.results_df.head(10).iterrows():
            md_content += f"""
#### {row['rank']}位: `{row['feature_name']}`

- **組み合わせ**: `{row['feature1']}` × `{row['feature2']}`
- **タイプ**: {row['interaction_type']}
- **Delta PR-AUC**: **{row['delta_pr_auc']:+.6f}** ({row['delta_pr_auc']*100:+.2f}%)
- **評価指標**:
  - PR-AUC: {row['pr_auc']:.6f}
  - ROC-AUC: {row['roc_auc']:.6f} (Δ{row['delta_roc_auc']:+.6f})
  - F1 Score: {row['f1']:.6f} (Δ{row['delta_f1']:+.6f})
  - Accuracy: {row['accuracy']:.6f}
  - Precision: {row['precision']:.6f}
  - Recall: {row['recall']:.6f}
- **データ品質**:
  - ユニーク数: {int(row['n_unique'])}
  - 欠損率: {row['missing_rate']*100:.2f}%

---
"""
        
        md_content += """

## 推奨事項

### モデル改善のための次のステップ

1. **Top 10 の特徴量を追加**
   - 最も効果的な交互作用特徴量をモデルに追加
   - 再訓練してPR-AUCの向上を確認

2. **複数の特徴量を組み合わせて評価**
   - Top 5を同時に追加してモデルを訓練
   - 相乗効果があるか検証

3. **特徴量選択の最適化**
   - 後方除去法や前方選択法で最適な組み合わせを探索

4. **ドメイン知識の活用**
   - 上位の交互作用が意味的に妥当か確認
   - 直感に反する組み合わせがある場合は注意深く検証

---

**レポート作成日**: {timestamp}  
**作成者**: Antigravity AI Agent
"""
        
        # 保存
        output_path = self.output_dir / 'interaction_features_ranking_report.md'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        print(f"Markdownレポートを保存: {output_path}")
        return str(output_path)
    
    def generate_all_reports(self):
        """すべてのレポートと可視化を生成"""
        print("\n" + "="*60)
        print("レポート生成開始")
        print("="*60)
        
        # 各種プロット生成
        bar_chart = self.generate_bar_chart(top_n=20)
        heatmap = self.generate_heatmap(top_n=30)
        distribution = self.generate_distribution_plot()
        
        # Markdownレポート生成
        markdown_report = self.generate_markdown_report(top_n=100)
        
        print("\n" + "="*60)
        print("✅ すべてのレポート生成が完了しました！")
        print("="*60)
        print(f"出力先ディレクトリ: {self.output_dir}")
        print(f"\n生成ファイル:")
        print(f"  - {bar_chart}")
        print(f"  - {heatmap}")
        print(f"  - {distribution}")
        print(f"  - {markdown_report}")
        
        return {
            'bar_chart': bar_chart,
            'heatmap': heatmap,
            'distribution': distribution,
            'markdown_report': markdown_report
        }


def main():
    """メイン処理"""
    # 設定（実行時に最新のCSVパスに変更してください）
    RANKING_CSV = 'results/interaction_features/interaction_features_ranking_full_20251211_140000.csv'
    OUTPUT_DIR = 'results/interaction_features'
    
    print("="*60)
    print("交互作用特徴量ランキングレポート生成")
    print("="*60)
    print(f"入力CSV: {RANKING_CSV}")
    print(f"出力先: {OUTPUT_DIR}")
    print("="*60)
    
    # レポート生成器の初期化
    generator = RankingReportGenerator(
        ranking_csv_path=RANKING_CSV,
        output_dir=OUTPUT_DIR
    )
    
    # すべてのレポートを生成
    reports = generator.generate_all_reports()
    
    print("\n次のステップ: レポートを確認して、Top 10の特徴量をモデルに追加してください")


if __name__ == '__main__':
    main()
