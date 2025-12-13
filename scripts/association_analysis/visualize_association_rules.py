"""
アソシエーション分析結果可視化スクリプト

抽出されたアソシエーションルールを可視化します。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo']
plt.rcParams['axes.unicode_minus'] = False

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def load_rules(file_path: str) -> pd.DataFrame:
    """ルールデータを読み込む"""
    print(f"ルールデータを読み込み中: {file_path}")
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print(f"読み込み完了: {len(df)} ルール")
    return df


def plot_support_confidence_scatter(rules: pd.DataFrame, output_path: Path, title: str = "Support-Confidence散布図"):
    """Support-Confidence散布図を作成"""
    print(f"\n{title}を作成中...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Lift値で色分け
    scatter = ax.scatter(rules['Support'], rules['Confidence'], 
                        c=rules['Lift'], cmap='YlOrRd', 
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Support (支持度)', fontsize=12)
    ax.set_ylabel('Confidence (確信度)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # カラーバー
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Lift (リフト値)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"保存完了: {output_path}")


def plot_lift_ranking(rules: pd.DataFrame, output_path: Path, top_n: int = 20, 
                     title: str = "Lift値ランキング Top 20"):
    """Lift値ランキングを作成"""
    print(f"\n{title}を作成中...")
    
    # 上位N件を取得
    top_rules = rules.head(top_n).copy()
    
    # ルール名を作成(条件 → 結果)
    top_rules['rule_name'] = top_rules.apply(
        lambda x: f"{x['条件'][:30]}... → {x['結果'][:20]}...", axis=1
    )
    
    # プロット
    fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.4)))
    
    bars = ax.barh(range(len(top_rules)), top_rules['Lift'], 
                   color=plt.cm.YlOrRd(top_rules['Lift'] / top_rules['Lift'].max()))
    
    ax.set_yticks(range(len(top_rules)))
    ax.set_yticklabels(top_rules['rule_name'], fontsize=9)
    ax.set_xlabel('Lift (リフト値)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # 値をバーに表示
    for i, (bar, lift) in enumerate(zip(bars, top_rules['Lift'])):
        ax.text(lift, i, f' {lift:.2f}', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"保存完了: {output_path}")


def plot_metrics_comparison(rules: pd.DataFrame, output_path: Path, top_n: int = 15,
                           title: str = "上位ルールの評価指標比較"):
    """複数の評価指標を比較"""
    print(f"\n{title}を作成中...")
    
    # 上位N件を取得
    top_rules = rules.head(top_n).copy()
    
    # ルール名を作成
    top_rules['rule_name'] = top_rules.apply(
        lambda x: f"R{len(top_rules) - top_rules.index.get_loc(x.name)}", axis=1
    )
    
    # プロット
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, top_n * 0.3)))
    
    metrics = ['Support', 'Confidence', 'Lift']
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    for ax, metric, color in zip(axes, metrics, colors):
        ax.barh(range(len(top_rules)), top_rules[metric], color=color, alpha=0.7)
        ax.set_yticks(range(len(top_rules)))
        ax.set_yticklabels(top_rules['rule_name'], fontsize=9)
        ax.set_xlabel(metric, fontsize=11)
        ax.set_title(f'{metric}値', fontsize=12, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        # 値を表示
        for i, val in enumerate(top_rules[metric]):
            ax.text(val, i, f' {val:.3f}', va='center', fontsize=8)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"保存完了: {output_path}")


def plot_condition_frequency(rules: pd.DataFrame, output_path: Path, top_n: int = 15,
                            title: str = "頻出する条件要素 Top 15"):
    """ルールの条件部分で頻出する要素を可視化"""
    print(f"\n{title}を作成中...")
    
    # 条件を分解して頻度をカウント
    condition_counts = {}
    for conditions in rules['条件']:
        for condition in conditions.split(', '):
            condition_counts[condition] = condition_counts.get(condition, 0) + 1
    
    # DataFrameに変換してソート
    freq_df = pd.DataFrame(list(condition_counts.items()), columns=['条件', '出現回数'])
    freq_df = freq_df.sort_values('出現回数', ascending=False).head(top_n)
    
    # プロット
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.4)))
    
    bars = ax.barh(range(len(freq_df)), freq_df['出現回数'], 
                   color=plt.cm.Blues(freq_df['出現回数'] / freq_df['出現回数'].max()))
    
    ax.set_yticks(range(len(freq_df)))
    ax.set_yticklabels(freq_df['条件'], fontsize=10)
    ax.set_xlabel('出現回数', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    
    # 値をバーに表示
    for i, (bar, count) in enumerate(zip(bars, freq_df['出現回数'])):
        ax.text(count, i, f' {count}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"保存完了: {output_path}")


def create_summary_table(rules: pd.DataFrame, output_path: Path, top_n: int = 10):
    """サマリーテーブルを作成"""
    print(f"\nサマリーテーブルを作成中...")
    
    # 上位N件
    top_rules = rules.head(top_n)[['条件', '結果', 'Support', 'Confidence', 'Lift']].copy()
    
    # 数値を丸める
    top_rules['Support'] = top_rules['Support'].round(4)
    top_rules['Confidence'] = top_rules['Confidence'].round(4)
    top_rules['Lift'] = top_rules['Lift'].round(4)
    
    # 保存
    top_rules.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"保存完了: {output_path}")
    
    # コンソールに表示
    print(f"\n=== 上位{top_n}ルール ===")
    print(top_rules.to_string(index=False))


def visualize_fatal_accident_rules(output_dir: Path):
    """死亡事故関連ルールを可視化"""
    print("\n" + "=" * 100)
    print("死亡事故関連ルールの可視化")
    print("=" * 100)
    
    # ルール読み込み
    rules_path = output_dir / "fatal_accident_rules.csv"
    if not rules_path.exists():
        print(f"ファイルが見つかりません: {rules_path}")
        return
    
    rules = load_rules(rules_path)
    
    if len(rules) == 0:
        print("ルールが見つかりませんでした。")
        return
    
    # 可視化
    plot_support_confidence_scatter(
        rules, 
        output_dir / "fatal_scatter_plot.png",
        title="死亡事故ルール: Support-Confidence散布図"
    )
    
    plot_lift_ranking(
        rules,
        output_dir / "fatal_lift_ranking.png",
        top_n=min(20, len(rules)),
        title="死亡事故ルール: Lift値ランキング Top 20"
    )
    
    plot_metrics_comparison(
        rules,
        output_dir / "fatal_metrics_comparison.png",
        top_n=min(15, len(rules)),
        title="死亡事故ルール: 評価指標比較"
    )
    
    plot_condition_frequency(
        rules,
        output_dir / "fatal_condition_frequency.png",
        top_n=15,
        title="死亡事故ルール: 頻出する条件要素"
    )
    
    create_summary_table(
        rules,
        output_dir / "fatal_top_rules_summary.csv",
        top_n=min(20, len(rules))
    )


def visualize_fatal_only_rules(output_dir: Path):
    """死亡事故のみのルールを可視化"""
    print("\n" + "=" * 100)
    print("死亡事故のみのルールの可視化")
    print("=" * 100)
    
    # ルール読み込み
    rules_path = output_dir / "fatal_only_rules.csv"
    if not rules_path.exists():
        print(f"ファイルが見つかりません: {rules_path}")
        return
    
    rules = load_rules(rules_path)
    
    if len(rules) == 0:
        print("ルールが見つかりませんでした。")
        return
    
    # 可視化
    plot_support_confidence_scatter(
        rules,
        output_dir / "fatal_only_scatter_plot.png",
        title="死亡事故のみ: Support-Confidence散布図"
    )
    
    plot_lift_ranking(
        rules,
        output_dir / "fatal_only_lift_ranking.png",
        top_n=min(20, len(rules)),
        title="死亡事故のみ: Lift値ランキング Top 20"
    )
    
    plot_condition_frequency(
        rules,
        output_dir / "fatal_only_condition_frequency.png",
        top_n=15,
        title="死亡事故のみ: 頻出する条件要素"
    )
    
    create_summary_table(
        rules,
        output_dir / "fatal_only_top_rules_summary.csv",
        top_n=min(20, len(rules))
    )


def main():
    """メイン処理"""
    print("=" * 100)
    print("アソシエーション分析結果の可視化")
    print("=" * 100)
    
    # 出力ディレクトリ
    output_dir = project_root / "results" / "association_analysis"
    
    # 1. 死亡事故関連ルールの可視化
    visualize_fatal_accident_rules(output_dir)
    
    # 2. 死亡事故のみのルールの可視化
    visualize_fatal_only_rules(output_dir)
    
    print("\n" + "=" * 100)
    print("可視化完了!")
    print("=" * 100)
    print(f"\n結果は以下に保存されました: {output_dir}")


if __name__ == "__main__":
    main()
