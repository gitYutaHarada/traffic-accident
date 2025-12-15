"""
アソシエーション分析実行スクリプト

Aprioriアルゴリズムを用いて、交通事故データから
死亡事故に繋がる要因のアソシエーションルールを抽出します。
"""

import pandas as pd
import numpy as np
from pathlib import Path
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import sys
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))


def load_transactions(file_path: str) -> list:
    """トランザクションデータを読み込む"""
    print(f"トランザクションデータを読み込み中: {file_path}")
    
    transactions = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            items = line.strip().split(',')
            if items:
                transactions.append(items)
    
    print(f"読み込み完了: {len(transactions):,} トランザクション")
    return transactions


def create_one_hot_dataframe(transactions: list, sample_size: int = None) -> pd.DataFrame:
    """トランザクションをOne-Hot形式のDataFrameに変換"""
    print("\nOne-Hot形式に変換中...")
    
    # サンプリング
    if sample_size and sample_size < len(transactions):
        print(f"サンプリング: {len(transactions):,} → {sample_size:,} トランザクション")
        import random
        random.seed(42)
        transactions = random.sample(transactions, sample_size)
    
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    
    print(f"変換完了: {len(df)} 行 × {len(df.columns)} 列")
    print(f"ユニークアイテム数: {len(df.columns)}")
    
    return df


def run_fpgrowth(df: pd.DataFrame, min_support: float = 0.01) -> pd.DataFrame:
    """FP-Growthアルゴリズムで頻出アイテムセットを抽出(メモリ効率的)"""
    print(f"\nFP-Growthアルゴリズム実行中 (min_support={min_support})...")
    
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    
    print(f"頻出アイテムセット数: {len(frequent_itemsets)}")
    
    # アイテムセットのサイズを追加
    frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
    
    # サイズ別の統計
    print("\nアイテムセットサイズ別の統計:")
    size_counts = frequent_itemsets['length'].value_counts().sort_index()
    for size, count in size_counts.items():
        print(f"  サイズ {size}: {count} 個")
    
    return frequent_itemsets


def extract_association_rules(frequent_itemsets: pd.DataFrame, 
                              min_confidence: float = 0.3,
                              min_lift: float = 1.2) -> pd.DataFrame:
    """アソシエーションルールを抽出"""
    print(f"\nアソシエーションルール抽出中...")
    print(f"  min_confidence={min_confidence}")
    print(f"  min_lift={min_lift}")
    
    rules = association_rules(frequent_itemsets, 
                              metric="confidence", 
                              min_threshold=min_confidence)
    
    # Lift値でフィルタリング
    rules = rules[rules['lift'] >= min_lift]
    
    print(f"抽出されたルール数: {len(rules)}")
    
    if len(rules) > 0:
        # ルールを読みやすい形式に変換
        rules['antecedents_str'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents_str'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))
        
        # Lift値でソート
        rules = rules.sort_values('lift', ascending=False)
    
    return rules


def filter_fatal_accident_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """死亡事故に関連するルールのみを抽出"""
    print("\n死亡事故関連ルールをフィルタリング中...")
    
    # 結果に「死亡事故あり」を含むルールのみ
    fatal_rules = rules[rules['consequents_str'].str.contains('死亡事故=死亡事故あり')]
    
    print(f"死亡事故関連ルール数: {len(fatal_rules)}")
    
    return fatal_rules


def save_results(frequent_itemsets: pd.DataFrame, 
                rules: pd.DataFrame,
                fatal_rules: pd.DataFrame,
                output_dir: Path):
    """結果を保存"""
    print("\n結果を保存中...")
    
    # 頻出アイテムセット
    itemsets_path = output_dir / "frequent_itemsets.csv"
    frequent_itemsets_save = frequent_itemsets.copy()
    frequent_itemsets_save['itemsets'] = frequent_itemsets_save['itemsets'].apply(lambda x: ', '.join(list(x)))
    frequent_itemsets_save.to_csv(itemsets_path, index=False, encoding='utf-8-sig')
    print(f"  頻出アイテムセット: {itemsets_path}")
    
    # 全ルール
    if len(rules) > 0:
        rules_path = output_dir / "association_rules.csv"
        rules_save = rules[['antecedents_str', 'consequents_str', 'support', 
                           'confidence', 'lift', 'leverage', 'conviction']].copy()
        rules_save.columns = ['条件', '結果', 'Support', 'Confidence', 
                             'Lift', 'Leverage', 'Conviction']
        rules_save.to_csv(rules_path, index=False, encoding='utf-8-sig')
        print(f"  全ルール: {rules_path}")
    
    # 死亡事故関連ルール
    if len(fatal_rules) > 0:
        fatal_rules_path = output_dir / "fatal_accident_rules.csv"
        fatal_rules_save = fatal_rules[['antecedents_str', 'consequents_str', 'support', 
                                        'confidence', 'lift', 'leverage', 'conviction']].copy()
        fatal_rules_save.columns = ['条件', '結果', 'Support', 'Confidence', 
                                    'Lift', 'Leverage', 'Conviction']
        fatal_rules_save.to_csv(fatal_rules_path, index=False, encoding='utf-8-sig')
        print(f"  死亡事故関連ルール: {fatal_rules_path}")
        
        # 上位ルール(Lift値順)
        top_rules_path = output_dir / "top_fatal_accident_rules.csv"
        top_rules = fatal_rules_save.head(50)
        top_rules.to_csv(top_rules_path, index=False, encoding='utf-8-sig')
        print(f"  上位50ルール: {top_rules_path}")


def print_top_rules(rules: pd.DataFrame, n: int = 10):
    """上位ルールを表示"""
    if len(rules) == 0:
        print("\nルールが見つかりませんでした。")
        return
    
    print(f"\n=== 上位{n}ルール (Lift値順) ===")
    print("-" * 100)
    
    for idx, row in rules.head(n).iterrows():
        print(f"\nルール {idx + 1}:")
        print(f"  条件: {row['antecedents_str']}")
        print(f"  結果: {row['consequents_str']}")
        print(f"  Support: {row['support']:.4f} | Confidence: {row['confidence']:.4f} | Lift: {row['lift']:.4f}")


def analyze_all_data(output_dir: Path, min_support: float, min_confidence: float, min_lift: float):
    """全データの分析(サンプリング使用)"""
    print("\n" + "=" * 100)
    print("全データのアソシエーション分析(サンプリング版)")
    print("=" * 100)
    
    # トランザクション読み込み
    transactions_path = output_dir / "transactions_all.csv"
    transactions = load_transactions(transactions_path)
    
    # メモリ節約のため10万件にサンプリング
    SAMPLE_SIZE = 100000
    print(f"\n⚠️ メモリ効率化のため、{SAMPLE_SIZE:,}件にサンプリングします")
    
    # One-Hot形式に変換(サンプリング)
    df = create_one_hot_dataframe(transactions, sample_size=SAMPLE_SIZE)
    
    # FP-Growth実行
    frequent_itemsets = run_fpgrowth(df, min_support=min_support)
    
    # ルール抽出
    rules = extract_association_rules(frequent_itemsets, 
                                     min_confidence=min_confidence,
                                     min_lift=min_lift)
    
    # 死亡事故関連ルールのフィルタリング
    fatal_rules = filter_fatal_accident_rules(rules)
    
    # 結果保存
    save_results(frequent_itemsets, rules, fatal_rules, output_dir)
    
    # 上位ルール表示
    print_top_rules(fatal_rules, n=10)
    
    return frequent_itemsets, rules, fatal_rules


def analyze_fatal_only(output_dir: Path, min_support: float, min_confidence: float, min_lift: float):
    """死亡事故のみの分析"""
    print("\n" + "=" * 100)
    print("死亡事故のみのアソシエーション分析")
    print("=" * 100)
    
    # トランザクション読み込み
    transactions_path = output_dir / "transactions_fatal.csv"
    transactions = load_transactions(transactions_path)
    
    # One-Hot形式に変換(死亡事故は16,267件なのでサンプリング不要)
    df = create_one_hot_dataframe(transactions)
    
    # FP-Growth実行
    frequent_itemsets = run_fpgrowth(df, min_support=min_support)
    
    # ルール抽出
    rules = extract_association_rules(frequent_itemsets, 
                                     min_confidence=min_confidence,
                                     min_lift=min_lift)
    
    # 結果保存
    if len(rules) > 0:
        rules_path = output_dir / "fatal_only_rules.csv"
        rules_save = rules[['antecedents_str', 'consequents_str', 'support', 
                           'confidence', 'lift', 'leverage', 'conviction']].copy()
        rules_save.columns = ['条件', '結果', 'Support', 'Confidence', 
                             'Lift', 'Leverage', 'Conviction']
        rules_save.to_csv(rules_path, index=False, encoding='utf-8-sig')
        print(f"\n死亡事故のみのルール保存: {rules_path}")
    
    # 上位ルール表示
    print_top_rules(rules, n=10)
    
    return frequent_itemsets, rules


def main():
    """メイン処理"""
    # パラメータ設定
    MIN_SUPPORT = 0.01      # 最小支持度 (1%)
    MIN_CONFIDENCE = 0.3    # 最小確信度 (30%)
    MIN_LIFT = 1.2          # 最小リフト値
    
    print("=" * 100)
    print("アソシエーション分析")
    print("=" * 100)
    print(f"\nパラメータ:")
    print(f"  最小支持度 (min_support): {MIN_SUPPORT}")
    print(f"  最小確信度 (min_confidence): {MIN_CONFIDENCE}")
    print(f"  最小リフト値 (min_lift): {MIN_LIFT}")
    
    # 出力ディレクトリ
    output_dir = project_root / "results" / "association_analysis"
    
    # 1. 全データの分析
    freq_all, rules_all, fatal_rules = analyze_all_data(
        output_dir, MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT
    )
    
    # 2. 死亡事故のみの分析
    freq_fatal, rules_fatal = analyze_fatal_only(
        output_dir, MIN_SUPPORT, MIN_CONFIDENCE, MIN_LIFT
    )
    
    print("\n" + "=" * 100)
    print("分析完了!")
    print("=" * 100)
    print(f"\n結果は以下に保存されました: {output_dir}")


if __name__ == "__main__":
    main()
