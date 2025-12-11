"""
ロジスティック回帰 vs LightGBM 比較分析 - 統合実行スクリプト

Phase 1〜3のスクリプトをワンコマンドで実行します。

実行内容:
1. ロジスティック回帰の訓練と評価
2. LightGBMとの統合比較
3. 可視化の生成

使用方法:
    python scripts/model_comparison/run_full_comparison.py
"""

import subprocess
import sys
import os
from datetime import datetime
import time


def run_command(description, command, cwd='.'):
    """コマンドを実行"""
    print("\n" + "="*80)
    print(f"[実行] {description}")
    print("="*80)
    print(f"コマンド: {command}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            check=True,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ 完了（所要時間: {elapsed_time:.1f}秒）")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ エラー発生（所要時間: {elapsed_time:.1f}秒）")
        print(f"エラーメッセージ: {e}")
        return False


def main():
    """メイン処理"""
    print("="*80)
    print("ロジスティック回帰 vs LightGBM 比較分析 - 統合実行")
    print("="*80)
    print(f"開始時刻: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    
    overall_start = time.time()
    
    # カレントディレクトリを確認
    cwd = os.getcwd()
    print(f"\nカレントディレクトリ: {cwd}")
    
    # Phase 1: ロジスティック回帰の訓練と評価
    success1 = run_command(
        "Phase 1: ロジスティック回帰の訓練と評価",
        "python scripts/model_comparison/train_logistic_regression_updated.py"
    )
    
    if not success1:
        print("\n❌ Phase 1でエラーが発生しました。処理を中断します。")
        return
    
    # Phase 2: 統合比較
    success2 = run_command(
        "Phase 2: ロジスティック回帰 vs LightGBM 統合比較",
        "python scripts/model_comparison/compare_models.py"
    )
    
    if not success2:
        print("\n❌ Phase 2でエラーが発生しました。処理を中断します。")
        return
    
    # Phase 3: 可視化
    success3 = run_command(
        "Phase 3: 比較結果の可視化",
        "python scripts/model_comparison/visualize_comparison.py"
    )
    
    if not success3:
        print("\n⚠️ Phase 3でエラーが発生しましたが、主要な処理は完了しています。")
    
    # 完了メッセージ
    overall_elapsed = time.time() - overall_start
    
    print("\n" + "="*80)
    print("🎉 すべての処理が完了しました！")
    print("="*80)
    print(f"総所要時間: {overall_elapsed/60:.1f}分")
    print(f"完了時刻: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}")
    
    print("\n[出力ファイル]")
    print("  - ロジスティック回帰結果: results/model_comparison/logistic_regression_updated/")
    print("  - 比較レポート: results/model_comparison/comparison_report_*.md")
    print("  - 可視化: results/model_comparison/visualizations/")
    
    print("\n[次のステップ]")
    print("  1. 比較レポート（comparison_report_*.md）を確認")
    print("  2. 可視化（visualizations/）を確認")
    print("  3. LightGBMの優位性を確認して、モデル選択を決定")


if __name__ == '__main__':
    main()
