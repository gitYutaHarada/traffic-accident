"""
リークなしデータセット影響分析
==============================
'honhyo_for_analysis_with_traffic_no_leakage.csv' を使用して学習を行い、
これまでの実験結果との比較を行います。

実行方法:
    python scripts/experiment/compare_leakage_impact.py
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.modeling.train_two_stage_final import TwoStageFinalPipeline
import pandas as pd


def main():
    print("=" * 70)
    print("リークなしデータセット影響分析")
    print("=" * 70)
    
    # パイプライン設定
    pipeline = TwoStageFinalPipeline(
        data_path="data/processed/honhyo_for_analysis_with_traffic_no_leakage.csv",
        target_col="fatal",  # 新しいデータセットではターゲット列が `fatal`
        output_dir="results/leakage_impact_analysis",
        stage1_recall_target=0.99,
    )
    
    # 学習と評価を実行
    results = pipeline.run()
    
    print("\n" + "=" * 70)
    print("📊 結果サマリ")
    print("=" * 70)
    
    # ベースライン結果の読み込み（存在する場合）
    baseline_path = "results/two_stage_model/final_pipeline/final_results.csv"
    if os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)
        print("\n📈 ベースラインとの比較:")
        print("-" * 50)
        
        comparison_metrics = [
            ('final_precision', 'Precision (閾値0.5)'),
            ('final_recall', 'Recall (閾値0.5)'),
            ('final_f1', 'F1 Score'),
            ('final_auc', 'AUC'),
            ('test_precision', 'Test Precision'),
            ('test_recall', 'Test Recall'),
            ('test_f1', 'Test F1'),
            ('test_auc', 'Test AUC'),
        ]
        
        print(f"{'指標':<25} {'ベースライン':>12} {'リークなし':>12} {'差分':>10}")
        print("-" * 60)
        
        for metric_key, metric_name in comparison_metrics:
            baseline_val = baseline_df[metric_key].values[0] if metric_key in baseline_df.columns else None
            new_val = results.get(metric_key, None)
            
            if baseline_val is not None and new_val is not None:
                diff = new_val - baseline_val
                diff_str = f"{diff:+.4f}"
                print(f"{metric_name:<25} {baseline_val:>12.4f} {new_val:>12.4f} {diff_str:>10}")
            elif new_val is not None:
                print(f"{metric_name:<25} {'N/A':>12} {new_val:>12.4f} {'N/A':>10}")
    else:
        print("\n⚠️ ベースライン結果が見つかりません。")
        print(f"   期待されるパス: {baseline_path}")
        print("\n📈 新データセットでの結果:")
        print("-" * 50)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    print("✅ 分析完了！")
    print(f"   詳細レポート: results/leakage_impact_analysis/experiment_report.md")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    main()
