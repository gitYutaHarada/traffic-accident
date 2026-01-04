"""
リークなしデータセット影響分析（病院データ追加版）
=================================================
'honhyo_for_analysis_with_traffic_hospital_no_leakage.csv' を使用して学習を行い、
これまでの実験結果との比較を行います。

実行方法:
    python scripts/experiment/compare_leakage_impact_with_hospital.py
"""

import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.modeling.train_two_stage_final import TwoStageFinalPipeline
import pandas as pd


def main():
    print("=" * 70)
    print("リークなしデータセット影響分析（病院データ追加版）")
    print("=" * 70)

    # データファイルの存在確認
    data_path = "data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv"
    if not os.path.exists(data_path):
        print(f"\n❌ エラー: データファイルが見つかりません。")
        print(f"   期待されるパス: {data_path}")
        print(f"\n💡 先に以下のコマンドを実行してください:")
        print(f"   python scripts/data_processing/add_hospital_features.py")
        return

    # パイプライン設定
    pipeline = TwoStageFinalPipeline(
        data_path=data_path,
        target_col="fatal",  # ターゲット列
        output_dir="results/leakage_impact_analysis_with_hospital",
        stage1_recall_target=0.99,
    )

    # 学習と評価を実行
    results = pipeline.run()

    print("\n" + "=" * 70)
    print("📊 結果サマリ")
    print("=" * 70)

    # ベースライン結果の読み込み（存在する場合）
    baseline_path = "results/two_stage_model/final_pipeline/final_results.csv"
    traffic_only_path = "results/leakage_impact_analysis/final_results.csv"

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

    # 比較テーブル作成
    print("\n📈 3データセット比較:")
    print("-" * 80)
    print(f"{'指標':<25} {'ベースライン':>12} {'交通量のみ':>12} {'交通量+病院':>12}")
    print("-" * 80)

    baseline_df = None
    traffic_df = None

    if os.path.exists(baseline_path):
        baseline_df = pd.read_csv(baseline_path)
    if os.path.exists(traffic_only_path):
        traffic_df = pd.read_csv(traffic_only_path)

    for metric_key, metric_name in comparison_metrics:
        baseline_val = baseline_df[metric_key].values[0] if baseline_df is not None and metric_key in baseline_df.columns else None
        traffic_val = traffic_df[metric_key].values[0] if traffic_df is not None and metric_key in traffic_df.columns else None
        new_val = results.get(metric_key, None)

        baseline_str = f"{baseline_val:.4f}" if baseline_val is not None else "N/A"
        traffic_str = f"{traffic_val:.4f}" if traffic_val is not None else "N/A"
        new_str = f"{new_val:.4f}" if new_val is not None else "N/A"

        print(f"{metric_name:<25} {baseline_str:>12} {traffic_str:>12} {new_str:>12}")

    print("-" * 80)

    # 病院データの効果を計算
    if traffic_df is not None:
        print("\n📊 病院データ追加による変化（交通量のみ vs 交通量+病院）:")
        print("-" * 50)
        for metric_key, metric_name in comparison_metrics:
            traffic_val = traffic_df[metric_key].values[0] if metric_key in traffic_df.columns else None
            new_val = results.get(metric_key, None)

            if traffic_val is not None and new_val is not None:
                diff = new_val - traffic_val
                print(f"   {metric_name:<25}: {diff:+.4f}")

    print("\n" + "=" * 70)
    print("✅ 分析完了！")
    print(f"   詳細レポート: results/leakage_impact_analysis_with_hospital/experiment_report.md")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
