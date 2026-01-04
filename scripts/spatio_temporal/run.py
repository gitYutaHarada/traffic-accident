"""
Spatio-Temporal Stage2 統合実行スクリプト
========================================
データ読み込みから地図可視化まで一括実行
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional
import warnings

warnings.filterwarnings('ignore')

# パスを追加
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))


def run_pipeline(
    data_path: str = "data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv",
    output_dir: str = "results/spatio_temporal",
    data_dir: Optional[str] = None,
    models: list = ['mlp', 'knn_gnn'],
    train_years: str = "2018,2019",
    val_years: str = "2020,2020",
    test_years: str = "2021,2024",
    run_optuna: bool = False,
    n_optuna_trials: int = 50,
    epochs: int = 100,
    batch_size: int = 1024,
    k_neighbors: int = 8,
    skip_preprocess: bool = False,
    skip_train: bool = False,
    skip_visualize: bool = False,
):
    """
    完全なパイプライン実行
    
    1. 前処理
    2. グラフ構築
    3. (オプション) Optuna探索
    4. モデル学習
    5. 評価
    6. 可視化
    7. レポート生成
    """
    
    print("=" * 70)
    print("🚀 Spatio-Temporal Stage2 Pipeline")
    print(f"   開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    start_time = datetime.now()
    
    # ディレクトリ設定
    # ディレクトリ設定
    if data_dir is None:
        data_dir = Path("data/spatio_temporal")
    else:
        data_dir = Path(data_dir)
    
    # データディレクトリを作成（なければ）
    data_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # =====================
    # 1. 前処理
    # =====================
    if not skip_preprocess:
        print("\n" + "=" * 50)
        print("📦 Step 1: 前処理")
        print("=" * 50)
        
        from preprocess_spatio_temporal import SpatioTemporalPreprocessor
        
        train_y = tuple(map(int, train_years.split(',')))
        val_y = tuple(map(int, val_years.split(',')))
        test_y = tuple(map(int, test_years.split(',')))
        
        preprocessor = SpatioTemporalPreprocessor(
            data_path=data_path,
            output_dir=str(data_dir),
            train_years=train_y,
            val_years=val_y,
            test_years=test_y,
        )
        
        preprocess_result = preprocessor.run()
        results['preprocess'] = preprocess_result
    else:
        print("\n⏭️ 前処理をスキップ")
    
    # =====================
    # 2. グラフ構築
    # =====================
    print("\n" + "=" * 50)
    print("🔗 Step 2: グラフ構築")
    print("=" * 50)
    
    import pandas as pd
    from graph_builder import build_geohash_level_graph
    
    train_df = pd.read_parquet(data_dir / "preprocessed_train.parquet")
    graph_data = build_geohash_level_graph(train_df, k=k_neighbors, output_dir=data_dir)
    
    results['graph'] = {
        'n_nodes': graph_data['n_nodes'],
        'n_edges': graph_data['edge_index'].shape[1],
    }
    
    # =====================
    # 3. Optuna探索（オプション）
    # =====================
    best_params = None
    if run_optuna:
        print("\n" + "=" * 50)
        print("🔍 Step 3: Optuna探索")
        print("=" * 50)
        
        from optuna_search import run_optuna_search
        
        best_params = run_optuna_search(
            data_dir=str(data_dir),
            output_dir=str(output_dir / "optuna"),
            model_type='knn_gnn',
            n_trials=n_optuna_trials,
            n_epochs=50,
        )
        
        results['optuna'] = best_params
    
    # =====================
    # 4. モデル学習
    # =====================
    if not skip_train:
        print("\n" + "=" * 50)
        print("🌿 Step 4: モデル学習")
        print("=" * 50)
        
        from train_spatio_temporal import SpatioTemporalTrainer
        
        model_results = {}
        
        for model_type in models:
            print(f"\n--- {model_type} ---")
            
            config = {
                'hidden_dim': best_params.get('hidden_dim', 128) if best_params else 128,
                'num_layers': best_params.get('num_layers', 2) if best_params else 2,
                'dropout': best_params.get('dropout', 0.3) if best_params else 0.3,
                'learning_rate': best_params.get('learning_rate', 0.001) if best_params else 0.001,
                'batch_size': batch_size,
                'epochs': epochs,
                'patience': 15,
                'focal_alpha': best_params.get('focal_alpha', 0.75) if best_params else 0.75,
                'focal_gamma': best_params.get('focal_gamma', 2.0) if best_params else 2.0,
                'k_neighbors': k_neighbors,
            }
            
            trainer = SpatioTemporalTrainer(
                data_dir=str(data_dir),
                output_dir=str(output_dir),
                model_type=model_type,
                config=config,
            )
            
            result = trainer.run()
            model_results[model_type] = result
        
        results['models'] = model_results
    else:
        print("\n⏭️ 学習をスキップ")
    
    # =====================
    # 5. 可視化
    # =====================
    if not skip_visualize:
        print("\n" + "=" * 50)
        print("📊 Step 5: 可視化")
        print("=" * 50)
        
        from visualize import Visualizer, plot_pr_curve, plot_roc_curve, create_heatmap
        import numpy as np
        
        visualizer = Visualizer(output_dir=str(output_dir))
        
        # 予測結果の読み込み
        test_pred_path = output_dir / "test_predictions.parquet"
        if test_pred_path.exists():
            test_df = pd.read_parquet(test_pred_path)
            
            # ヒートマップ生成
            if 'lat' in test_df.columns and 'lon' in test_df.columns:
                create_heatmap(
                    test_df.dropna(subset=['lat', 'lon', 'prediction']),
                    str(output_dir / "heatmap.html"),
                )
                
                # Top-N地図
                from visualize import create_top_n_map
                create_top_n_map(
                    test_df.dropna(subset=['lat', 'lon', 'prediction']),
                    str(output_dir / "top_n_map.html"),
                    n=100,
                )
            
            # PR/ROC曲線
            if 'fatal' in test_df.columns and 'prediction' in test_df.columns:
                y_true = test_df['fatal'].values
                y_pred = test_df['prediction'].values
                
                model_results_for_curves = {
                    'Spatio-Temporal': (y_true, y_pred)
                }
                
                plot_pr_curve(model_results_for_curves, str(output_dir / "pr_curve.png"))
                plot_roc_curve(model_results_for_curves, str(output_dir / "roc_curve.png"))
        
        print("   可視化完了")
    else:
        print("\n⏭️ 可視化をスキップ")
    
    # =====================
    # 6. レポート生成
    # =====================
    print("\n" + "=" * 50)
    print("📄 Step 6: レポート生成")
    print("=" * 50)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    # results_summary.json
    summary = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'data_path': data_path,
        'models': models,
        'results': results,
    }
    
    # モデル結果から主要指標を抽出
    if 'models' in results:
        for model_name, model_result in results['models'].items():
            if 'test_metrics' in model_result:
                summary[f'{model_name}_pr_auc'] = model_result['test_metrics'].get('pr_auc', 0)
                summary[f'{model_name}_roc_auc'] = model_result['test_metrics'].get('roc_auc', 0)
    
    with open(output_dir / "results_summary.json", 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    # Markdownレポート
    report = generate_markdown_report(results, elapsed, output_dir)
    
    with open(output_dir / "experiment_report.md", 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("\n" + "=" * 70)
    print("✅ パイプライン完了！")
    print(f"   所要時間: {elapsed:.1f}秒 ({elapsed/60:.1f}分)")
    print(f"   結果: {output_dir}")
    print("=" * 70)
    
    return results


def generate_markdown_report(results: dict, elapsed: float, output_dir: Path) -> str:
    """Markdownレポート生成"""
    
    report = []
    report.append("# Spatio-Temporal Stage2 実験レポート\n")
    report.append(f"**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**所要時間**: {elapsed:.1f}秒\n")
    
    # 前処理結果
    if 'preprocess' in results:
        report.append("\n## 1. 前処理\n")
        p = results['preprocess']
        report.append(f"- Train: {p.get('train_size', 0):,} 件\n")
        report.append(f"- Validation: {p.get('val_size', 0):,} 件\n")
        report.append(f"- Test: {p.get('test_size', 0):,} 件\n")
        report.append(f"- 特徴量数: {p.get('n_features', 0)}\n")
    
    # グラフ構築
    if 'graph' in results:
        report.append("\n## 2. グラフ構築\n")
        g = results['graph']
        report.append(f"- ノード数: {g.get('n_nodes', 0):,}\n")
        report.append(f"- エッジ数: {g.get('n_edges', 0):,}\n")
    
    # モデル結果
    if 'models' in results:
        report.append("\n## 3. モデル評価結果\n")
        report.append("\n| モデル | PR-AUC | ROC-AUC | ECE | Brier Score |\n")
        report.append("|--------|--------|---------|-----|-------------|\n")
        
        for model_name, model_result in results['models'].items():
            if 'test_metrics' in model_result:
                m = model_result['test_metrics']
                report.append(
                    f"| {model_name} | {m.get('pr_auc', 0):.4f} | "
                    f"{m.get('roc_auc', 0):.4f} | {m.get('ece', 0):.4f} | "
                    f"{m.get('brier_score', 0):.4f} |\n"
                )
        
        # Precision/Recall@k
        report.append("\n### Precision/Recall@k\n")
        report.append("\n| モデル | P@100 | R@100 | P@500 | R@500 |\n")
        report.append("|--------|-------|-------|-------|-------|\n")
        
        for model_name, model_result in results['models'].items():
            if 'test_metrics' in model_result:
                m = model_result['test_metrics']
                report.append(
                    f"| {model_name} | {m.get('precision_at_100', 0):.4f} | "
                    f"{m.get('recall_at_100', 0):.4f} | {m.get('precision_at_500', 0):.4f} | "
                    f"{m.get('recall_at_500', 0):.4f} |\n"
                )
    
    # 可視化ファイル
    report.append("\n## 4. 生成ファイル\n")
    report.append(f"- [ヒートマップ](heatmap.html)\n")
    report.append(f"- [Top-N地点マップ](top_n_map.html)\n")
    report.append(f"- [PR曲線](pr_curve.png)\n")
    report.append(f"- [ROC曲線](roc_curve.png)\n")
    report.append(f"- [結果サマリ](results_summary.json)\n")
    
    return "".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Spatio-Temporal Stage2 Complete Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 全工程実行
  python run.py --all
  
  # 前処理のみ
  python run.py --preprocess-only
  
  # 学習のみ（前処理済み）
  python run.py --skip-preprocess
  
  # Optuna探索付き
  python run.py --all --optuna
""")
    
    parser.add_argument('--all', action='store_true', help='全工程を実行')
    parser.add_argument('--data-path', type=str,
                        default="data/processed/honhyo_for_analysis_with_traffic_hospital_no_leakage.csv")
    parser.add_argument('--data-dir', type=str, default=None,
                        help="前処理済みデータの出力先ディレクトリ（デフォルト: data/spatio_temporal）")
    parser.add_argument('--output-dir', type=str, default="results/spatio_temporal")
    parser.add_argument('--models', type=str, default="mlp,knn_gnn",
                        help="学習するモデル（カンマ区切り）")
    parser.add_argument('--train-years', type=str, default="2018,2019")
    parser.add_argument('--val-years', type=str, default="2020,2020")
    parser.add_argument('--test-years', type=str, default="2021,2024")
    
    parser.add_argument('--optuna', action='store_true', help='Optuna探索を実行')
    parser.add_argument('--n-optuna-trials', type=int, default=50)
    
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--k', type=int, default=8, help='kNN graph k')
    
    parser.add_argument('--skip-preprocess', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-visualize', action='store_true')
    parser.add_argument('--preprocess-only', action='store_true')
    
    args = parser.parse_args()
    
    if args.preprocess_only:
        args.skip_train = True
        args.skip_visualize = True
    
    models = args.models.split(',')
    
    run_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        models=models,
        train_years=args.train_years,
        val_years=args.val_years,
        test_years=args.test_years,
        run_optuna=args.optuna,
        n_optuna_trials=args.n_optuna_trials,
        epochs=args.epochs,
        batch_size=args.batch_size,
        k_neighbors=args.k,
        skip_preprocess=args.skip_preprocess,
        skip_train=args.skip_train,
        skip_visualize=args.skip_visualize,
    )


if __name__ == "__main__":
    main()
