"""
交互作用特徴量分析 統合パイプライン

3つのスクリプトを統合し、ワンコマンドで実行できるパイプライン:
1. generate_interaction_features.py: 交互作用特徴量の生成
2. evaluate_interaction_importance.py: LightGBMで重要度評価
3. generate_ranking_report.py: ランキングレポート生成

使用方法:
    python scripts/feature_engineering/run_interaction_analysis.py

オプション:
    --skip-generation: 特徴量生成をスキップ（既に生成済みの場合）
    --interaction-dir: 既存の交互作用特徴量ディレクトリを指定
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import subprocess
import time


class InteractionAnalysisPipeline:
    """交互作用特徴量分析の統合パイプライン"""
    
    def __init__(
        self,
        data_path='data/processed/honhyo_clean_predictable_only.csv',
        target_column='死者数',
        output_base_dir='results/interaction_features',
        skip_generation=False,
        interaction_dir=None
    ):
        """
        Parameters:
        -----------
        data_path : str
            元データのパス
        target_column : str
            目的変数のカラム名
        output_base_dir : str
            結果の出力先ディレクトリ
        skip_generation : bool
            特徴量生成をスキップするか
        interaction_dir : str or None
            既存の交互作用特徴量ディレクトリ（skip_generation=Trueの場合に指定）
        """
        self.data_path = data_path
        self.target_column = target_column
        self.output_base_dir = Path(output_base_dir)
        self.skip_generation = skip_generation
        self.interaction_dir = Path(interaction_dir) if interaction_dir else None
        
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # ログファイル
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_base_dir / f'pipeline_log_{self.timestamp}.txt'
        
    def log(self, message):
        """ログ出力"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_message + '\n')
    
    def step1_generate_features(self):
        """ステップ1: 交互作用特徴量の生成"""
        if self.skip_generation:
            self.log("ステップ1: スキップ（既存の交互作用特徴量を使用）")
            if not self.interaction_dir or not self.interaction_dir.exists():
                raise ValueError("interaction_dir が指定されていないか、存在しません")
            return str(self.interaction_dir)
        
        self.log("="*60)
        self.log("ステップ1: 交互作用特徴量の生成")
        self.log("="*60)
        
        start_time = time.time()
        
        # スクリプトをインポートして実行
        try:
            sys.path.insert(0, 'scripts/feature_engineering')
            from generate_interaction_features import InteractionFeatureGenerator
            
            # 出力ディレクトリ
            interaction_output_dir = f'data/interaction_features_{self.timestamp}'
            
            # 生成器の初期化
            generator = InteractionFeatureGenerator(
                data_path=self.data_path,
                target_column=self.target_column,
                output_dir=interaction_output_dir
            )
            
            # 交互作用特徴量を生成
            metadata_df = generator.generate_all_interactions()
            
            elapsed_time = time.time() - start_time
            self.log(f"✅ ステップ1完了（所要時間: {elapsed_time/60:.1f}分）")
            
            return interaction_output_dir
            
        except Exception as e:
            self.log(f"❌ ステップ1でエラー: {e}")
            raise
    
    def step2_evaluate_importance(self, interaction_dir):
        """ステップ2: LightGBMで重要度評価"""
        self.log("="*60)
        self.log("ステップ2: LightGBMで重要度評価")
        self.log("="*60)
        self.log("⚠️ この処理は数時間〜数十時間かかる可能性があります")
        
        start_time = time.time()
        
        try:
            sys.path.insert(0, 'scripts/feature_engineering')
            from evaluate_interaction_importance import InteractionFeatureEvaluator
            
            # メタデータパス
            metadata_path = Path(interaction_dir) / 'interaction_features_metadata.csv'
            
            # 評価器の初期化
            evaluator = InteractionFeatureEvaluator(
                data_path=self.data_path,
                interaction_metadata_path=str(metadata_path),
                interaction_dir=interaction_dir,
                target_column=self.target_column,
                n_folds=5,
                random_state=42
            )
            
            # ベースライン評価
            baseline_scores = evaluator.evaluate_baseline()
            
            # すべての交互作用特徴量を評価
            results_df = evaluator.evaluate_all_interactions(baseline_scores)
            
            # 結果を保存
            full_csv, top100_csv = evaluator.save_results(
                results_df, 
                output_dir=str(self.output_base_dir)
            )
            
            elapsed_time = time.time() - start_time
            self.log(f"✅ ステップ2完了（所要時間: {elapsed_time/60:.1f}分）")
            
            return full_csv
            
        except Exception as e:
            self.log(f"❌ ステップ2でエラー: {e}")
            raise
    
    def step3_generate_report(self, ranking_csv):
        """ステップ3: ランキングレポート生成"""
        self.log("="*60)
        self.log("ステップ3: ランキングレポート生成")
        self.log("="*60)
        
        start_time = time.time()
        
        try:
            sys.path.insert(0, 'scripts/feature_engineering')
            from generate_ranking_report import RankingReportGenerator
            
            # レポート生成器の初期化
            generator = RankingReportGenerator(
                ranking_csv_path=ranking_csv,
                output_dir=str(self.output_base_dir)
            )
            
            # すべてのレポートを生成
            reports = generator.generate_all_reports()
            
            elapsed_time = time.time() - start_time
            self.log(f"✅ ステップ3完了（所要時間: {elapsed_time:.1f}秒）")
            
            return reports
            
        except Exception as e:
            self.log(f"❌ ステップ3でエラー: {e}")
            raise
    
    def run(self):
        """パイプライン全体を実行"""
        self.log("="*60)
        self.log("交互作用特徴量分析パイプライン 開始")
        self.log("="*60)
        self.log(f"データパス: {self.data_path}")
        self.log(f"出力先: {self.output_base_dir}")
        self.log(f"ログファイル: {self.log_file}")
        self.log("="*60)
        
        pipeline_start_time = time.time()
        
        try:
            # ステップ1: 特徴量生成
            interaction_dir = self.step1_generate_features()
            
            # ステップ2: 重要度評価
            ranking_csv = self.step2_evaluate_importance(interaction_dir)
            
            # ステップ3: レポート生成
            reports = self.step3_generate_report(ranking_csv)
            
            # 完了メッセージ
            total_elapsed_time = time.time() - pipeline_start_time
            
            self.log("\n" + "="*60)
            self.log("🎉 パイプライン完了！")
            self.log("="*60)
            self.log(f"総所要時間: {total_elapsed_time/3600:.2f}時間")
            self.log(f"\n生成されたファイル:")
            self.log(f"  - ランキングCSV: {ranking_csv}")
            self.log(f"  - Markdownレポート: {reports['markdown_report']}")
            self.log(f"  - 棒グラフ: {reports['bar_chart']}")
            self.log(f"  - ヒートマップ: {reports['heatmap']}")
            self.log(f"  - 分布プロット: {reports['distribution']}")
            self.log(f"\n次のステップ:")
            self.log(f"  1. レポートを確認: {reports['markdown_report']}")
            self.log(f"  2. Top 10の特徴量をモデルに追加")
            self.log(f"  3. モデルを再訓練してPR-AUCの向上を確認")
            
            return reports
            
        except Exception as e:
            self.log(f"\n❌ パイプライン実行中にエラーが発生しました: {e}")
            raise


def main():
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description='交互作用特徴量分析パイプライン',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 全ステップを実行
  python scripts/feature_engineering/run_interaction_analysis.py
  
  # 既存の交互作用特徴量を使用して評価のみ実行
  python scripts/feature_engineering/run_interaction_analysis.py \\
    --skip-generation \\
    --interaction-dir data/interaction_features_20251211_140000
        """
    )
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/processed/honhyo_clean_predictable_only.csv',
        help='元データのパス（デフォルト: data/processed/honhyo_clean_predictable_only.csv）'
    )
    
    parser.add_argument(
        '--target-column',
        type=str,
        default='死者数',
        help='目的変数のカラム名（デフォルト: 死者数）'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/interaction_features',
        help='結果の出力先ディレクトリ（デフォルト: results/interaction_features）'
    )
    
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='特徴量生成をスキップ（既に生成済みの場合）'
    )
    
    parser.add_argument(
        '--interaction-dir',
        type=str,
        default=None,
        help='既存の交互作用特徴量ディレクトリ（--skip-generation使用時に指定）'
    )
    
    args = parser.parse_args()
    
    # パイプラインの初期化
    pipeline = InteractionAnalysisPipeline(
        data_path=args.data_path,
        target_column=args.target_column,
        output_base_dir=args.output_dir,
        skip_generation=args.skip_generation,
        interaction_dir=args.interaction_dir
    )
    
    # 実行
    pipeline.run()


if __name__ == '__main__':
    main()
