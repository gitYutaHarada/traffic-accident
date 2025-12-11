"""
モデル比較の可視化スクリプト

ロジスティック回帰 vs LightGBM の比較結果を可視化します。

生成する可視化:
1. 評価指標の比較棒グラフ
2. PR曲線の重ね合わせ
3. ROC曲線の重ね合わせ
4. Box Plot（CV結果の分布）
5. 混同行列の並列表示
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style('whitegrid')


class ComparisonVisualizer:
    """モデル比較の可視化"""
    
    def __init__(self, logreg_csv, lightgbm_csv, output_dir='results/model_comparison/visualizations'):
        """
        Parameters:
        -----------
        logreg_csv : str
            ロジスティック回帰のCV結果CSVパス
        lightgbm_csv : str
            LightGBMのCV結果CSVパス
        output_dir : str
            可視化の出力先ディレクトリ
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # データ読み込み
        print("="*80)
        print("モデル比較の可視化")
        print("="*80)
        print(f"\n[読み込み] ロジスティック回帰: {logreg_csv}")
        print(f"[読み込み] LightGBM: {lightgbm_csv}")
        
        self.logreg_df = pd.read_csv(logreg_csv)
        self.lightgbm_df = pd.read_csv(lightgbm_csv)
        
        print(f"✅ 読み込み完了")
        
    def plot_metrics_comparison(self):
        """評価指標の比較棒グラフ"""
        print("\n[生成] 評価指標の比較棒グラフ...")
        
        metrics = ['pr_auc', 'roc_auc', 'f1', 'accuracy', 'precision', 'recall']
        metric_names = ['PR-AUC', 'ROC-AUC', 'F1 Score', 'Accuracy', 'Precision', 'Recall']
        
        # 平均値と標準偏差を計算
        logreg_means = [self.logreg_df[m].mean() for m in metrics]
        logreg_stds = [self.logreg_df[m].std() for m in metrics]
        lightgbm_means = [self.lightgbm_df[m].mean() for m in metrics]
        lightgbm_stds = [self.lightgbm_df[m].std() for m in metrics]
        
        # グラフ作成
        x = np.arange(len(metric_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bars1 = ax.bar(x - width/2, logreg_means, width, 
                      yerr=logreg_stds, label='ロジスティック回帰',
                      color='skyblue', capsize=5, alpha=0.8)
        bars2 = ax.bar(x + width/2, lightgbm_means, width,
                      yerr=lightgbm_stds, label='LightGBM',
                      color='coral', capsize=5, alpha=0.8)
        
        ax.set_xlabel('評価指標', fontsize=13, fontweight='bold')
        ax.set_ylabel('スコア', fontsize=13, fontweight='bold')
        ax.set_title('ロジスティック回帰 vs LightGBM - 評価指標比較', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, fontsize=11)
        ax.legend(fontsize=12, loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        
        # 値をバーの上に表示
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        output_path = f'{self.output_dir}/metrics_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 保存: {output_path}")
        return output_path
    
    def plot_box_plots(self):
        """Box Plot（CV結果の分布）"""
        print("\n[生成] Box Plot（CV結果の分布）...")
        
        metrics = ['pr_auc', 'roc_auc', 'f1']
        metric_names = ['PR-AUC', 'ROC-AUC', 'F1 Score']
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            
            # データ準備
            data = pd.DataFrame({
                'ロジスティック回帰': self.logreg_df[metric],
                'LightGBM': self.lightgbm_df[metric]
            })
            
            # Box Plot
            bp = ax.boxplot([data['ロジスティック回帰'], data['LightGBM']],
                           labels=['ロジスティック回帰', 'LightGBM'],
                           patch_artist=True,
                           widths=0.6)
            
            # 色設定
            colors = ['skyblue', 'coral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # 個別のデータポイントを重ねて表示
            for i, col in enumerate(['ロジスティック回帰', 'LightGBM']):
                y = data[col]
                x = np.random.normal(i+1, 0.04, size=len(y))
                ax.scatter(x, y, alpha=0.6, s=80, color=colors[i], edgecolors='black', linewidth=0.5)
            
            ax.set_ylabel('スコア', fontsize=11, fontweight='bold')
            ax.set_title(name, fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        plt.suptitle('モデル間のスコア分布比較 (5-fold CV)', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        output_path = f'{self.output_dir}/box_plots.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 保存: {output_path}")
        return output_path
    
    def plot_training_time_comparison(self):
        """訓練時間と予測時間の比較"""
        print("\n[生成] 訓練時間・予測時間の比較...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 訓練時間
        ax1 = axes[0]
        train_times = [
            self.logreg_df['train_time'].mean(),
            self.lightgbm_df['train_time'].mean()
        ]
        train_stds = [
            self.logreg_df['train_time'].std(),
            self.lightgbm_df['train_time'].std()
        ]
        
        bars1 = ax1.bar(['ロジスティック回帰', 'LightGBM'], train_times,
                       yerr=train_stds, capsize=5, 
                       color=['skyblue', 'coral'], alpha=0.8)
        ax1.set_ylabel('時間 (秒)', fontsize=11, fontweight='bold')
        ax1.set_title('訓練時間の比較', fontsize=12, fontweight='bold')
        ax1.grid(axis='y', alpha=0.3)
        
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}秒',
                    ha='center', va='bottom', fontsize=10)
        
        # 予測時間
        ax2 = axes[1]
        pred_times = [
            self.logreg_df['pred_time'].mean(),
            self.lightgbm_df['pred_time'].mean()
        ]
        pred_stds = [
            self.logreg_df['pred_time'].std(),
            self.lightgbm_df['pred_time'].std()
        ]
        
        bars2 = ax2.bar(['ロジスティック回帰', 'LightGBM'], pred_times,
                       yerr=pred_stds, capsize=5,
                       color=['skyblue', 'coral'], alpha=0.8)
        ax2.set_ylabel('時間 (秒)', fontsize=11, fontweight='bold')
        ax2.set_title('予測時間の比較', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}秒',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        output_path = f'{self.output_dir}/time_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 保存: {output_path}")
        return output_path
    
    def generate_all_visualizations(self):
        """すべての可視化を生成"""
        print("\n[開始] 可視化生成")
        print("="*80)
        
        outputs = {}
        
        # 1. 評価指標の比較棒グラフ
        outputs['metrics_comparison'] = self.plot_metrics_comparison()
        
        # 2. Box Plot
        outputs['box_plots'] = self.plot_box_plots()
        
        # 3. 訓練時間・予測時間の比較
        outputs['time_comparison'] = self.plot_training_time_comparison()
        
        print("\n" + "="*80)
        print("✅ すべての可視化を生成完了！")
        print("="*80)
        print(f"\n出力先: {self.output_dir}")
        for name, path in outputs.items():
            print(f"  - {name}: {path}")
        
        return outputs


def main():
    """メイン処理"""
    import argparse
    
    parser = argparse.ArgumentParser(description='モデル比較の可視化')
    parser.add_argument('--logreg-csv', type=str, 
                       default='results/model_comparison/logreg_cv_results_*.csv',
                       help='ロジスティック回帰のCV結果CSV')
    parser.add_argument('--lightgbm-csv', type=str,
                       default='results/model_comparison/lightgbm_cv_results_*.csv',
                       help='LightGBMのCV結果CSV')
    parser.add_argument('--output-dir', type=str,
                       default='results/model_comparison/visualizations',
                       help='出力ディレクトリ')
    
    args = parser.parse_args()
    
    # ワイルドカードを解決（最新のファイルを使用）
    import glob
    
    if '*' in args.logreg_csv:
        logreg_files = sorted(glob.glob(args.logreg_csv))
        if logreg_files:
            args.logreg_csv = logreg_files[-1]
            print(f"[自動選択] ロジスティック回帰: {args.logreg_csv}")
    
    if '*' in args.lightgbm_csv:
        lightgbm_files = sorted(glob.glob(args.lightgbm_csv))
        if lightgbm_files:
            args.lightgbm_csv = lightgbm_files[-1]
            print(f"[自動選択] LightGBM: {args.lightgbm_csv}")
    
    # 可視化器の初期化
    visualizer = ComparisonVisualizer(
        logreg_csv=args.logreg_csv,
        lightgbm_csv=args.lightgbm_csv,
        output_dir=args.output_dir
    )
    
    # すべての可視化を生成
    outputs = visualizer.generate_all_visualizations()


if __name__ == '__main__':
    main()
