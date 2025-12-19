import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from scipy.stats import ks_2samp
import matplotlib as mpl

# 日本語フォント設定
mpl.rcParams['font.family'] = 'MS Gothic'
mpl.rcParams['axes.unicode_minus'] = False

class DistributionComparator:
    def __init__(self, tp_path, fp_path, output_dir):
        self.tp_path = tp_path
        self.fp_path = fp_path
        self.output_dir = output_dir
        self.img_dir = os.path.join(output_dir, 'plots')
        os.makedirs(self.img_dir, exist_ok=True)
        
        self.df_tp = None
        self.df_fp = None
        self.divergence_df = None

        # カテゴリとして扱うカラムの定義（analyze_false_positives_v2.pyより）
        self.known_categoricals = [
            '都道府県コード', '市区町村コード', '警察署等コード',
            '昼夜', '天候', '地形', '路面状態', '道路形状', '信号機',
            '衝突地点', 'ゾーン規制', '中央分離帯施設等', '歩車道区分',
            '事故類型', '曜日(発生年月日)', '祝日(発生年月日)',
            'road_type', 'area_id', '地点コード',
            # 追加でカテゴリとみなすべきもの
            '当事者種別（当事者A）', '当事者種別（当事者B）', 
            '事故内容', '車両の損壊程度（当事者A）', '車両の損壊程度（当事者B）',
            'エアバッグの装備（当事者A）', 'エアバッグの装備（当事者B）',
            'サイドエアバッグの装備（当事者A）', 'サイドエアバッグの装備（当事者B）',
            '用途別（当事者A）', '用途別（当事者B）',
            '車両の衝突部位（当事者A）', '車両の衝突部位（当事者B）',
            '速度規制（指定のみ）（当事者A）', '速度規制（指定のみ）（当事者B）',
            '一時停止規制　標識（当事者A）', '一時停止規制　標識（当事者B）',
            '一時停止規制　表示（当事者A）', '一時停止規制　表示（当事者B）'
        ]

    def load_data(self):
        print(f"Loading TP: {self.tp_path}")
        self.df_tp = pd.read_csv(self.tp_path)
        print(f"Loading FP: {self.fp_path}")
        self.df_fp = pd.read_csv(self.fp_path)
        
        # 不要なカラムの削除
        drop_cols = ['index', 'Unnamed: 0', 'pred_prob', 'true_label', 'pred_label', '発生日時']
        for col in drop_cols:
            if col in self.df_tp.columns:
                self.df_tp = self.df_tp.drop(columns=[col])
            if col in self.df_fp.columns:
                self.df_fp = self.df_fp.drop(columns=[col])
                
        print(f"TP columns: {len(self.df_tp.columns)}, Rows: {len(self.df_tp)}")
        print(f"FP columns: {len(self.df_fp.columns)}, Rows: {len(self.df_fp)}")

    def _calculate_numerical_divergence(self, col):
        """KS統計量 (0.0 - 1.0) を計算。大きいほど分布が異なる"""
        try:
            val_tp = self.df_tp[col].dropna()
            val_fp = self.df_fp[col].dropna()
            if len(val_tp) == 0 or len(val_fp) == 0:
                return 0.0
            stat, _ = ks_2samp(val_tp, val_fp)
            return stat
        except Exception:
            return 0.0

    def _calculate_categorical_divergence(self, col):
        """Total Variation Distance (0.0 - 1.0) を計算"""
        try:
            # 頻度分布（正規化）
            tp_counts = self.df_tp[col].value_counts(normalize=True)
            fp_counts = self.df_fp[col].value_counts(normalize=True)
            
            # 和集合のインデックスを取得
            all_cats = list(set(tp_counts.index) | set(fp_counts.index))
            
            tvd = 0.0
            for cat in all_cats:
                p = tp_counts.get(cat, 0.0)
                q = fp_counts.get(cat, 0.0)
                tvd += abs(p - q)
            
            return tvd / 2.0
        except Exception:
            return 0.0

    def analyze_divergence(self):
        print("\nCalculating divergence for all features...")
        results = []
        
        for col in self.df_tp.columns:
            # 数値かカテゴリか判定
            is_categorical = False
            if col in self.known_categoricals or self.df_tp[col].dtype == 'object':
                is_categorical = True
            elif self.df_tp[col].nunique() < 15: # ユニーク数が少ない数値はカテゴリ扱い
                is_categorical = True
            
            if is_categorical:
                score = self._calculate_categorical_divergence(col)
                dtype_str = 'Categorical'
            else:
                score = self._calculate_numerical_divergence(col)
                dtype_str = 'Numerical'
                
            results.append({
                'Feature': col,
                'Divergence_Score': score,
                'Type': dtype_str
            })
            
        self.divergence_df = pd.DataFrame(results).sort_values('Divergence_Score', ascending=False)
        csv_path = os.path.join(self.output_dir, 'feature_divergence_ranking.csv')
        self.divergence_df.to_csv(csv_path, index=False)
        print(f"Saved ranking to: {csv_path}")
        print("\nTop 10 Divergent Features:")
        print(self.divergence_df.head(10))

    def plot_top_features(self, top_n=20):
        print(f"\nPlotting top {top_n} features...")
        top_features = self.divergence_df.head(top_n)['Feature'].tolist()
        
        for feature in top_features:
            ftype = self.divergence_df[self.divergence_df['Feature'] == feature]['Type'].iloc[0]
            score = self.divergence_df[self.divergence_df['Feature'] == feature]['Divergence_Score'].iloc[0]
            
            plt.figure(figsize=(10, 6))
            
            if ftype == 'Numerical':
                # KDE Plot
                try:
                    sns.kdeplot(data=self.df_tp, x=feature, label='True Positive (TP)', fill=True, common_norm=False, color='blue', alpha=0.3)
                    sns.kdeplot(data=self.df_fp, x=feature, label='False Positive (FP)', fill=True, common_norm=False, color='red', alpha=0.3)
                except:
                    # Fallback to hist if kde fails (e.g. constant value)
                    plt.hist(self.df_tp[feature], bins=30, alpha=0.5, density=True, label='TP', color='blue')
                    plt.hist(self.df_fp[feature], bins=30, alpha=0.5, density=True, label='FP', color='red')
            else:
                # Categorical Bar Plot (Percentage)
                # データ結合してmeltする
                prop_tp = self.df_tp[feature].value_counts(normalize=True).rename('Percentage').reset_index()
                prop_tp['Group'] = 'TP'
                prop_fp = self.df_fp[feature].value_counts(normalize=True).rename('Percentage').reset_index()
                prop_fp['Group'] = 'FP'
                
                # index列の名前がpandasのバージョンによって違う可能性があるため調整
                col_name = prop_tp.columns[0]
                
                df_plot = pd.concat([prop_tp, prop_fp], axis=0)
                
                # カテゴリ数が多い場合はTop 15に絞る
                if df_plot[col_name].nunique() > 15:
                    top_cats = df_plot.groupby(col_name)['Percentage'].sum().sort_values(ascending=False).head(15).index
                    df_plot = df_plot[df_plot[col_name].isin(top_cats)]
                
                sns.barplot(data=df_plot, x=col_name, y='Percentage', hue='Group', palette={'TP': 'blue', 'FP': 'red'}, alpha=0.7)
                plt.xticks(rotation=45, ha='right')

            plt.title(f"Distribution Comparison: {feature} (Score: {score:.4f})")
            plt.legend()
            plt.tight_layout()
            
            # ファイル名に使えない文字を置換
            safe_name = feature.replace('/', '_').replace(':', '_').replace('\\', '_')
            save_path = os.path.join(self.img_dir, f"{safe_name}.png")
            plt.savefig(save_path)
            plt.close()
            # print(f"Saved plot: {save_path}")

    def create_report(self):
        report_path = os.path.join(self.output_dir, 'tp_fp_comparison_report.md')
        
        md = "# TP vs FP 特徴量分布比較レポート\n\n"
        md += "Stage 2 モデルの精度向上に向け、True Positive（正解の死亡事故）と False Positive（誤検知）の分布の違いを分析しました。\n\n"
        
        md += "## 分析概要\n"
        md += f"- **TPデータ数**: {len(self.df_tp)}\n"
        md += f"- **FPデータ数**: {len(self.df_fp)}\n"
        md += "- **乖離度指標**: \n"
        md += "  - 数値変数: KS統計量 (0.0 - 1.0)\n"
        md += "  - カテゴリ変数: Total Variation Distance (0.0 - 1.0)\n\n"
        
        md += "## 乖離度が大きい特徴量 Top 20\n"
        md += "スコアが高いほど、TPとFPで分布が異なっています（＝モデルが誤検知する要因、あるいは区別に重要な特徴量）。\n\n"
        md += "| Rank | Feature | Score | Type |\n"
        md += "| :--- | :--- | :--- | :--- |\n"
        
        for i, row in self.divergence_df.head(20).iterrows():
            md += f"| {i+1} | {row['Feature']} | {row['Divergence_Score']:.4f} | {row['Type']} |\n"
            
        md += "\n## 分布プロット\n"
        md += "上位の特徴量の分布比較グラフ。\n\n"
        
        md += "````carousel\n"
        for i, row in self.divergence_df.head(20).iterrows():
            feature = row['Feature']
            safe_name = feature.replace('/', '_').replace(':', '_').replace('\\', '_')
            img_path = os.path.join('plots', f"{safe_name}.png")
            
            md += f"![{feature}]({img_path})\n"
            md += f"> **{feature}** (Score: {row['Divergence_Score']:.4f})\n"
            if i < 19:
                md += "<!-- slide -->\n"
        md += "````\n"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(md)
        print(f"\nReport generated: {report_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp_path', required=True)
    parser.add_argument('--fp_path', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    
    analyzer = DistributionComparator(args.tp_path, args.fp_path, args.output_dir)
    analyzer.load_data()
    analyzer.analyze_divergence()
    analyzer.plot_top_features(top_n=20)
    analyzer.create_report()

if __name__ == '__main__':
    main()
