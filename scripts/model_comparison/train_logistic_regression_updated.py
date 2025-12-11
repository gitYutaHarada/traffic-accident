"""
ロジスティック回帰による死亡事故予測モデル（更新版）

LightGBMと公平に比較するため、以下を統一:
- データセット: data/processed/honhyo_clean_predictable_only.csv
- 評価方法: 5-fold StratifiedKFold交差検証
- 評価指標: PR-AUC, ROC-AUC, F1, Accuracy, Precision, Recall
- クラス不均衡対策: class_weight='balanced'

更新内容（2025-12-11）:
- データセットをLightGBMと統一
- PR-AUCを主要評価指標として追加
- 発生日時カラムの除外処理を追加
- 出力先を updated ディレクトリに変更
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score  # PR-AUC
)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from tqdm import tqdm

warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'MS Gothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LogisticRegressionModel:
    """ロジスティック回帰モデルの訓練と評価"""
    
    def __init__(
        self, 
        data_path='data/processed/honhyo_clean_predictable_only.csv',
        target_column='死者数',
        n_folds=5,
        random_state=42
    ):
        """
        Parameters:
        -----------
        data_path : str
            データセットのパス
        target_column : str
            目的変数のカラム名
        n_folds : int
            交差検証のフォールド数
        random_state : int
            乱数シード
        """
        self.data_path = data_path
        self.target_column = target_column
        self.n_folds = n_folds
        self.random_state = random_state
        
        print("="*80)
        print("ロジスティック回帰モデル（LightGBM比較用）")
        print("="*80)
        
        # データ読み込み
        print(f"\n[データ読み込み] {data_path}")
        self.df = pd.read_csv(data_path)
        print(f"✅ 読み込み完了: {len(self.df):,} 件")
        
        # 前処理
        self._preprocess_data()
        
    def _preprocess_data(self):
        """データの前処理"""
        print("\n[前処理] データ準備中...")
        
        # 目的変数を分離
        self.y = self.df[self.target_column]
        self.X = self.df.drop(columns=[self.target_column])
        
        # 発生日時を除外（LightGBMと同じ）
        if '発生日時' in self.X.columns:
            self.X = self.X.drop(columns=['発生日時'])
            print("  - 発生日時カラムを除外")
        
        # 数値型とカテゴリ型の分類
        self.numeric_cols = self.X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = self.X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # カテゴリカル変数として扱うべき数値カラム
        explicit_cat_cols = [
            '都道府県コード', '路線コード', '地点コード', '市区町村コード',
            '昼夜', '天候', '地形', '路面状態', '道路形状', '信号機',
            '衝突地点', 'ゾーン規制', '中央分離帯施設等', '歩車道区分',
            '事故類型', '曜日(発生年月日)', '祝日(発生年月日)'
        ]
        
        # 実際に存在するカラムのみを対象
        explicit_cat_cols = [c for c in explicit_cat_cols if c in self.X.columns]
        
        # カテゴリカル変数リストを更新
        self.categorical_cols = list(set(self.categorical_cols + explicit_cat_cols))
        self.numeric_cols = [c for c in self.numeric_cols if c not in self.categorical_cols]
        
        print(f"  - 数値型特徴量: {len(self.numeric_cols)} 個")
        print(f"  - カテゴリカル特徴量: {len(self.categorical_cols)} 個")
        
        # カテゴリカル変数を文字列に統一
        for col in self.categorical_cols:
            if col in self.X.columns:
                self.X[col] = self.X[col].astype(str)
        
        # クラス不均衡比を計算
        pos_count = self.y.sum()
        neg_count = len(self.y) - pos_count
        self.class_imbalance_ratio = neg_count / pos_count
        
        print(f"\n[クラス不均衡]")
        print(f"  - Negative (0): {neg_count:,}")
        print(f"  - Positive (1): {pos_count:,}")
        print(f"  - 不均衡比: {self.class_imbalance_ratio:.2f}:1")
        
    def build_pipeline(self):
        """前処理とモデルのパイプライン構築"""
        # 数値型の前処理
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # カテゴリカル型の前処理
        # LightGBMと公平にするため、シンプルな処理のみ
        from sklearn.preprocessing import OrdinalEncoder
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        # 前処理の統合
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ],
            remainder='drop'
        )
        
        # モデルパイプライン
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(
                penalty='l2',
                C=1.0,
                solver='saga',
                max_iter=1000,
                class_weight='balanced',  # クラス不均衡対策
                random_state=self.random_state,
                n_jobs=-1,
                verbose=0
            ))
        ])
        
        return pipeline
    
    def cross_validate(self):
        """5-fold交差検証で評価"""
        print(f"\n[開始] {self.n_folds}-fold 交差検証")
        print("="*80)
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # 結果を格納
        fold_metrics = []
        y_true_all = []
        y_prob_all = []
        
        # 各foldの処理
        for fold, (train_idx, val_idx) in enumerate(tqdm(skf.split(self.X, self.y), total=self.n_folds, desc="Cross-Validation")):
            print(f"\n--- Fold {fold+1}/{self.n_folds} ---")
            
            X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
            y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            
            # パイプライン構築
            pipeline = self.build_pipeline()
            
            # 訓練
            print("  訓練中...")
            pipeline.fit(X_train, y_train)
            
            # 予測（確率）
            y_prob = pipeline.predict_proba(X_val)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            
            # 評価指標計算
            pr_auc = average_precision_score(y_val, y_prob)
            roc_auc = roc_auc_score(y_val, y_prob)
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            
            print(f"  PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f} | F1: {f1:.4f}")
            
            # 結果を保存
            fold_metrics.append({
                'Fold': fold + 1,
                'PR-AUC': pr_auc,
                'ROC-AUC': roc_auc,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            })
            
            # 全データに蓄積
            y_true_all.extend(y_val)
            y_prob_all.extend(y_prob)
        
        # 結果をDataFrameに変換
        self.fold_metrics = pd.DataFrame(fold_metrics)
        self.y_true_all = np.array(y_true_all)
        self.y_prob_all = np.array(y_prob_all)
        
        # 平均スコアを計算
        self.avg_metrics = self.fold_metrics.mean()
        
        print("\n" + "="*80)
        print("[結果] 5-fold CV 平均スコア")
        print("="*80)
        print(f"  PR-AUC:    {self.avg_metrics['PR-AUC']:.4f}")
        print(f"  ROC-AUC:   {self.avg_metrics['ROC-AUC']:.4f}")
        print(f"  Accuracy:  {self.avg_metrics['Accuracy']:.4f}")
        print(f"  Precision: {self.avg_metrics['Precision']:.4f}")
        print(f"  Recall:    {self.avg_metrics['Recall']:.4f}")
        print(f"  F1 Score:  {self.avg_metrics['F1 Score']:.4f}")
        
        return self.fold_metrics
    
    def save_results(self, output_dir='results/model_comparison/logistic_regression_updated'):
        """結果を保存"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        print(f"\n[保存] 結果を保存中: {output_dir}")
        
        # 1. Fold別の評価指標をCSV保存
        metrics_path = f'{output_dir}/metrics_{timestamp}.csv'
        self.fold_metrics.to_csv(metrics_path, index=False, encoding='utf-8-sig')
        print(f"  ✅ Fold別評価指標: {metrics_path}")
        
        # 2. PR曲線
        precisions, recalls, thresholds = precision_recall_curve(self.y_true_all, self.y_prob_all)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, marker='.', label=f'Logistic Regression (PR-AUC={self.avg_metrics["PR-AUC"]:.4f})')
        plt.xlabel('Recall (再現率)', fontsize=12)
        plt.ylabel('Precision (適合率)', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        pr_path = f'{output_dir}/pr_curve_{timestamp}.png'
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ PR曲線: {pr_path}")
        
        # 3. ROC曲線
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(self.y_true_all, self.y_prob_all)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, marker='.', label=f'Logistic Regression (ROC-AUC={self.avg_metrics["ROC-AUC"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        
        roc_path = f'{output_dir}/roc_curve_{timestamp}.png'
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ ROC曲線: {roc_path}")
        
        # 4. 混同行列
        y_pred = (self.y_prob_all >= 0.5).astype(int)
        cm = confusion_matrix(self.y_true_all, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['非死亡', '死亡'], yticklabels=['非死亡', '死亡'])
        plt.title('Confusion Matrix (Threshold=0.5)', fontsize=14, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        
        cm_path = f'{output_dir}/confusion_matrix_{timestamp}.png'
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ 混同行列: {cm_path}")
        
        # 5. サマリーレポート
        self._generate_summary_report(output_dir, timestamp)
        
        print(f"\n✅ すべての結果を保存完了: {output_dir}")
        
    def _generate_summary_report(self, output_dir, timestamp):
        """サマリーレポートを生成"""
        report_lines = [
            "# ロジスティック回帰 実験結果（更新版）",
            "",
            f"**実験日時**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}",
            "**目的**: LightGBMとの公平な比較",
            "",
            "---",
            "",
            "## 📊 実験設定",
            "",
            "### データセット",
            f"- ファイル: `{self.data_path}`",
            f"- 総データ数: {len(self.df):,} 件",
            f"- Positive (死亡事故): {self.y.sum():,} 件",
            f"- Negative (非死亡): {(self.y == 0).sum():,} 件",
            f"- クラス不均衡比: {self.class_imbalance_ratio:.2f}:1",
            "",
            "### 特徴量",
            f"- 数値型: {len(self.numeric_cols)} 個",
            f"- カテゴリカル型: {len(self.categorical_cols)} 個",
            f"- 総特徴量数: {len(self.X.columns)}",
            "",
            "### モデル設定",
            "```python",
            "LogisticRegression(",
            "    penalty='l2',",
            "    C=1.0,",
            "    solver='saga',",
            "    max_iter=1000,",
            "    class_weight='balanced',  # クラス不均衡対策",
            "    random_state=42",
            ")",
            "```",
            "",
            "---",
            "",
            "## 📈 評価結果",
            "",
            "### 5-fold CV 平均スコア",
            "",
            "| 指標 | スコア | 標準偏差 |",
            "|------|--------|----------|",
            f"| **PR-AUC** | **{self.avg_metrics['PR-AUC']:.4f}** | {self.fold_metrics['PR-AUC'].std():.4f} |",
            f"| **ROC-AUC** | {self.avg_metrics['ROC-AUC']:.4f} | {self.fold_metrics['ROC-AUC'].std():.4f} |",
            f"| **F1 Score** | {self.avg_metrics['F1 Score']:.4f} | {self.fold_metrics['F1 Score'].std():.4f} |",
            f"| **Accuracy** | {self.avg_metrics['Accuracy']:.4f} | {self.fold_metrics['Accuracy'].std():.4f} |",
            f"| **Precision** | {self.avg_metrics['Precision']:.4f} | {self.fold_metrics['Precision'].std():.4f} |",
            f"| **Recall** | {self.avg_metrics['Recall']:.4f} | {self.fold_metrics['Recall'].std():.4f} |",
            "",
            "### Fold別詳細",
            "",
            self.fold_metrics.to_markdown(index=False),
            "",
            "---",
            "",
            "## 💡 特徴",
            "",
            "### LightGBMとの違い",
            "- **前処理**: 数値型は標準化、カテゴリ型は順序エンコーディング",
            "- **モデル**: 線形モデル（特徴量間の複雑な相互作用を捉えにくい）",
            "- **クラス不均衡対策**: `class_weight='balanced'`",
            "",
            "### 長所",
            "- 解釈性が高い（係数から各特徴量の影響を読み取れる）",
            "- 訓練が高速",
            "- 過学習しにくい",
            "",
            "### 短所",
            "- 非線形な関係を捉えにくい",
            "- 特徴量間の相互作用を自動で学習できない",
            "",
            "---",
            "",
            f"**レポート作成日**: {datetime.now().strftime('%Y年%m月%d日 %H:%M')}  ",
            f"**作成者**: Antigravity AI Agent",
        ]
        
        report_path = f'{output_dir}/summary_report_{timestamp}.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"  ✅ サマリーレポート: {report_path}")


def main():
    """メイン処理"""
    # ロジスティック回帰モデルの初期化
    model = LogisticRegressionModel(
        data_path='data/processed/honhyo_clean_predictable_only.csv',
        target_column='死者数',
        n_folds=5,
        random_state=42
    )
    
    # 交差検証で評価
    fold_metrics = model.cross_validate()
    
    # 結果を保存
    model.save_results()
    
    print("\n" + "="*80)
    print("✅ ロジスティック回帰の訓練・評価が完了しました！")
    print("="*80)
    print("\n次のステップ: compare_models.py を実行してLightGBMと比較してください")


if __name__ == '__main__':
    main()
