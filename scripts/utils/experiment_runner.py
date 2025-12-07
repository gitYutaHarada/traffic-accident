import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, roc_auc_score
)
import matplotlib as mpl

# 日本語フォントの設定
mpl.rcParams['font.family'] = 'MS Gothic'

class ExperimentRunner:
    def __init__(self, X, y, experiment_name, output_dir='results'):
        self.X = X
        self.y = y
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.viz_dir = os.path.join(output_dir, 'visualizations')
        self.analysis_dir = os.path.join(output_dir, 'analysis')
        
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)

    def run_cv(self, model_factory, k_folds=5):
        print(f"\n🔄 {k_folds}-fold 交差検証を開始: {self.experiment_name}")
        
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        fold_metrics = []
        y_true_all = []
        y_prob_all = []
        feature_importances = pd.DataFrame()

        for i, (train_index, val_index) in enumerate(skf.split(self.X, self.y)):
            print(f"\n--- Fold {i+1}/{k_folds} ---")
            
            X_train, X_val = self.X.iloc[train_index], self.X.iloc[val_index]
            y_train, y_val = self.y.iloc[train_index], self.y.iloc[val_index]
            
            # モデル生成と学習
            model = model_factory()
            model.fit(X_train, y_train)
            
            # 予測
            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_val)[:, 1]
            else:
                y_prob = model.decision_function(X_val) # For SVM etc.
                
            y_true_all.extend(y_val)
            y_prob_all.extend(y_prob)
            
            # 評価 (閾値0.5)
            y_pred = (y_prob >= 0.5).astype(int)
            metrics = self._calculate_metrics(y_val, y_pred)
            print(f"  [Threshold 0.5] Acc: {metrics['Accuracy']:.4f}, Prec: {metrics['Precision']:.4f}, Recall: {metrics['Recall']:.4f}, F1: {metrics['F1 Score']:.4f}")
            
            # Fold情報追加
            metrics['Fold'] = i + 1
            fold_metrics.append(metrics)
            
            # 特徴量重要度 (もし利用可能なら)
            # Pipelineの場合は最終ステップから取得
            estimator = model
            if hasattr(model, 'steps'):
                estimator = model.steps[-1][1]
            
            if hasattr(estimator, 'feature_importances_'):
                fi = pd.DataFrame()
                fi['feature'] = self.X.columns
                fi['importance'] = estimator.feature_importances_
                fi['fold'] = i + 1
                feature_importances = pd.concat([feature_importances, fi], axis=0)

        # 全体評価
        self._evaluate_global(y_true_all, y_prob_all, fold_metrics, feature_importances)

    def _calculate_metrics(self, y_true, y_pred):
        return {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'Recall': recall_score(y_true, y_pred, average='binary'),
            'F1 Score': f1_score(y_true, y_pred, average='binary')
        }

    def _evaluate_global(self, y_true, y_prob, fold_metrics, feature_importances):
        y_true = np.array(y_true)
        y_prob = np.array(y_prob)
        
        # AUC
        auc_score = roc_auc_score(y_true, y_prob)
        print(f"\n📈 Overall AUC Score: {auc_score:.4f}")
        
        # メトリクス保存
        metrics_df = pd.DataFrame(fold_metrics)
        metrics_path = os.path.join(self.analysis_dir, f'{self.experiment_name}_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"✓ 評価メトリクスを保存: {metrics_path}")

        # PR曲線
        self._plot_pr_curve(y_true, y_prob)
        
        # 混同行列
        y_pred = (y_prob >= 0.5).astype(int)
        self._plot_confusion_matrix(y_true, y_pred)
        
        # 特徴量重要度
        if not feature_importances.empty:
            self._save_feature_importance(feature_importances)

    def _plot_pr_curve(self, y_true, y_prob):
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recalls, precisions, marker='.', label=self.experiment_name)
        plt.xlabel('Recall (再現率)')
        plt.ylabel('Precision (適合率)')
        plt.title(f'Precision-Recall Curve - {self.experiment_name}')
        plt.legend()
        plt.grid(True)
        
        path = os.path.join(self.viz_dir, f'{self.experiment_name}_pr_curve.png')
        plt.savefig(path)
        plt.close()
        print(f"✓ PR曲線を保存: {path}")

    def _plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['非死亡', '死亡'], yticklabels=['非死亡', '死亡'])
        plt.title(f'Confusion Matrix - {self.experiment_name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        path = os.path.join(self.viz_dir, f'{self.experiment_name}_confusion_matrix.png')
        plt.savefig(path)
        plt.close()
        print(f"✓ 混同行列を保存: {path}")

    def _save_feature_importance(self, feature_importances):
        feat_imp_mean = feature_importances.groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        csv_path = os.path.join(self.analysis_dir, f'{self.experiment_name}_feature_importance.csv')
        feat_imp_mean.to_csv(csv_path)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=feat_imp_mean.head(20).values, y=feat_imp_mean.head(20).index, palette='viridis')
        plt.title(f'Feature Importance (Top 20) - {self.experiment_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        png_path = os.path.join(self.viz_dir, f'{self.experiment_name}_feature_importance.png')
        plt.savefig(png_path)
        plt.close()
        print(f"✓ 特徴量重要度を保存: {csv_path}, {png_path}")
