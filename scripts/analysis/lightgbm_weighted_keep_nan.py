import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
# 警告抑制（LightGBMのcategory型対応で警告が出ることがあるため）
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score
)
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os

# 日本語フォントの設定 (Windows向け)
mpl.rcParams['font.family'] = 'MS Gothic'

def main():
    """
    scale_pos_weight（重み付け）を使用し、Recall改善を目指すスクリプト
    SMOTEは使用せず、純粋な重み付けの効果を検証する
    """
    
    print("=" * 80)
    print("モデル改善実験: LightGBM + scale_pos_weight (重み付け)")
    print("=" * 80)
    
    # データ読み込み
    # データ読み込み（前処理済みのデータを使用）
    file_path = 'data/processed/honhyo_model_ready.csv'
    print(f"\n📂 データ読み込み中: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ データ読み込み完了: {len(df):,} 件")
    except Exception as e:
        print(f"❌ エラー: {e}")
        return
    
    # 目的変数
    target_col = '死者数'
    
    # 除外する列（事後情報・データリーク原因を徹底排除）
    drop_cols = [
        '資料区分', '本票番号',
        '人身損傷程度（当事者A）', '人身損傷程度（当事者B）',
        '車両の損壊程度（当事者A）', '車両の損壊程度（当事者B）',
        '負傷者数',
        '車両の衝突部位（当事者A）', '車両の衝突部位（当事者B）',
        'エアバッグの装備（当事者A）', 'エアバッグの装備（当事者B）',
        'サイドエアバッグの装備（当事者A）', 'サイドエアバッグの装備（当事者B）',
        '事故内容'  # データリーク原因
    ]
    
    print("\n🔧 データ前処理中（事後情報の除外）...")
    df_clean = df.drop(columns=drop_cols, errors='ignore')
    
    # 特徴量と目的変数
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # 数値列の欠損値処理
    # 数値列の欠損値処理
    # LightGBMは欠損値をそのまま扱えるため、埋めずにそのままにする (NaN維持)
    print("\n⚠️ 欠損値の穴埋め(Imputation)は行わず、NaNとして扱います")

    # カテゴリカル変数として扱う列を明示的に指定
    categorical_candidates = [
        '都道府県コード', '路線コード', '地点コード', '市区町村コード',
        '昼夜', '天候', '地形', '路面状態', '道路形状', '信号機',
        '一時停止規制 標識', '一時停止規制 表示', '車道幅員', '道路線形',
        '衝突地点', 'ゾーン規制', '中央分離帯施設等', '歩車道区分',
        '事故類型', '年齢', '当事者種別', '用途別', '車両形状',
        'オートマチック車', 'サポカー', '速度規制（指定のみ）',
        '曜日', '祝日', '発生月', '発生時', '発生年', 'Area_Cluster_ID'
    ]
    
    # 実際にデータフレームに存在する列のみを対象とする
    explicit_cat_cols = [c for c in categorical_candidates if c in X.columns]
    
    # 文字列型の列もカテゴリとして扱う
    object_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # 統合したカテゴリカル変数のリスト
    final_cat_cols = list(set(explicit_cat_cols + object_cols))
    
    print(f"\n🏷️ カテゴリカル変数の変換: {len(final_cat_cols)} カラム")
    
    for col in final_cat_cols:
        # category型に変換 (NaNはNaNとして維持される)
        X[col] = X[col].astype('category')

    # LabelEncoderは不要になったため削除
    # LightGBMは category 型を直接扱える
        
    print(f"✓ 前処理完了 - 特徴量数: {X.shape[1]}")
    
    # クラスの不均衡比を計算し、scale_pos_weightに設定
    # scale_pos_weight = (negative samples) / (positive samples)
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_pos_weight = neg_count / pos_count
    
    print(f"\n⚖️ クラス不均衡比の計算:")
    print(f"  Negative (0): {neg_count:,}")
    print(f"  Positive (1): {pos_count:,}")
    print(f"  Calculated scale_pos_weight: {scale_pos_weight:.2f}")
    
    # LightGBMのパラメータ
    lgbm_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42,
        'n_jobs': -1,
        'scale_pos_weight': scale_pos_weight  # ★ここが変更点
    }
    
    # 交差検証 (5-fold)
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    print(f"\n🔄 {k_folds}-fold 交差検証を開始 (Weighted)...")
    
    fold_metrics = []
    
    # 全体の予測結果を格納する配列
    y_true_all = []
    y_prob_all = []
    feature_importances = pd.DataFrame()
    
    for i, (train_index, val_index) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {i+1}/{k_folds} ---")
        
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # モデル構築（Pipeline不要、直接LGBMClassifier）
        model = lgb.LGBMClassifier(**lgbm_params)
        
        # 学習
        model.fit(X_train, y_train)
        
        # 予測（確率）
        y_prob = model.predict_proba(X_val)[:, 1]
        
        # 全体の結果に蓄積
        y_true_all.extend(y_val)
        y_prob_all.extend(y_prob)
        
        # デフォルト閾値(0.5)での評価
        y_pred_default = (y_prob >= 0.5).astype(int)
        
        acc = accuracy_score(y_val, y_pred_default)
        prec = precision_score(y_val, y_pred_default, average='binary', zero_division=0)
        rec = recall_score(y_val, y_pred_default, average='binary')
        f1 = f1_score(y_val, y_pred_default, average='binary')
        
        print(f"  [Threshold 0.5] Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        fold_metrics.append({
            'Fold': i+1,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1
        })

        # 特徴量重要度の取得
        fi = pd.DataFrame()
        fi['feature'] = X.columns
        fi['importance'] = model.feature_importances_
        fi['fold'] = i + 1
        feature_importances = pd.concat([feature_importances, fi], axis=0)

    # 全データでの評価
    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all)
    
    # AUCの計算
    auc_score = roc_auc_score(y_true_all, y_prob_all)
    print(f"\n📈 AUC Score: {auc_score:.4f}")
    
    with open('results/analysis/weighted_auc_score.txt', 'w') as f:
        f.write(str(auc_score))

    # PR曲線と最適閾値の探索
    precisions, recalls, thresholds = precision_recall_curve(y_true_all, y_prob_all)
    
    # F1スコアが最大になる閾値
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print("\n" + "=" * 80)
    print("🎯 最適閾値の探索結果")
    print("=" * 80)
    print(f"Best Threshold (Max F1): {best_threshold:.4f}")
    print(f"Max F1 Score: {best_f1:.4f}")
    print(f"Precision at Best: {precisions[best_idx]:.4f}")
    print(f"Recall at Best: {recalls[best_idx]:.4f}")
    
    # Recall重視の閾値設定（例: Recall >= 0.8 を満たす中で最大のPrecision）
    # 重み付けモデルなので、デフォルトでもRecallは高くなるはずだが、さらに探索する
    target_recall = 0.8
    valid_indices = np.where(recalls >= target_recall)[0]
    if len(valid_indices) > 0:
        best_prec_idx = valid_indices[np.argmax(precisions[valid_indices])]
        recall_threshold = thresholds[best_prec_idx] if best_prec_idx < len(thresholds) else thresholds[-1]
        
        print(f"\n[Recall重視設定 (Target >= {target_recall})]")
        print(f"Threshold: {recall_threshold:.4f}")
        print(f"Precision: {precisions[best_prec_idx]:.4f}")
        print(f"Recall: {recalls[best_prec_idx]:.4f}")
    
    # PR曲線のプロット
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, marker='.', label='LightGBM + Weighted')
    plt.xlabel('Recall (再現率)')
    plt.ylabel('Precision (適合率)')
    plt.title('Precision-Recall Curve (Weighted Model)')
    plt.legend()
    plt.grid(True)
    
    pr_path = 'results/visualizations/pr_curve_weighted.png'
    plt.savefig(pr_path)
    print(f"\n✓ PR曲線を保存: {pr_path}")
    
    # 混同行列（デフォルト閾値 0.5 での評価が重要）
    # 重み付けを行った場合、閾値0.5でもRecallが高くなることが期待される
    y_pred_05 = (y_prob_all >= 0.5).astype(int)
    cm = confusion_matrix(y_true_all, y_pred_05)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['非死亡', '死亡'], yticklabels=['非死亡', '死亡'])
    plt.title(f'Confusion Matrix (Weighted, Threshold=0.5)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    cm_path = 'results/visualizations/confusion_matrix_weighted.png'
    plt.savefig(cm_path)
    print(f"✓ 混同行列を保存: {cm_path}")
    
    # 評価メトリクスの保存
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv('results/analysis/weighted_model_metrics.csv', index=False)
    print("✓ 評価メトリクスを保存: results/analysis/weighted_model_metrics.csv")
    
    # 特徴量重要度の集計と保存
    feat_imp_mean = feature_importances.groupby('feature')['importance'].mean().sort_values(ascending=False)
    feat_imp_mean.to_csv('results/analysis/feature_importance.csv')
    print("✓ 特徴量重要度を保存: results/analysis/feature_importance.csv")

    # 特徴量重要度の可視化（Top 20）
    plt.figure(figsize=(10, 8))
    sns.barplot(x=feat_imp_mean.head(20).values, y=feat_imp_mean.head(20).index, palette='viridis')
    plt.title('LightGBM Feature Importance (Top 20)')
    plt.xlabel('Importance (Split)')
    plt.tight_layout()
    plt.savefig('results/visualizations/feature_importance.png')
    print("✓ 特徴量重要度グラフを保存: results/visualizations/feature_importance.png")
    
    print("\n✅ 実験完了")

if __name__ == "__main__":
    main()
