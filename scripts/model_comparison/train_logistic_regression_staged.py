"""
ロジスティック回帰による段階的モデル比較実験

段階的サンプリングアプローチにより、大規模データセットでも実行可能に改良。
コマンドライン引数でサンプリング率を指定可能。

使用例:
    # 1%サンプル
    python train_logistic_regression_staged.py --sample-rate 0.01
    
    # 10%サンプル
    python train_logistic_regression_staged.py --sample-rate 0.1
    
    # 全データ
    python train_logistic_regression_staged.py --sample-rate 1.0
"""

import pandas as pd
import numpy as np
import os
import warnings
import argparse
import time
from datetime import timedelta
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
    roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

warnings.filterwarnings('ignore')

# 日本語フォントの設定
mpl.rcParams['font.family'] = 'MS Gothic'

def format_time(seconds):
    """秒数を人間が読みやすい形式に変換"""
    return str(timedelta(seconds=int(seconds)))

def main():
    """
    段階的ロジスティック回帰による死亡事故予測モデル
    """
    # コマンドライン引数のパース
    parser = argparse.ArgumentParser(description='段階的ロジスティック回帰実験')
    parser.add_argument('--sample-rate', type=float, default=0.01,
                        help='サンプリング率 (0.01=1%%, 0.1=10%%, 1.0=全データ)')
    args = parser.parse_args()
    
    sample_rate = args.sample_rate
    
    # 実行開始時刻
    script_start_time = time.time()
    
    print("=" * 80)
    print(f"モデル比較実験: ロジスティック回帰 (サンプリング率: {sample_rate*100:.1f}%)")
    print("=" * 80)
    
    # データ読み込み(LightGBMと同じデータセット)
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
    
    # サンプリング(層化サンプリングでクラス比率を維持)
    if sample_rate < 1.0:
        print(f"\n🎲 {sample_rate*100:.1f}% サンプリング中(層化サンプリング)...")
        # 各クラスから同じ割合でサンプリング
        df_0 = df[df[target_col] == 0].sample(frac=sample_rate, random_state=42)
        df_1 = df[df[target_col] == 1].sample(frac=sample_rate, random_state=42)
        df = pd.concat([df_0, df_1], ignore_index=True)
        # シャッフル
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"✓ サンプリング完了: {len(df):,} 件")
    
    # 除外する列(LightGBMと同じ事後情報を除外)
    drop_cols = [
        '資料区分', '本票番号',
        '人身損傷程度(当事者A)', '人身損傷程度(当事者B)',
        '車両の損壊程度(当事者A)', '車両の損壊程度(当事者B)',
        '負傷者数',
        '車両の衝突部位(当事者A)', '車両の衝突部位(当事者B)',
        'エアバッグの装備(当事者A)', 'エアバッグの装備(当事者B)',
        'サイドエアバッグの装備(当事者A)', 'サイドエアバッグの装備(当事者B)',
        '事故内容'
    ]
    
    # カラム名の正規化(全角括弧を半角に統一)
    df.columns = df.columns.str.replace('(', '(').str.replace(')', ')')
    
    print("\n🔧 データ前処理中(事後情報の除外)...")
    df_clean = df.drop(columns=drop_cols, errors='ignore')
    
    # 特徴量と目的変数
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    print(f"✓ 前処理完了 - 特徴量数: {X.shape[1]}")
    
    # カテゴリカル変数と数値変数の分類
    # カウントエンコーディング列は数値として扱う
    count_encoding_cols = [col for col in X.columns if col.endswith('_count')]
    
    # 数値型の列
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # カテゴリカル型の列(文字列型 + category型)
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # カテゴリカル変数として明示的に扱うべき列(数値コードだがカテゴリ)
    explicit_cat_cols = [
        '都道府県コード', '路線コード', '地点コード', '市区町村コード',
        '昼夜', '天候', '地形', '路面状態', '道路形状', '信号機',
        '一時停止規制 標識', '一時停止規制 表示', '車道幅員', '道路線形',
        '衝突地点', 'ゾーン規制', '中央分離帯施設等', '歩車道区分',
        '事故類型', '年齢', '当事者種別', '用途別', '車両形状',
        'オートマチック車', 'サポカー', '速度規制(指定のみ)',
        '曜日', '祝日', '発生月', '発生時', '発生年', 'Area_Cluster_ID'
    ]
    
    # 実際に存在する列のみを対象
    explicit_cat_cols = [c for c in explicit_cat_cols if c in X.columns and c not in count_encoding_cols]
    
    # 統合したカテゴリカル変数リスト
    final_cat_cols = list(set(categorical_cols + explicit_cat_cols))
    
    # 数値変数リスト(カテゴリカルでないもの)
    final_numeric_cols = [c for c in numeric_cols if c not in final_cat_cols]
    
    print(f"\n🏷️ カテゴリカル変数: {len(final_cat_cols)} カラム")
    print(f"🔢 数値変数: {len(final_numeric_cols)} カラム")
    
    # カテゴリカル変数を文字列型に統一(OneHotEncoderが型の混在を許さないため)
    print("\n🔄 カテゴリカル変数を文字列型に変換中...")
    for col in final_cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)
    
    # カーディナリティが高すぎる変数の処理(上位N個以外を'その他'にまとめる)
    high_cardinality_threshold = 50  # 100 → 50 に削減
    for col in final_cat_cols:
        if col in X.columns:
            nunique = X[col].nunique()
            if nunique > high_cardinality_threshold:
                print(f"  ⚠️ '{col}' のカーディナリティが高い({nunique})ため、上位{high_cardinality_threshold}個以外を'その他'にまとめます")
                top_categories = X[col].value_counts().head(high_cardinality_threshold).index
                X[col] = X[col].apply(lambda x: x if x in top_categories else 'その他')
    
    # 前処理パイプラインの構築
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=30))  # 50 → 30
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, final_numeric_cols),
            ('cat', categorical_transformer, final_cat_cols)
        ],
        remainder='drop'  # その他の列は削除
    )
    
    # ロジスティック回帰モデル
    # class_weight='balanced'でクラス不均衡に対応(LightGBMのscale_pos_weightに相当)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(
            penalty='l2',
            C=1.0,
            solver='saga',  # 大規模データに適したソルバー
            max_iter=500,  # 1000 → 500 に削減
            class_weight='balanced',
            random_state=42,
            verbose=1,  # 進捗表示を有効化
            n_jobs=-1
        ))
    ])
    
    # クラスの不均衡比を表示
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    print(f"\n⚖️ クラス不均衡比:")
    print(f"  Negative (0): {neg_count:,}")
    print(f"  Positive (1): {pos_count:,}")
    print(f"  比率: {neg_count/pos_count:.2f}:1")
    
    # 5-fold交差検証(LightGBMと同じ)
    k_folds = 5
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    print(f"\n🔄 {k_folds}-fold 交差検証を開始...")
    print(f"💡 進捗表示: ソルバーの収束状況が表示されます\n")
    
    fold_metrics = []
    y_true_all = []
    y_prob_all = []
    
    cv_start_time = time.time()
    
    for i, (train_index, val_index) in enumerate(skf.split(X, y)):
        fold_start_time = time.time()
        elapsed = fold_start_time - cv_start_time
        
        print("=" * 80)
        print(f"--- Fold {i+1}/{k_folds} ---")
        print(f"総経過時間: {format_time(elapsed)}")
        print("=" * 80)
        
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # 学習
        print(f"\n  📚 学習中 (訓練データ: {len(X_train):,} 件)...")
        fit_start = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - fit_start
        print(f"  ✓ 学習完了 (所要時間: {format_time(fit_time)})")
        
        # 予測(確率)
        print(f"  🔮 予測中 (検証データ: {len(X_val):,} 件)...")
        pred_start = time.time()
        y_prob = model.predict_proba(X_val)[:, 1]
        pred_time = time.time() - pred_start
        print(f"  ✓ 予測完了 (所要時間: {format_time(pred_time)})")
        
        # 全体の結果に蓄積
        y_true_all.extend(y_val)
        y_prob_all.extend(y_prob)
        
        # デフォルト閾値(0.5)での評価
        y_pred_default = (y_prob >= 0.5).astype(int)
        
        acc = accuracy_score(y_val, y_pred_default)
        prec = precision_score(y_val, y_pred_default, average='binary', zero_division=0)
        rec = recall_score(y_val, y_pred_default, average='binary')
        f1 = f1_score(y_val, y_pred_default, average='binary')
        
        print(f"\n  📊 [Threshold 0.5] Acc: {acc:.4f}, Prec: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        fold_total_time = time.time() - fold_start_time
        print(f"  ⏱️  Fold {i+1} 合計時間: {format_time(fold_total_time)}")
        
        fold_metrics.append({
            'Fold': i+1,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'Fit Time (sec)': fit_time,
            'Predict Time (sec)': pred_time,
            'Total Time (sec)': fold_total_time
        })
    
    cv_total_time = time.time() - cv_start_time
    print("\n" + "=" * 80)
    print(f"✅ 全Fold完了 (合計時間: {format_time(cv_total_time)})")
    print("=" * 80)
    
    # 全データでの評価
    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all)
    
    # AUCの計算
    auc_score = roc_auc_score(y_true_all, y_prob_all)
    print(f"\n📈 AUC Score: {auc_score:.4f}")
    
    # 出力ディレクトリの作成
    if sample_rate < 1.0:
        output_dir = f'results/model_comparison/logistic_regression_{int(sample_rate*100)}pct'
    else:
        output_dir = 'results/model_comparison/logistic_regression'
    os.makedirs(output_dir, exist_ok=True)
    
    # AUCの保存
    with open(f'{output_dir}/auc_score.txt', 'w') as f:
        f.write(f"{auc_score:.4f}")
    
    # PR曲線と最適閾値の探索
    precisions, recalls, thresholds = precision_recall_curve(y_true_all, y_prob_all)
    
    # F1スコアが最大になる閾値
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)
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
    
    # Recall重視の閾値設定
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
    plt.plot(recalls, precisions, marker='.', label=f'Logistic Regression ({sample_rate*100:.1f}% sample)')
    plt.xlabel('Recall (再現率)')
    plt.ylabel('Precision (適合率)')
    plt.title(f'Precision-Recall Curve (Logistic Regression, {sample_rate*100:.1f}% sample)')
    plt.legend()
    plt.grid(True)
    
    pr_path = f'{output_dir}/pr_curve.png'
    plt.savefig(pr_path)
    print(f"\n✓ PR曲線を保存: {pr_path}")
    plt.close()
    
    # 混同行列(デフォルト閾値 0.5)
    y_pred_05 = (y_prob_all >= 0.5).astype(int)
    cm = confusion_matrix(y_true_all, y_pred_05)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['非死亡', '死亡'], yticklabels=['非死亡', '死亡'])
    plt.title(f'Confusion Matrix (Logistic Regression, {sample_rate*100:.1f}% sample, Threshold=0.5)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    cm_path = f'{output_dir}/confusion_matrix.png'
    plt.savefig(cm_path)
    print(f"✓ 混同行列を保存: {cm_path}")
    plt.close()
    
    # 評価メトリクスの保存
    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(f'{output_dir}/metrics.csv', index=False)
    print(f"✓ 評価メトリクスを保存: {output_dir}/metrics.csv")
    
    # 平均値の計算
    avg_metrics = metrics_df.mean()
    print(f"\n📊 {k_folds}-fold CV 平均スコア:")
    print(f"  Accuracy:  {avg_metrics['Accuracy']:.4f}")
    print(f"  Precision: {avg_metrics['Precision']:.4f}")
    print(f"  Recall:    {avg_metrics['Recall']:.4f}")
    print(f"  F1 Score:  {avg_metrics['F1 Score']:.4f}")
    
    print(f"\n⏱️  平均実行時間 (1 Fold):")
    print(f"  学習時間:  {format_time(avg_metrics['Fit Time (sec)'])}")
    print(f"  予測時間:  {format_time(avg_metrics['Predict Time (sec)'])}")
    print(f"  合計時間:  {format_time(avg_metrics['Total Time (sec)'])}")
    
    # 全データでの実行時間推定
    script_total_time = time.time() - script_start_time
    if sample_rate < 1.0:
        estimated_full_time = script_total_time / sample_rate
        print(f"\n🔮 全データ(100%)での推定実行時間:")
        print(f"  現在のサンプル率: {sample_rate*100:.1f}%")
        print(f"  実測時間: {format_time(script_total_time)}")
        print(f"  推定時間(線形スケーリング): {format_time(estimated_full_time)}")
        
        if estimated_full_time > 3600:  # 1時間以上
            print(f"  ⚠️  推定時間が長いため、さらなる最適化が必要な可能性があります")
    
    # サマリーレポートの作成
    summary_lines = [
        f"# ロジスティック回帰 実験結果 ({sample_rate*100:.1f}% サンプル)",
        "",
        "**実験日時:** " + pd.Timestamp.now().strftime('%Y年%m月%d日 %H:%M:%S'),
        "**目的:** LightGBMとの比較のためのベースラインモデル",
        f"**サンプリング率:** {sample_rate*100:.1f}%",
        "",
        "---",
        "",
        "## 📊 実験概要",
        "",
        "### データセット",
        f"- ファイル: `{file_path}`",
        f"- 元データ数: 1,895,275 件",
        f"- 使用データ数: {len(df):,} 件 ({sample_rate*100:.1f}%)",
        f"- Positive(死亡事故): {pos_count:,} 件",
        f"- Negative(非死亡): {neg_count:,} 件",
        f"- 不均衡比: {neg_count/pos_count:.2f}:1",
        "",
        "### 特徴量",
        f"- カテゴリカル変数: {len(final_cat_cols)} カラム",
        f"- 数値変数: {len(final_numeric_cols)} カラム",
        f"- 総特徴量数: {X.shape[1]} (One-Hot Encoding後は増加)",
        f"- 高カーディナリティ処理: 上位{high_cardinality_threshold}カテゴリ以外を'その他'に統合",
        "",
        "### モデル設定",
        "```python",
        "LogisticRegression(",
        "    penalty='l2',",
        "    C=1.0,",
        "    solver='saga',",
        "    max_iter=500,",
        "    class_weight='balanced',  # クラス不均衡対策",
        "    verbose=1,",
        "    random_state=42",
        ")",
        "```",
        "",
        "---",
        "",
        "## 📈 評価結果",
        "",
        "### 5-fold CV 平均スコア (Threshold 0.5)",
        "| 指標 | スコア |",
        "|------|--------|",
        f"| **Accuracy** | {avg_metrics['Accuracy']:.4f} |",
        f"| **Precision** | {avg_metrics['Precision']:.4f} |",
        f"| **Recall** | {avg_metrics['Recall']:.4f} |",
        f"| **F1 Score** | {avg_metrics['F1 Score']:.4f} |",
        f"| **AUC** | **{auc_score:.4f}** |",
        "",
        "### 最適閾値の探索結果",
        "| 設定 | 閾値 | Recall | Precision | F1 Score |",
        "|------|------|--------|-----------|----------|",
        f"| **Max F1** | {best_threshold:.4f} | {recalls[best_idx]:.4f} | {precisions[best_idx]:.4f} | {best_f1:.4f} |",
    ]
    
    if len(valid_indices) > 0:
        summary_lines.append(f"| **Recall≥0.8** | {recall_threshold:.4f} | {recalls[best_prec_idx]:.4f} | {precisions[best_prec_idx]:.4f} | - |")
    
    summary_lines.extend([
        "",
        "---",
        "",
        "## ⏱️ 実行時間",
        "",
        f"| 項目 | 時間 |",
        f"|------|------|",
        f"| **合計実行時間** | {format_time(script_total_time)} |",
        f"| **交差検証時間** | {format_time(cv_total_time)} |",
        f"| **平均学習時間(1 Fold)** | {format_time(avg_metrics['Fit Time (sec)'])} |",
        f"| **平均予測時間(1 Fold)** | {format_time(avg_metrics['Predict Time (sec)'])} |",
    ])
    
    if sample_rate < 1.0:
        estimated_full_time = script_total_time / sample_rate
        summary_lines.extend([
            "",
            "### 全データでの推定時間",
            f"- 現在のサンプル率: **{sample_rate*100:.1f}%**",
            f"- 実測時間: **{format_time(script_total_time)}**",
            f"- 推定時間(線形スケーリング): **{format_time(estimated_full_time)}**",
        ])
    
    summary_lines.extend([
        "",
        "---",
        "",
        "## 💡 考察",
        "",
        "### 前処理の違い",
        "- **カテゴリカル変数**: One-Hot Encodingを使用(LightGBMはcategory型を直接扱える)",
        "- **数値変数**: StandardScalerで標準化(LightGBMは不要)",
        "- **欠損値**: SimpleImputerで補完(LightGBMは欠損値をそのまま扱える)",
        "",
        "### モデルの特徴",
        "- **線形モデル**: 特徴量間の複雑な相互作用を捉えにくい",
        "- **解釈性**: 係数(Coefficients)から各特徴量の影響を直接読み取れる",
        "- **計算コスト**: LightGBMより学習時間が短い(ただしOne-Hot Encodingで特徴量数が増加)",
        "",
        "---",
        "",
        "## 📁 出力ファイル",
        f"- [PR曲線]({pr_path})",
        f"- [混同行列]({cm_path})",
        f"- [評価指標CSV]({output_dir}/metrics.csv)",
        f"- [AUCスコア]({output_dir}/auc_score.txt)",
    ])
    
    summary_path = f'{output_dir}/summary_report.md'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(summary_lines))
    print(f"✓ サマリーレポートを保存: {summary_path}")
    
    print("\n" + "=" * 80)
    print("✅ 実験完了")
    print("=" * 80)
    print(f"📂 結果は以下に保存されました: {output_dir}")
    print(f"⏱️  総実行時間: {format_time(script_total_time)}")
    
    if sample_rate < 1.0 and estimated_full_time < 28800:  # 8時間以内なら
        next_sample = min(sample_rate * 10, 1.0)
        print(f"\n💡 次のステップ: {next_sample*100:.0f}%サンプルでの実行を検討してください")
        print(f"   コマンド: python {__file__} --sample-rate {next_sample}")

if __name__ == "__main__":
    main()
