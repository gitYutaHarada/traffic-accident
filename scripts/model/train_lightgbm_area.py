
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 日本語フォント設定（環境に合わせて調整が必要な場合があります）
plt.rcParams['font.family'] = 'Meiryo'

def dms_to_degree(dms_val):
    """
    度分秒形式（例: 1394234 -> 139度42分34秒）を度（Decimal Degrees）に変換する
    """
    if pd.isna(dms_val):
        return np.nan
    
    dms_str = str(int(dms_val)).zfill(9) # 最低9桁確保（度3桁+分2桁+秒2桁+小数点以下）
    # ただしデータによっては整数部のみの場合もあるので、桁数に応じて処理
    # データを確認したところ 1394234000 のような形式ではなく 1365993000 (9-10桁) のような値
    # Analyze結果より: mean 1.365993e+09 -> 1,365,993,000 ?
    # いや、Variable Types Analysisの出力では:
    # 25% 1.351156e+09 ... max 1.454811e+09
    # これはミリ秒単位まで含んでいる可能性がある、または単純に数値として大きい。
    # 日本の経度は120-150度。1,365,993,000 だとすると...
    # コードブック（抽出版）には「世界測地系（度分秒）とする」とある。
    # 通常、dms=1394234 は 139度42分34秒。
    # CSVの値の例: 1413100440 (10桁) -> 141度31分00.440秒 か 141度31分00440 ?
    # 一般的な交通事故データOpenDataでは「DDDMMSS.sss」を整数化したものが多い。
    # 例: 1394234567 -> 139度42分34.567秒
    # ここでは10桁前後の値を想定して変換する。
    
    val_str = str(int(dms_val))
    length = len(val_str)
    
    if length < 7: # 異常値または0
        return np.nan

    # 下からミリ秒、秒、分を切り出す
    # 想定: DDDMMSSsss (度3桁, 分2桁, 秒2桁, ミリ秒3桁)
    
    # 秒の小数点以下の扱いが不明確だが、経度緯度のクラスタリングには
    # おおまかな度への変換で十分な場合が多い。
    # とりあえず下3桁を取り除いてみる
    
    try:
        # 秒の小数部があると仮定して下3桁を除く
        seconds_float_part = val_str[-3:]
        dms_int_part = val_str[:-3]
        
        # 秒
        if len(dms_int_part) >= 2:
            seconds = float(dms_int_part[-2:] + "." + seconds_float_part)
            rem = dms_int_part[:-2]
        else:
            return np.nan
            
        # 分
        if len(rem) >= 2:
            minutes = float(rem[-2:])
            degrees = float(rem[:-2])
        else:
             return np.nan

        return degrees + minutes / 60 + seconds / 3600

    except Exception:
        return np.nan


def preprocess_data(df, n_clusters=500):
    print("前処理を開始します...")
    
    # 1. 緯度経度の変換
    print("緯度経度を変換中...")
    # NOTE: データの実測値に基づいて変換ロジックを調整するのがベストだが、
    # ここでは簡易的に dms_to_degree を適用する。
    # 実際にはベクトル演算で高速化したいが、DMSパースが複雑なためapplyを使用（遅い可能性あり）
    # 高速化のため、単純な数値計算で近似する
    # DDDMMSSsss -> DDD + MM/60 + SS.sss/3600
    
    # 経度
    lon_val = df['地点　経度（東経）'].fillna(0).astype(np.int64)
    # 緯度
    lat_val = df['地点　緯度（北緯）'].fillna(0).astype(np.int64)
    
    def convert_vectorized(v):
        # 0の場合はNaN
        v = np.where(v == 0, np.nan, v)
        # ミリ秒 (下3桁)
        ms = v % 1000
        v = v // 1000
        # 秒 (次の2桁)
        ss = v % 100
        v = v // 100
        # 分 (次の2桁)
        mm = v % 100
        # 度 (残り)
        dd = v // 100
        
        return dd + mm/60 + (ss + ms/1000)/3600

    df['lon_deg'] = convert_vectorized(lon_val)
    df['lat_deg'] = convert_vectorized(lat_val)
    
    # 変換失敗（NaN）の行を削除するか、埋めるか
    # 今回は0除外しているのでNaNになるのは元が0の場所＝欠損
    # クラスタリングできないので、欠損のままにするか、-1等のエリアIDを割り当てる
    # エリアID作成用にdropnaした一時データを使う
    
    print("エリアクラスタリングを実行中...")
    # 座標があるデータのみでクラスタリング学習
    coords = df[['lat_deg', 'lon_deg']].dropna()
    
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=4096, n_init=10)
    kmeans.fit(coords)
    
    # 全データに対して予測（欠損値は -1 とする）
    df['area_id'] = -1
    valid_idx = coords.index
    df.loc[valid_idx, 'area_id'] = kmeans.predict(coords)
    
    # area_idをカテゴリカルに
    df['area_id'] = df['area_id'].astype('category')
    
    # 2. 日時の分解
    print("日時データを分解中...")
    df['発生日時'] = pd.to_datetime(df['発生日時'])
    df['month'] = df['発生日時'].dt.month.astype('category')
    df['day'] = df['発生日時'].dt.day.astype('category')
    df['hour'] = df['発生日時'].dt.hour.astype('category')
    # 曜日は既存カラム '曜日(発生年月日)' を使用
    
    # 不要なカラムの削除
    drop_cols = ['発生日時', '地点　経度（東経）', '地点　緯度（北緯）', 'lon_deg', 'lat_deg']
    df = df.drop(columns=drop_cols)
    
    # 3. カテゴリカル化
    print("カテゴリカル変数を設定中...")
    # 目的変数以外の全てのカラムをカテゴリカルにする（緯度経度は既に削除済み、area_id等は設定済み）
    target_col = '死者数'
    
    for col in df.columns:
        if col != target_col and df[col].dtype.name != 'category':
             df[col] = df[col].astype('category')
             
    print(f"前処理完了: {df.shape}")
    return df

def train_and_evaluate(df):
    target_col = '死者数'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # クラス不均衡の計算
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"クラス不均衡: 負例={neg_count}, 正例={pos_count}, scale_pos_weight={scale_pos_weight:.2f}")

    # StratifiedKFold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics = []
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = X.columns
    feature_importances['importance'] = 0
    
    oof_preds = np.zeros(len(df))
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'scale_pos_weight': scale_pos_weight,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }

    print("学習を開始します...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        print(f"Fold {fold+1}/5")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # LightGBM Dataset
        # categorical_feature='auto' でcategory型のカラムを自動認識
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # コールバックの設定
        callbacks = [
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            num_boost_round=1000,
            callbacks=callbacks
        )
        
        # 予測
        y_pred_prob = model.predict(X_val, num_iteration=model.best_iteration)
        oof_preds[val_idx] = y_pred_prob
        y_pred = (y_pred_prob >= 0.5).astype(int)
        
        # 評価
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_pred_prob)
        
        metrics.append([fold+1, acc, prec, rec, f1, auc])
        
        # Feature Importance (Gain)
        fold_importance = model.feature_importance(importance_type='gain')
        feature_importances['importance'] += fold_importance / 5
        
        print(f"  AUC: {auc:.4f}, Recall: {rec:.4f}, Precision: {prec:.4f}")

    # 集計
    metrics_df = pd.DataFrame(metrics, columns=['Fold', 'Accuracy', 'Precision', 'Recall', 'F1', 'AUC'])
    print("\n交差検証結果:")
    print(metrics_df)
    print("\n平均スコア:")
    print(metrics_df.mean())
    
    # Feature Importanceソート
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)
    
    return metrics_df, feature_importances, oof_preds

def save_results(metrics_df, feature_importances, df, oof_preds):
    output_dir = 'results/model_lightgbm_area'
    os.makedirs(output_dir, exist_ok=True)
    
    # 指標保存
    metrics_df.to_csv(f'{output_dir}/metrics.csv', index=False)
    metrics_df.mean().to_csv(f'{output_dir}/metrics_mean.csv')
    
    # 重要度保存
    feature_importances.to_csv(f'{output_dir}/feature_importance.csv', index=False)
    
    # 重要度プロット
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importances.head(20))
    plt.title('LightGBM Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png')
    
    # 混乱行列
    y_true = df['死者数']
    y_pred = (oof_preds >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (OOF)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    
    print(f"\n結果を {output_dir} に保存しました。")

def main():
    input_file = r"c:\Users\socce\software-lab\traffic-accident\data\processed\honhyo_clean_predictable_only.csv"
    
    print(f"データ読み込み: {input_file}")
    df = pd.read_csv(input_file)
    
    # 前処理
    df_processed = preprocess_data(df)
    
    # 学習・評価
    metrics_df, feature_importances, oof_preds = train_and_evaluate(df_processed)
    
    # 保存
    save_results(metrics_df, feature_importances, df_processed, oof_preds)

if __name__ == "__main__":
    main()
