import pandas as pd
import numpy as np
import shap
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# 日本語フォントの設定 (Windows向け)
mpl.rcParams['font.family'] = 'MS Gothic'
# マイナス記号の文字化け防止
mpl.rcParams['axes.unicode_minus'] = False

def main():
    """
    LightGBMモデルのSHAP分析を行い、予測の根拠を可視化する
    """
    
    print("=" * 80)
    print("モデル解釈: SHAP分析による可視化")
    print("=" * 80)
    
    # データ読み込み
    file_path = 'data/raw/honhyo_all_shishasuu_binary.csv'
    print(f"\n📂 データ読み込み中: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"✓ データ読み込み完了: {len(df):,} 件")
    except Exception as e:
        print(f"❌ エラー: {e}")
        return
    
    # 目的変数
    target_col = '死者数'
    
    # 除外する列（事後情報・データリーク原因）
    drop_cols = [
        '資料区分', '本票番号',
        '人身損傷程度（当事者A）', '人身損傷程度（当事者B）',
        '車両の損壊程度（当事者A）', '車両の損壊程度（当事者B）',
        '負傷者数',
        '車両の衝突部位（当事者A）', '車両の衝突部位（当事者B）',
        'エアバッグの装備（当事者A）', 'エアバッグの装備（当事者B）',
        'サイドエアバッグの装備（当事者A）', 'サイドエアバッグの装備（当事者B）',
        '事故内容'
    ]
    
    print("\n🔧 データ前処理中...")
    df_clean = df.drop(columns=drop_cols, errors='ignore')
    
    # 特徴量と目的変数
    X = df_clean.drop(columns=[target_col])
    y = df_clean[target_col]
    
    # 欠損値処理
    num_cols = X.select_dtypes(include=[np.number]).columns
    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    
    cat_cols = X.select_dtypes(include=['object']).columns
    for col in cat_cols:
        X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
    
    # エンコーディング
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))
        
    print(f"✓ 前処理完了 - 特徴量数: {X.shape[1]}")
    print("使用されている特徴量一覧:")
    print(list(X.columns))
    
    # 除外リストに含まれる列が残っていないかチェック
    remaining_drop_cols = [col for col in drop_cols if col in X.columns]
    if remaining_drop_cols:
        print(f"⚠️ 警告: 除外すべき列が残っています: {remaining_drop_cols}")
    else:
        print("✓ 除外リストの列はすべて削除されています")
    
    # データを分割（学習用とSHAP計算用）
    # SHAP計算は重いため、テストデータの一部を使用する
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 訓練データに対してSMOTE適用
    print("\n🔄 SMOTEによるオーバーサンプリングを実行中...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"  訓練データ数: {len(X_train)} -> {len(X_train_res)}")
    
    # LightGBMモデルの学習
    print("\n🌲 LightGBMモデルを学習中...")
    lgbm_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(X_train_res, y_train_res)
    print("✓ モデル学習完了")
    
    # SHAP値の計算
    print("\n🔍 SHAP値を計算中...")
    # 計算時間を考慮し、テストデータからランダムにサンプリング（例: 2000件）
    # 死亡事故（少数派）を多めに含めると特徴が見えやすいが、
    # 全体の傾向を見るためにランダムサンプリングとする
    sample_size = 2000
    if len(X_test) > sample_size:
        X_shap = X_test.sample(n=sample_size, random_state=42)
    else:
        X_shap = X_test
        
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)
    
    # LightGBMのbinary classificationの場合、shap_valuesはリストで返る場合とarrayで返る場合がある
    # shap ver 0.40以降の挙動を確認
    if isinstance(shap_values, list):
        # クラス1（死亡事故）に対するSHAP値を取得
        shap_values_target = shap_values[1]
    else:
        # arrayの場合 (n_samples, n_features) または (n_samples, n_features, n_classes)
        if len(shap_values.shape) == 3:
             shap_values_target = shap_values[:, :, 1]
        else:
             shap_values_target = shap_values

    print("✓ SHAP値計算完了")
    
    # 保存ディレクトリ
    output_dir = 'results/visualizations/shap'
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Summary Plot (Bar) - 特徴量重要度
    print("\n📊 Summary Plot (Bar) を作成中...")
    plt.figure()
    shap.summary_plot(shap_values_target, X_shap, plot_type="bar", show=False)
    plt.title('SHAP 特徴量重要度', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_summary_bar.png', bbox_inches='tight')
    plt.close()
    
    # 2. Summary Plot (Dot) - 影響の方向性
    print("📊 Summary Plot (Dot) を作成中...")
    plt.figure()
    shap.summary_plot(shap_values_target, X_shap, show=False)
    plt.title('SHAP Summary Plot', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/shap_summary_dot.png', bbox_inches='tight')
    plt.close()
    
    # 3. Dependence Plot - 上位特徴量の詳細分析
    # 重要度上位の特徴量を取得
    # shap_valuesの絶対値の平均をとってランク付け
    mean_abs_shap = np.abs(shap_values_target).mean(axis=0)
    top_features_indices = np.argsort(mean_abs_shap)[::-1][:3] # Top 3
    top_features = X.columns[top_features_indices]
    
    print(f"📊 Dependence Plot を作成中 (Top 3特徴量: {list(top_features)})...")
    
    for feature in top_features:
        plt.figure()
        shap.dependence_plot(feature, shap_values_target, X_shap, show=False)
        plt.title(f'SHAP Dependence Plot: {feature}', fontsize=14)
        plt.tight_layout()
        # ファイル名に使えない文字を置換
        safe_feature_name = feature.replace('/', '_').replace(':', '_').replace(' ', '_')
        plt.savefig(f'{output_dir}/shap_dependence_{safe_feature_name}.png', bbox_inches='tight')
        plt.close()
        
    print("\n✅ 分析完了")
    print(f"結果は {output_dir} に保存されました。")

if __name__ == "__main__":
    main()
