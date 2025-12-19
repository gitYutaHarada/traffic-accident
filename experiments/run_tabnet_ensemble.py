"""
LightGBM + TabNet Ensemble Experiment

自転車・歩行者など「非線形な挙動」を示す主体の誤検知削減のため、
LightGBM と TabNet のアンサンブルを実装する。

Output:
- ensemble_report.md
- ensemble_predictions.csv
- model_comparison.png
"""

import pandas as pd
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# TabNet import (要インストール: pip install pytorch-tabnet)
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    print("Warning: pytorch-tabnet not installed. Run: pip install pytorch-tabnet")
    TABNET_AVAILABLE = False

# フォント設定
fonts = [f.name for f in fm.fontManager.ttflist]
if 'MS Gothic' in fonts:
    mpl.rcParams['font.family'] = 'MS Gothic'
mpl.rcParams['axes.unicode_minus'] = False

# パス設定
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "honhyo_with_interactions.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "experiments", "ensemble")
os.makedirs(RESULTS_DIR, exist_ok=True)

# カテゴリ変数
CATEGORICAL_COLS = [
    'night_terrain', 'road_shape_terrain', 'signal_road_shape',
    'night_road_condition', 'speed_shape_interaction',
    'party_type_daytime', 'party_type_road_shape'
]


def load_data():
    """データ読み込み"""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape}")
    return df


def prepare_features(df: pd.DataFrame):
    """特徴量準備"""
    print("\nPreparing features...")
    
    target_col = '死者数'
    drop_cols = [target_col]
    
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = (df[target_col] > 0).astype(int)
    
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target: {y.sum()} positives / {len(y)} total ({y.mean()*100:.2f}%)")
    
    return X, y, feature_cols


def prepare_category_encoding(X: pd.DataFrame, feature_cols: list):
    """
    TabNet用にカテゴリ変数をLabelEncodingし、
    カテゴリのインデックス(cat_idxs)と次元数(cat_dims)を計算する。
    ※数値変数はそのまま返す（CVループ内で処理する）
    """
    X_enc = X.copy()
    cat_idxs = []
    cat_dims = []
    cat_col_names = []
    
    for i, col in enumerate(feature_cols):
        # 指定されたカテゴリ変数 または オブジェクト型
        if col in CATEGORICAL_COLS or X_enc[col].dtype == 'object':
            le = LabelEncoder()
            # 欠損は 'missing' として扱う
            X_enc[col] = X_enc[col].astype(str).fillna('missing')
            X_enc[col] = le.fit_transform(X_enc[col])
            
            cat_idxs.append(i)
            cat_dims.append(len(le.classes_))
            cat_col_names.append(col)
        else:
            # 数値変数はここでは何もしない（欠損値もそのまま）
            pass
    
    return X_enc, cat_idxs, cat_dims, cat_col_names


def train_lightgbm_cv(X: pd.DataFrame, y: pd.Series, feature_cols: list, n_folds: int = 5):
    """LightGBM CV学習"""
    print("\n" + "=" * 60)
    print("Training LightGBM...")
    print("=" * 60)
    
    # カテゴリ変数処理
    X_lgb = X.copy()
    cat_cols = [c for c in feature_cols if c in CATEGORICAL_COLS or X[c].dtype == 'object']
    for col in cat_cols:
        X_lgb[col] = X_lgb[col].astype('category')
    
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 63,
        'min_child_samples': 10,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42,
        'n_jobs': -1,
        'is_unbalance': True
    }
    
    oof_proba = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold + 1}/{n_folds}...")
        
        X_train, X_val = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=cat_cols)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=1000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, first_metric_only=True),
                lgb.log_evaluation(200)
            ]
        )
        
        oof_proba[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        gc.collect()
    
    auc = roc_auc_score(y, oof_proba)
    print(f"\n  LightGBM OOF AUC: {auc:.4f}")
    
    return oof_proba, auc


def train_tabnet_cv(X: pd.DataFrame, y: pd.Series, feature_cols: list, n_folds: int = 5):
    """TabNet CV学習（修正版：スケーリングとリーク防止対応）"""
    print("\n" + "=" * 60)
    print("Training TabNet (Corrected: Scaling + No Data Leakage)...")
    print("=" * 60)
    
    if not TABNET_AVAILABLE:
        print("  TabNet not available. Returning zeros.")
        return np.zeros(len(X)), 0.0
    
    # 1. カテゴリ変数のエンコーディング（IDの整合性のため全体で実施）
    X_enc, cat_idxs, cat_dims, cat_col_names = prepare_category_encoding(X, feature_cols)
    X_values = X_enc.values.copy()
    y_values = y.values
    
    # 数値変数の列インデックスを特定
    num_idxs = [i for i, c in enumerate(feature_cols) if c not in cat_col_names]
    print(f"  Categorical features: {len(cat_col_names)}")
    print(f"  Numerical features: {len(num_idxs)}")
    
    oof_proba = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"  Fold {fold + 1}/{n_folds}...")
        
        # データの分割
        X_train = X_values[train_idx].copy().astype(np.float32)
        X_val = X_values[val_idx].copy().astype(np.float32)
        y_train, y_val = y_values[train_idx], y_values[val_idx]
        
        # --- 数値変数の前処理 (CVループ内で実施: データリーク防止) ---
        if len(num_idxs) > 0:
            # 2. 欠損値埋め (Trainの中央値で埋める)
            imputer = SimpleImputer(strategy='median')
            X_train[:, num_idxs] = imputer.fit_transform(X_train[:, num_idxs])
            X_val[:, num_idxs] = imputer.transform(X_val[:, num_idxs])
            
            # 3. スケーリング (Trainの統計量で標準化)
            scaler = StandardScaler()
            X_train[:, num_idxs] = scaler.fit_transform(X_train[:, num_idxs])
            X_val[:, num_idxs] = scaler.transform(X_val[:, num_idxs])
        # -----------------------------------------------
        
        model = TabNetClassifier(
            n_d=16,
            n_a=16,
            n_steps=3,
            gamma=1.5,
            lambda_sparse=1e-3,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=1,
            optimizer_params={'lr': 0.02},
            scheduler_params={'step_size': 10, 'gamma': 0.9},
            verbose=0,
            seed=42
        )
        
        # バッチサイズ調整 (メモリ不足時は下げてください)
        BATCH_SIZE = 4096
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['auc'],
            max_epochs=100,
            patience=20,
            batch_size=BATCH_SIZE,
            virtual_batch_size=256
        )
        
        preds = model.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = preds
        
        del model
        gc.collect()
    
    auc = roc_auc_score(y, oof_proba)
    print(f"\n  TabNet OOF AUC: {auc:.4f}")
    
    return oof_proba, auc


def ensemble_predictions(lgb_proba: np.ndarray, tabnet_proba: np.ndarray, 
                         lgb_weight: float = 0.6):
    """アンサンブル（重み付け平均）"""
    tabnet_weight = 1 - lgb_weight
    ensemble_proba = lgb_weight * lgb_proba + tabnet_weight * tabnet_proba
    return ensemble_proba


def evaluate_at_recall(y_true: np.ndarray, y_proba: np.ndarray, target_recall: float = 0.99):
    """特定Recall時のPrecisionを計算"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # 指定Recall以上で最大のPrecisionを探す
    valid_idx = np.where(recall >= target_recall)[0]
    if len(valid_idx) == 0:
        return 0.0, 0.0
    
    best_idx = valid_idx[np.argmax(precision[valid_idx])]
    return precision[best_idx], thresholds[best_idx] if best_idx < len(thresholds) else 0.5


def generate_report(y: pd.Series, lgb_proba: np.ndarray, tabnet_proba: np.ndarray,
                    ensemble_proba: np.ndarray, lgb_auc: float, tabnet_auc: float):
    """レポート生成"""
    print("\n" + "=" * 60)
    print("Generating Report...")
    print("=" * 60)
    
    # 各モデルの評価
    ensemble_auc = roc_auc_score(y, ensemble_proba)
    
    lgb_prec, lgb_thresh = evaluate_at_recall(y.values, lgb_proba)
    tabnet_prec, tabnet_thresh = evaluate_at_recall(y.values, tabnet_proba)
    ensemble_prec, ensemble_thresh = evaluate_at_recall(y.values, ensemble_proba)
    
    print(f"  LightGBM   - AUC: {lgb_auc:.4f}, Precision@99%Recall: {lgb_prec:.4f}")
    print(f"  TabNet     - AUC: {tabnet_auc:.4f}, Precision@99%Recall: {tabnet_prec:.4f}")
    print(f"  Ensemble   - AUC: {ensemble_auc:.4f}, Precision@99%Recall: {ensemble_prec:.4f}")
    
    # 比較プロット
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # AUC比較
    models = ['LightGBM', 'TabNet', 'Ensemble']
    aucs = [lgb_auc, tabnet_auc, ensemble_auc]
    colors = ['#3498db', '#e74c3c', '#2ecc71']
    axes[0].bar(models, aucs, color=colors)
    axes[0].set_ylabel('AUC')
    axes[0].set_title('Model Comparison: AUC')
    axes[0].set_ylim(0.8, 0.9)
    for i, v in enumerate(aucs):
        axes[0].text(i, v + 0.002, f'{v:.4f}', ha='center')
    
    # Precision比較
    precs = [lgb_prec, tabnet_prec, ensemble_prec]
    axes[1].bar(models, precs, color=colors)
    axes[1].set_ylabel('Precision @ 99% Recall')
    axes[1].set_title('Model Comparison: Precision')
    for i, v in enumerate(precs):
        axes[1].text(i, v + 0.0002, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'model_comparison.png'), dpi=150)
    plt.close()
    
    # レポートファイル
    report_path = os.path.join(RESULTS_DIR, 'ensemble_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# LightGBM + TabNet Ensemble Report\n\n")
        f.write("## 概要\n")
        f.write("自転車・歩行者など「非線形な挙動」を示す主体の誤検知削減を目的とした\n")
        f.write("LightGBM と TabNet のアンサンブル実験結果。\n\n")
        
        f.write("## モデル性能比較\n\n")
        f.write("| Model | AUC | Precision@99%Recall | Threshold |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| LightGBM | {lgb_auc:.4f} | {lgb_prec:.4f} | {lgb_thresh:.4f} |\n")
        f.write(f"| TabNet | {tabnet_auc:.4f} | {tabnet_prec:.4f} | {tabnet_thresh:.4f} |\n")
        f.write(f"| **Ensemble** | **{ensemble_auc:.4f}** | **{ensemble_prec:.4f}** | {ensemble_thresh:.4f} |\n\n")
        
        f.write("## アンサンブル設定\n")
        f.write("- **重み**: LightGBM 60% + TabNet 40%\n")
        f.write("- **TabNet パラメータ**: n_d=16, n_a=16, n_steps=3, gamma=1.5\n\n")
        
        f.write("## 可視化\n")
        f.write("![Model Comparison](model_comparison.png)\n\n")
        
        improvement = (ensemble_prec - lgb_prec) / lgb_prec * 100 if lgb_prec > 0 else 0
        if improvement > 0:
            f.write(f"## 結論\n")
            f.write(f"> [!TIP]\n")
            f.write(f"> アンサンブルにより Precision が **{improvement:.1f}%** 向上しました。\n")
        else:
            f.write(f"## 結論\n")
            f.write(f"> [!NOTE]\n")
            f.write(f"> アンサンブルによる大きな改善は見られませんでした。LightGBM単体が既に強力です。\n")
    
    print(f"\nReport saved: {report_path}")
    
    return {
        'lgb_auc': lgb_auc,
        'tabnet_auc': tabnet_auc,
        'ensemble_auc': ensemble_auc,
        'lgb_prec': lgb_prec,
        'tabnet_prec': tabnet_prec,
        'ensemble_prec': ensemble_prec
    }


def main():
    print("=" * 60)
    print("LightGBM + TabNet Ensemble Experiment")
    print("=" * 60)
    
    # データ読み込み
    df = load_data()
    X, y, feature_cols = prepare_features(df)
    
    # LightGBM学習
    lgb_proba, lgb_auc = train_lightgbm_cv(X, y, feature_cols)
    
    # TabNet学習
    tabnet_proba, tabnet_auc = train_tabnet_cv(X, y, feature_cols)
    
    # アンサンブル
    ensemble_proba = ensemble_predictions(lgb_proba, tabnet_proba, lgb_weight=0.6)
    
    # 予測結果保存
    results_df = pd.DataFrame({
        'true_label': y,
        'lgb_proba': lgb_proba,
        'tabnet_proba': tabnet_proba,
        'ensemble_proba': ensemble_proba
    })
    results_df.to_csv(os.path.join(RESULTS_DIR, 'ensemble_predictions.csv'), index=False)
    
    # レポート生成
    generate_report(y, lgb_proba, tabnet_proba, ensemble_proba, lgb_auc, tabnet_auc)
    
    print("\n" + "=" * 60)
    print("Experiment Complete!")
    print(f"Results saved to: {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
