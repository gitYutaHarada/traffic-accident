"""
誤検知（False Positive）分析スクリプト v2.1
===========================================
Implementation Planに基づき、FPの詳細分析を行う。

主な機能:
1. OOF (Out-of-Fold) 予測スコアの生成・保存
2. 元データとの結合による重傷度情報の復元
3. 厳しい閾値による「頑固なFP (Hard FP)」の抽出
4. 重傷度・地理的分布の分析
5. 高確信度FPに対するSHAP個票分析

修正(v2.1):
- SHAP Force Plotエラー修正
- --skip-cv オプション追加（既存結果を用いて分析のみ再実行）

推定実行時間: 約30分〜1時間（--skip-cv時は数分）
"""

import pandas as pd
import numpy as np
import os
import gc
import json
import argparse
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings

warnings.filterwarnings('ignore')

# 日本語フォント設定 (Windows)
mpl.rcParams['font.family'] = 'MS Gothic'
mpl.rcParams['axes.unicode_minus'] = False


class FalsePositiveAnalyzer:
    """
    誤検知（False Positive）の詳細分析を行うクラス
    """
    
    def __init__(
        self,
        features_path: str = "data/processed/honhyo_clean_with_features.csv",
        raw_path: str = "data/raw/honhyo_all_shishasuu_binary.csv",
        target_col: str = "死者数",
        n_folds: int = 5,
        random_state: int = 42
    ):
        self.features_path = features_path
        self.raw_path = raw_path
        self.target_col = target_col
        self.n_folds = n_folds
        self.random_state = random_state
        
        self.output_dir = "results/analysis/fp_analysis_v2"
        self.shap_dir = os.path.join(self.output_dir, "shap_force_plots")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.shap_dir, exist_ok=True)
        
        self.oof_proba = None
        self.df_merged = None
        self.threshold_strict = None
        self.df_fp_hard = None
        
        print("=" * 70)
        print("誤検知（False Positive）分析スクリプト v2.1")
        print("=" * 70)
        print(f"出力先: {self.output_dir}")
    
    def load_data(self):
        """データ読み込みと前処理"""
        print("\n📂 データ読み込み中...")
        self.df_features = pd.read_csv(self.features_path)
        print(f"   特徴量データ: {self.df_features.shape}")
        
        self.df_raw = pd.read_csv(self.raw_path)
        print(f"   元データ: {self.df_raw.shape}")
        
        self.y = self.df_features[self.target_col].values
        self.X = self.df_features.drop(columns=[self.target_col])
        
        if '発生日時' in self.X.columns:
            self.X = self.X.drop(columns=['発生日時'])
        
        known_categoricals = [
            '都道府県コード', '市区町村コード', '警察署等コード',
            '昼夜', '天候', '地形', '路面状態', '道路形状', '信号機',
            '衝突地点', 'ゾーン規制', '中央分離帯施設等', '歩車道区分',
            '事故類型', '曜日(発生年月日)', '祝日(発生年月日)',
            'road_type', 'area_id', '地点コード'
        ]
        
        self.categorical_cols = []
        self.numeric_cols = []
        
        for col in self.X.columns:
            if col in known_categoricals or self.X[col].dtype == 'object':
                self.categorical_cols.append(col)
                self.X[col] = self.X[col].astype(str).fillna('Missing')
            else:
                self.numeric_cols.append(col)
                median_val = self.X[col].median()
                self.X[col] = self.X[col].fillna(median_val).astype(np.float32)
        
        print(f"   数値特徴量: {len(self.numeric_cols)}, カテゴリ特徴量: {len(self.categorical_cols)}")
        gc.collect()
        
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.ordinal_encoder.fit(self.X[self.categorical_cols])
        self.feature_names = self.numeric_cols + self.categorical_cols

    def train_full_model(self, model_name: str = "RandomForest"):
        """SHAP分析用に全データでモデルを再学習"""
        print(f"\n🧠 モデル再学習（全データ, {model_name}）...")
        X_cat_enc = self.ordinal_encoder.transform(self.X[self.categorical_cols])
        X_enc = np.hstack([self.X[self.numeric_cols].values, X_cat_enc])
        
        if model_name == "RandomForest":
            self.final_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_leaf=20,
                class_weight='balanced', random_state=self.random_state, n_jobs=-1
            )
            self.final_model.fit(X_enc, self.y)
            self.X_encoded = X_enc
        else:
            X_lgb = self.X.copy()
            for c in self.categorical_cols:
                X_lgb[c] = X_lgb[c].astype('category')
            lgb_params = {
                'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1,
                'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 31,
                'random_state': self.random_state, 'n_jobs': -1
            }
            self.final_model = lgb.LGBMClassifier(**lgb_params)
            self.final_model.fit(X_lgb, self.y)
            self.X_encoded = X_lgb
        print("   再学習完了")

    def generate_oof_predictions(self, model_name: str = "RandomForest"):
        """Out-of-Fold (OOF) 予測を生成"""
        print(f"\n🔮 OOF予測生成中 ({model_name})...")
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.oof_proba = np.zeros(len(self.y))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
            print(f"   Fold {fold + 1}/{self.n_folds}...")
            X_train = self.X.iloc[train_idx].copy()
            X_val = self.X.iloc[val_idx].copy()
            y_train = self.y[train_idx]
            
            X_train_cat_enc = self.ordinal_encoder.transform(X_train[self.categorical_cols])
            X_val_cat_enc = self.ordinal_encoder.transform(X_val[self.categorical_cols])
            
            X_train_enc = np.hstack([X_train[self.numeric_cols].values, X_train_cat_enc])
            X_val_enc = np.hstack([X_val[self.numeric_cols].values, X_val_cat_enc])
            
            if model_name == "RandomForest":
                model = RandomForestClassifier(
                    n_estimators=100, max_depth=10, min_samples_leaf=20,
                    class_weight='balanced', random_state=self.random_state, n_jobs=-1
                )
                model.fit(X_train_enc, y_train)
                y_prob = model.predict_proba(X_val_enc)[:, 1]
            else:
                pass 
            
            self.oof_proba[val_idx] = y_prob
            del model, X_train, X_val
            gc.collect()
        
        print(f"   OOF予測完了。平均予測確率: {self.oof_proba.mean():.4f}")
    
    def merge_raw_data(self):
        """元データと結合"""
        print("\n📎 元データとの結合...")
        severity_cols = ['人身損傷程度（当事者A）', '人身損傷程度（当事者B）', '負傷者数']
        self.df_merged = self.df_features.copy()
        self.df_merged['oof_proba'] = self.oof_proba
        self.df_merged['y_true'] = self.y
        for col in severity_cols:
            if col in self.df_raw.columns:
                self.df_merged[col] = self.df_raw[col].values
        print(f"   結合完了: {self.df_merged.shape}")
    
    def find_precision_threshold(self, target_precision: float = 0.20):
        """Precision目標達成閾値の探索"""
        print(f"\n📐 閾値探索 (目標Precision = {target_precision:.0%})...")
        thresholds = np.arange(0.1, 0.95, 0.01)
        
        for thresh in reversed(thresholds):
            y_pred = (self.oof_proba >= thresh).astype(int)
            tp = ((y_pred == 1) & (self.y == 1)).sum()
            fp = ((y_pred == 1) & (self.y == 0)).sum()
            if tp + fp > 0:
                precision = tp / (tp + fp)
                if precision >= target_precision:
                    self.threshold_strict = thresh
                    final_precision = precision
                    break
        else:
            self.threshold_strict = 0.5
            final_precision = 0.0 
        
        print(f"   選定閾値: {self.threshold_strict:.2f} → Precision: {final_precision:.2%}")

        threshold_analysis = []
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            y_pred = (self.oof_proba >= thresh).astype(int)
            tp = ((y_pred == 1) & (self.y == 1)).sum()
            fp = ((y_pred == 1) & (self.y == 0)).sum()
            fn = ((y_pred == 0) & (self.y == 1)).sum()
            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / (tp + fn) if tp + fn > 0 else 0
            threshold_analysis.append({
                'Threshold': thresh,
                'TP': tp, 'FP': fp, 'FN': fn, 'Precision': precision, 'Recall': recall
            })
        pd.DataFrame(threshold_analysis).to_csv(os.path.join(self.output_dir, "threshold_analysis.csv"), index=False)

    def analyze_false_positives(self):
        """FP詳細分析"""
        print("\n🔬 誤検知（FP）分析開始...")
        y_pred = (self.oof_proba >= self.threshold_strict).astype(int)
        fp_mask = (y_pred == 1) & (self.y == 0)
        self.df_fp_hard = self.df_merged[fp_mask].copy()
        
        print(f"   閾値{self.threshold_strict:.2f}でのHard FP数: {len(self.df_fp_hard):,}")
        self.df_fp_hard.to_csv(os.path.join(self.output_dir, "fp_high_confidence.csv"), index=False)
        
        print("\n📊 A. 重傷度分析...")
        if '人身損傷程度（当事者A）' in self.df_fp_hard.columns:
            severity_dist = self.df_fp_hard['人身損傷程度（当事者A）'].value_counts()
            severity_dist.to_csv(os.path.join(self.output_dir, "fp_severity_distribution.csv"))
            print(f"   重傷度分布 (Hard FP):")
            for val, cnt in severity_dist.head(5).items():
                print(f"      {val}: {cnt:,} ({cnt/len(self.df_fp_hard)*100:.1f}%)")
        
        print("\n📊 D. 地理的分析...")
        if 'area_id' in self.df_fp_hard.columns:
            total_by_area = self.df_merged['area_id'].value_counts()
            fp_by_area = self.df_fp_hard['area_id'].value_counts()
            fp_rate = (fp_by_area / total_by_area * 100).dropna().sort_values(ascending=False)
            fp_rate.head(20).to_csv(os.path.join(self.output_dir, "fp_rate_by_area.csv"))

    def shap_individual_analysis(self, top_n: int = 10):
        """SHAP個票分析（修正版 v2.2）"""
        print(f"\n🎯 C. SHAP個票分析 (Top {top_n})...")
        df_fp_sorted = self.df_fp_hard.sort_values('oof_proba', ascending=False).head(top_n)
        print(f"   対象: 最も確信度の高いFP {len(df_fp_sorted)} 件")
        
        try:
            # check_additivity=False to avoid minor numerical errors
            if isinstance(self.final_model, RandomForestClassifier):
                explainer = shap.TreeExplainer(self.final_model)
            else:
                explainer = shap.TreeExplainer(self.final_model)
        except Exception as e:
            print(f"   ⚠️ SHAP Explainer作成失敗: {e}")
            return
            
        for i, (idx, row) in enumerate(df_fp_sorted.iterrows()):
            print(f"   [{i+1}/{len(df_fp_sorted)}] Index={idx}, 確率={row['oof_proba']:.3f}")
            try:
                if isinstance(self.final_model, RandomForestClassifier):
                    X_point = self.X.iloc[[idx]]
                    X_cat_enc = self.ordinal_encoder.transform(X_point[self.categorical_cols])
                    X_point_enc = np.hstack([X_point[self.numeric_cols].values, X_cat_enc])
                    
                    shap_values = explainer.shap_values(X_point_enc, check_additivity=False)
                    
                    # 戻り値の型チェックと適切な取得
                    if isinstance(shap_values, list):
                        # print(f"      DEBUG: list len={len(shap_values)}")
                        if len(shap_values) > 1:
                            sv_target = shap_values[1][0] # Class 1
                        else:
                            sv_target = shap_values[0][0]
                    else:
                        # print(f"      DEBUG: array shape={shap_values.shape}")
                        if len(shap_values.shape) == 3:
                            sv_target = shap_values[0, :, 1]
                        elif len(shap_values.shape) == 2:
                             # (n_samples, n_features) -> binary classification single output
                             sv_target = shap_values[0]
                        else:
                             sv_target = shap_values[0] # Fallback
                    
                    base_value = explainer.expected_value
                    if isinstance(base_value, list):
                        base_value = base_value[1] # Class 1
                    elif isinstance(base_value, np.ndarray) and len(base_value) > 1:
                        base_value = base_value[1]
                        
                    plt.figure(figsize=(20, 6))
                    shap.force_plot(
                        base_value, sv_target, X_point_enc[0],
                        feature_names=self.feature_names, matplotlib=True, show=False
                    )
                else:
                    pass 
                
                plt.title(f"FP Index: {idx}, Prob: {row['oof_proba']:.3f}", fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(self.shap_dir, f"force_plot_{i+1:02d}_idx{idx}.png"), bbox_inches='tight', dpi=100)
                plt.close()
            except Exception as e:
                print(f"      ⚠️ Force Plot生成失敗: {e}")
                # import traceback
                # traceback.print_exc()

    def generate_report(self):
        """分析レポート生成"""
        print("\n📝 レポート生成中...")
        y_pred_05 = (self.oof_proba >= 0.5).astype(int)
        tp_05 = ((y_pred_05 == 1) & (self.y == 1)).sum()
        fp_05 = ((y_pred_05 == 1) & (self.y == 0)).sum()
        
        y_pred_strict = (self.oof_proba >= self.threshold_strict).astype(int)
        tp_s = ((y_pred_strict == 1) & (self.y == 1)).sum()
        fp_s = ((y_pred_strict == 1) & (self.y == 0)).sum()
        
        report = f"""# 誤検知（False Positive）分析レポート v2.1

生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 概要
本レポートは、死亡事故予測モデルの誤検知（False Positive）を詳細に分析した結果をまとめる。

## 1. 全体結果（閾値 = 0.5）
- True Positive (TP): {tp_05:,}
- False Positive (FP): {fp_05:,}
- Precision: {tp_05/(tp_05+fp_05)*100:.2f}%

## 2. 厳しい閾値での分析（閾値 = {self.threshold_strict:.2f}）
Precision 20%以上を目標とした閾値。
- Hard False Positive: {fp_s:,}
- Precision: {tp_s/(tp_s+fp_s)*100:.2f}%

## 3. 重傷度分析
詳細: `fp_severity_distribution.csv`

## 4. SHAP個票分析
保存先: `shap_force_plots/`
"""
        with open(os.path.join(self.output_dir, "fp_analysis_v2_report.md"), 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"   レポート保存: {os.path.join(self.output_dir, 'fp_analysis_v2_report.md')}")

    def run(self, model_name: str = "RandomForest", skip_cv: bool = False):
        self.load_data()
        
        oof_path = os.path.join(self.output_dir, "oof_proba.csv")
        
        if skip_cv and os.path.exists(oof_path):
            print(f"\n⏩ 既存のOOF予測結果を読み込み中: {oof_path}")
            self.oof_proba = pd.read_csv(oof_path)['oof_proba'].values
            if len(self.oof_proba) != len(self.y):
                print("⚠️ サイズ不一致のため再計算します")
                self.generate_oof_predictions(model_name=model_name)
        else:
            if skip_cv:
                print("⚠️ 既存のOOF予測結果が見つからないため、再計算を行います。")
            self.generate_oof_predictions(model_name=model_name)
            pd.DataFrame({'oof_proba': self.oof_proba}).to_csv(oof_path, index=False)

        self.train_full_model(model_name=model_name)
        self.merge_raw_data()
        self.find_precision_threshold(target_precision=0.20)
        self.analyze_false_positives()
        self.shap_individual_analysis(top_n=10)
        self.generate_report()

        print("\n" + "=" * 70)
        print("✅ 分析完了!")
        print(f"   結果ディレクトリ: {self.output_dir}")
        print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-cv', action='store_true', help='Skip CV if OOF probabilities are already saved')
    args = parser.parse_args()
    
    analyzer = FalsePositiveAnalyzer()
    analyzer.run(model_name="RandomForest", skip_cv=args.skip_cv)
