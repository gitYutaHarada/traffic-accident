"""
2段階モデル（Two-Stage Cascade Model）学習スクリプト
======================================================
Implementation Plan v5 に基づく実装

Stage 1: RandomForest (OOF, Recall 99%閾値)
Stage 2: LightGBM (動的クラス重み, 相互作用特徴量)

推定実行時間: 10-15分 (Core Ultra 9 / 64GB RAM)
"""

import pandas as pd
import numpy as np
import os
import gc
import joblib
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


class TwoStagePipeline:
    """
    2段階モデルのパイプライン
    """
    
    def __init__(
        self,
        data_path: str = "data/processed/honhyo_clean_with_features.csv",
        target_col: str = "死者数",
        n_folds: int = 5,
        random_state: int = 42,
        stage1_recall_target: float = 0.99,
        top_k_features: int = 10
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.n_folds = n_folds
        self.random_state = random_state
        self.stage1_recall_target = stage1_recall_target
        self.top_k_features = top_k_features
        
        self.output_dir = "results/two_stage_model"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # モデル保存先
        self.stage1_models = []
        self.stage2_model = None
        self.threshold_stage1 = None
        self.top_features = None
        self.interaction_pairs = None
        
        print("=" * 70)
        print("2段階モデル（Two-Stage Cascade Model）学習スクリプト")
        print("=" * 70)
        print(f"出力先: {self.output_dir}")
        print(f"Stage 1 Recall目標: {self.stage1_recall_target:.0%}")
    
    def load_data(self):
        """データ読み込みと前処理"""
        print("\n📂 データ読み込み中...")
        self.df = pd.read_csv(self.data_path)
        print(f"   データ形状: {self.df.shape}")
        
        self.y = self.df[self.target_col].values
        self.X = self.df.drop(columns=[self.target_col])
        
        if '発生日時' in self.X.columns:
            self.X = self.X.drop(columns=['発生日時'])
        
        # カテゴリカル特徴量の特定
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
        
        # エンコーダー準備
        self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.ordinal_encoder.fit(self.X[self.categorical_cols])
        
        self.feature_names = self.numeric_cols + self.categorical_cols
        
        # エンコード済みデータの作成
        X_cat_enc = self.ordinal_encoder.transform(self.X[self.categorical_cols])
        self.X_encoded = np.hstack([self.X[self.numeric_cols].values, X_cat_enc])
        
        print(f"   正例（死亡）: {self.y.sum():,} / {len(self.y):,} ({self.y.mean()*100:.3f}%)")
        gc.collect()
    
    def train_stage1_oof(self):
        """Stage 1: RandomForest OOF予測 & Feature Importance"""
        print(f"\n🌲 Stage 1: RandomForest OOF学習 ({self.n_folds}-Fold)...")
        
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        self.oof_proba_stage1 = np.zeros(len(self.y))
        feature_importances = np.zeros(len(self.feature_names))
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_encoded, self.y)):
            print(f"   Fold {fold + 1}/{self.n_folds}...")
            
            X_train = self.X_encoded[train_idx]
            X_val = self.X_encoded[val_idx]
            y_train = self.y[train_idx]
            
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=20,
                class_weight='balanced',
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            y_prob = model.predict_proba(X_val)[:, 1]
            self.oof_proba_stage1[val_idx] = y_prob
            
            feature_importances += model.feature_importances_ / self.n_folds
            self.stage1_models.append(model)
            
            del model, X_train, X_val
            gc.collect()
        
        # Feature Importance 集計
        self.feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        self.top_features = self.feature_importance_df.head(self.top_k_features)['feature'].tolist()
        print(f"\n   📊 Top {self.top_k_features} 特徴量: {self.top_features}")
        
        # OOF精度確認
        oof_pred = (self.oof_proba_stage1 >= 0.5).astype(int)
        recall_05 = recall_score(self.y, oof_pred)
        precision_05 = precision_score(self.y, oof_pred)
        print(f"   OOF (閾値0.5): Precision={precision_05:.4f}, Recall={recall_05:.4f}")
    
    def find_recall_threshold(self):
        """Recall目標を満たす閾値を探索"""
        print(f"\n📐 Recall {self.stage1_recall_target:.0%} 閾値探索...")
        
        # 閾値を下げていき、Recallが目標を超えるものを探す
        thresholds = np.arange(0.01, 0.5, 0.01)
        
        for thresh in thresholds:
            y_pred = (self.oof_proba_stage1 >= thresh).astype(int)
            recall = recall_score(self.y, y_pred)
            if recall >= self.stage1_recall_target:
                self.threshold_stage1 = thresh
                break
        else:
            # 目標Recallに達しない場合は最低閾値を使用
            self.threshold_stage1 = 0.01
        
        # 最終確認
        y_pred_final = (self.oof_proba_stage1 >= self.threshold_stage1).astype(int)
        recall_final = recall_score(self.y, y_pred_final)
        precision_final = precision_score(self.y, y_pred_final)
        n_candidates = y_pred_final.sum()
        
        print(f"   選定閾値: {self.threshold_stage1:.3f}")
        print(f"   Recall: {recall_final:.4f}, Precision: {precision_final:.4f}")
        print(f"   Stage 2 候補数: {n_candidates:,} / {len(self.y):,} ({n_candidates/len(self.y)*100:.2f}%)")
        
        # Stage 2用インデックス
        self.stage2_mask = self.oof_proba_stage1 >= self.threshold_stage1
    
    def generate_interaction_features(self, X_subset, indices):
        """重要度上位特徴量に基づく相互作用特徴量を生成"""
        print(f"\n🔧 相互作用特徴量生成 (Top {self.top_k_features} 間)...")
        
        # 特徴量インデックスの取得
        top_feature_indices = [self.feature_names.index(f) for f in self.top_features if f in self.feature_names]
        
        interaction_features = []
        interaction_names = []
        
        # ペアごとに掛け算特徴量を生成
        for i, idx1 in enumerate(top_feature_indices):
            for idx2 in top_feature_indices[i+1:]:
                f1_name = self.feature_names[idx1]
                f2_name = self.feature_names[idx2]
                
                interaction = X_subset[:, idx1] * X_subset[:, idx2]
                interaction_features.append(interaction)
                interaction_names.append(f"{f1_name}*{f2_name}")
        
        if interaction_features:
            interaction_matrix = np.column_stack(interaction_features)
            X_augmented = np.hstack([X_subset, interaction_matrix])
            augmented_names = self.feature_names + interaction_names
            print(f"   生成された相互作用特徴量: {len(interaction_names)}")
        else:
            X_augmented = X_subset
            augmented_names = self.feature_names
        
        self.interaction_names = interaction_names
        return X_augmented, augmented_names
    
    def train_stage2(self):
        """Stage 2: LightGBM学習"""
        print(f"\n🌿 Stage 2: LightGBM 学習...")
        
        # Stage 2用データ抽出
        X_stage2_base = self.X_encoded[self.stage2_mask]
        y_stage2 = self.y[self.stage2_mask]
        
        # 相互作用特徴量追加
        X_stage2, augmented_names = self.generate_interaction_features(
            X_stage2_base, np.where(self.stage2_mask)[0]
        )
        
        # 不均衡比の計算
        n_pos = y_stage2.sum()
        n_neg = len(y_stage2) - n_pos
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
        
        print(f"   Stage 2 データ: {len(y_stage2):,} (Pos: {n_pos:,}, Neg: {n_neg:,})")
        print(f"   動的 scale_pos_weight: {scale_pos_weight:.2f}")
        
        # LightGBM学習
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'n_estimators': 500,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': 8,
            'scale_pos_weight': scale_pos_weight,
            'random_state': self.random_state,
            'n_jobs': -1
        }
        
        self.stage2_model = lgb.LGBMClassifier(**lgb_params)
        self.stage2_model.fit(X_stage2, y_stage2)
        
        # Stage 2 OOF予測（簡易確認）
        y_prob_stage2 = self.stage2_model.predict_proba(X_stage2)[:, 1]
        y_pred_stage2 = (y_prob_stage2 >= 0.5).astype(int)
        
        recall_s2 = recall_score(y_stage2, y_pred_stage2)
        precision_s2 = precision_score(y_stage2, y_pred_stage2)
        print(f"   Stage 2 Train精度: Precision={precision_s2:.4f}, Recall={recall_s2:.4f}")
        
        self.augmented_feature_names = augmented_names
    
    def evaluate_pipeline(self):
        """パイプライン全体の評価"""
        print(f"\n📈 パイプライン全体評価...")
        
        # Stage 1 フィルタリング後のデータでStage 2予測
        X_stage2_base = self.X_encoded[self.stage2_mask]
        y_stage2_true = self.y[self.stage2_mask]
        
        # 相互作用特徴量追加
        X_stage2, _ = self.generate_interaction_features(
            X_stage2_base, np.where(self.stage2_mask)[0]
        )
        
        y_prob_final = self.stage2_model.predict_proba(X_stage2)[:, 1]
        
        # 全体に対する最終予測
        final_proba = np.zeros(len(self.y))
        final_proba[self.stage2_mask] = y_prob_final
        
        # 評価
        y_pred_final = (final_proba >= 0.5).astype(int)
        
        recall_final = recall_score(self.y, y_pred_final)
        precision_final = precision_score(self.y, y_pred_final) if y_pred_final.sum() > 0 else 0
        f1_final = f1_score(self.y, y_pred_final)
        
        print(f"\n   ========== 最終結果 ==========")
        print(f"   Precision: {precision_final:.4f}")
        print(f"   Recall:    {recall_final:.4f}")
        print(f"   F1-Score:  {f1_final:.4f}")
        
        # Baseline比較（Stage 1 単独）
        y_pred_s1 = (self.oof_proba_stage1 >= 0.5).astype(int)
        precision_s1 = precision_score(self.y, y_pred_s1)
        recall_s1 = recall_score(self.y, y_pred_s1)
        
        print(f"\n   --- Baseline（Stage 1単独, 閾値0.5）---")
        print(f"   Precision: {precision_s1:.4f}")
        print(f"   Recall:    {recall_s1:.4f}")
        
        # 改善幅
        precision_improvement = (precision_final - precision_s1) / precision_s1 * 100 if precision_s1 > 0 else 0
        print(f"\n   ⬆️ Precision改善: {precision_improvement:+.1f}%")
        
        # 結果保存
        results = {
            'stage1_threshold': self.threshold_stage1,
            'stage1_recall': recall_score(self.y, (self.oof_proba_stage1 >= self.threshold_stage1).astype(int)),
            'final_precision': precision_final,
            'final_recall': recall_final,
            'final_f1': f1_final,
            'baseline_precision': precision_s1,
            'baseline_recall': recall_s1,
            'precision_improvement_pct': precision_improvement
        }
        
        pd.DataFrame([results]).to_csv(os.path.join(self.output_dir, "evaluation_results.csv"), index=False)
        self.feature_importance_df.to_csv(os.path.join(self.output_dir, "feature_importance.csv"), index=False)
        
        return results
    
    def save_models(self):
        """モデル保存"""
        print(f"\n💾 モデル保存中...")
        joblib.dump(self.stage1_models, os.path.join(self.output_dir, "stage1_models.pkl"))
        joblib.dump(self.stage2_model, os.path.join(self.output_dir, "stage2_model.pkl"))
        joblib.dump({
            'threshold_stage1': self.threshold_stage1,
            'top_features': self.top_features,
            'interaction_names': self.interaction_names,
            'feature_names': self.feature_names,
            'ordinal_encoder': self.ordinal_encoder
        }, os.path.join(self.output_dir, "pipeline_config.pkl"))
        print(f"   保存完了: {self.output_dir}")
    
    def run(self):
        """パイプライン実行"""
        start_time = datetime.now()
        
        self.load_data()
        self.train_stage1_oof()
        self.find_recall_threshold()
        self.train_stage2()
        results = self.evaluate_pipeline()
        self.save_models()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\n{'='*70}")
        print(f"✅ 完了！ 実行時間: {elapsed/60:.1f}分")
        print(f"{'='*70}")
        
        return results


if __name__ == "__main__":
    pipeline = TwoStagePipeline()
    results = pipeline.run()
