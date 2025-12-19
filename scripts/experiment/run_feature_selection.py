"""
特徴量選択スクリプト
==================
Permutation Importance（順列重要度）と多重共線性の確認を行い、
削除候補となる特徴量をレポートします。

注意点:
- 相関行列は純粋な数値変数のみに限定（カテゴリ変数はPearson相関が無意味）
- 評価指標はLogLossを使用（不均衡データに対してより敏感）
- prob_stage1 / logits_stage1 は削除禁止リストで保護
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss, make_scorer
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')


# ============================================================
# 設定
# ============================================================

# 削除禁止リスト（保護する列）
KEEP_COLS = ['prob_stage1', 'logits_stage1', 'area_id']

# カテゴリ変数リスト（相関チェックから除外、LightGBMにはカテゴリとして渡す）
CATEGORICAL_COLS = [
    '都道府県コード', '市区町村コード', '警察署等コード',
    '昼夜', '天候', '地形', '路面状態', '道路形状', '信号機',
    '衝突地点', 'ゾーン規制', '中央分離帯施設等', '歩車道区分',
    '事故類型', '曜日(発生年月日)', '祝日(発生年月日)',
    'road_type', 'area_id', '地点コード'
]

# Permutation Importanceで「ノイズ」と判断する閾値
# importance_mean <= この値なら削除候補
NOISE_THRESHOLD = 0.0

# 相関係数の閾値（これを超えると冗長とみなす）
CORRELATION_THRESHOLD = 0.95

# Permutation Importance計算時のサンプル数（高速化のため）
# 全データで計算すると時間がかかるため、サンプリングして計算
PI_SAMPLE_SIZE = 50000

# Permutation Importanceの繰り返し回数（安定化のため）
N_REPEATS = 5


class FeatureSelector:
    """特徴量選択・分析クラス"""
    
    def __init__(
        self,
        data_path: str = "data/processed/honhyo_clean_with_features.csv",
        target_col: str = "死者数",
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.data_path = data_path
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        
        self.output_dir = "results/analysis"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("=" * 60)
        print("特徴量選択スクリプト")
        print("=" * 60)
        print(f"評価指標: LogLoss（不均衡データ対応）")
        print(f"削除禁止リスト: {KEEP_COLS}")
        print(f"Permutation Importance 繰り返し回数: {N_REPEATS}")
        print(f"相関係数閾値: {CORRELATION_THRESHOLD}")
        print("=" * 60)
    
    def load_data(self):
        """データ読み込みとTrain/Validation分割"""
        print("\n📂 データ読み込み中...")
        self.df = pd.read_csv(self.data_path)
        
        y_all = self.df[self.target_col].values
        X_all = self.df.drop(columns=[self.target_col])
        
        if '発生日時' in X_all.columns:
            X_all = X_all.drop(columns=['発生日時'])
        
        # Train/Validation分割（層化抽出）
        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(
            X_all, y_all, test_size=self.test_size,
            random_state=self.random_state, stratify=y_all
        )
        
        print(f"   Train: {len(self.X_train):,} (正例: {self.y_train.sum():,})")
        print(f"   Valid: {len(self.X_valid):,} (正例: {self.y_valid.sum():,})")
        
        # カテゴリ変数と数値変数を分類
        self.categorical_cols = []
        self.numeric_cols = []
        
        for col in self.X_train.columns:
            if col in CATEGORICAL_COLS or self.X_train[col].dtype == 'object':
                self.categorical_cols.append(col)
                self.X_train[col] = self.X_train[col].astype('category')
                self.X_valid[col] = self.X_valid[col].astype('category')
            else:
                self.numeric_cols.append(col)
                self.X_train[col] = self.X_train[col].astype(np.float32)
                self.X_valid[col] = self.X_valid[col].astype(np.float32)
        
        self.feature_names = list(self.X_train.columns)
        
        print(f"   特徴量数: {len(self.feature_names)}")
        print(f"   - 数値変数: {len(self.numeric_cols)}")
        print(f"   - カテゴリ変数: {len(self.categorical_cols)}")
    
    def train_baseline(self):
        """ベースラインLightGBMモデルの学習"""
        print("\n🌿 ベースラインLightGBM学習中...")
        
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'verbosity': -1,
            'num_leaves': 31,
            'max_depth': 8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'is_unbalance': True,  # 不均衡データ対応
            'n_estimators': 500,
            'learning_rate': 0.05,
            'n_jobs': -1,
            'random_state': self.random_state,
        }
        
        self.model = lgb.LGBMClassifier(**lgb_params)
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_valid, self.y_valid)],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # ベースラインのLogLoss
        y_pred_proba = self.model.predict_proba(self.X_valid)[:, 1]
        self.baseline_logloss = log_loss(self.y_valid, y_pred_proba)
        
        print(f"   ベースライン LogLoss: {self.baseline_logloss:.6f}")
        
        # Feature Importance (split)
        self.feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"   上位10特徴量 (Feature Importance):")
        for i, row in self.feature_importance_df.head(10).iterrows():
            print(f"      {row['feature']}: {row['importance']:.0f}")
    
    def calculate_permutation_importance(self):
        """Permutation Importance（順列重要度）の計算"""
        print(f"\n🔀 Permutation Importance 計算中... (n_repeats={N_REPEATS})")
        print("   ※ 検証データで各特徴量をシャッフルし、LogLossの悪化を測定")
        
        # サンプリング（高速化のため）
        n_valid = len(self.X_valid)
        if n_valid > PI_SAMPLE_SIZE:
            print(f"   📉 高速化のためサンプリング: {n_valid:,} → {PI_SAMPLE_SIZE:,} 件")
            np.random.seed(self.random_state)
            sample_idx = np.random.choice(n_valid, size=PI_SAMPLE_SIZE, replace=False)
            X_valid_sample = self.X_valid.iloc[sample_idx]
            y_valid_sample = self.y_valid[sample_idx]
        else:
            X_valid_sample = self.X_valid
            y_valid_sample = self.y_valid
        
        # LogLossスコアラー: 組み込みの 'neg_log_loss' を使用
        # （make_scorerでカスタム関数を渡すとPython 3.14で問題が発生するため）
        
        result = permutation_importance(
            self.model,
            X_valid_sample,
            y_valid_sample,
            scoring='neg_log_loss',  # 組み込みスコアラーを使用
            n_repeats=N_REPEATS,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # 結果をDataFrameに整理
        # sklearn の permutation_importance は:
        # - スコアが悪化（＝その特徴量が重要）した場合 → 正の値を返す
        # - スコアが変わらない or 良くなる（＝ノイズ）→ 0以下を返す
        # ※ neg_log_loss を使用しているため、LogLossが増えると負のスコアが減少し、
        #   importances_mean は正の値になる
        self.perm_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance_mean': result.importances_mean,  # そのまま使用（正=重要、0以下=ノイズ）
            'importance_std': result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        # ノイズ候補の特定（KEEP_COLSを除外）
        noise_candidates = self.perm_importance_df[
            (self.perm_importance_df['importance_mean'] <= NOISE_THRESHOLD) &
            (~self.perm_importance_df['feature'].isin(KEEP_COLS))
        ]['feature'].tolist()
        
        self.noise_features = noise_candidates
        
        print(f"   計算完了")
        print(f"   ノイズ候補 (importance <= {NOISE_THRESHOLD}): {len(noise_candidates)} 件")
        
        if noise_candidates:
            print("   ノイズ候補リスト:")
            for feat in noise_candidates[:10]:  # 最大10件表示
                imp = self.perm_importance_df[self.perm_importance_df['feature'] == feat]['importance_mean'].values[0]
                print(f"      - {feat}: {imp:.6f}")
            if len(noise_candidates) > 10:
                print(f"      ... 他 {len(noise_candidates) - 10} 件")
    
    def calculate_correlation_matrix(self):
        """多重共線性（相関行列）の確認 - 数値変数のみ"""
        print(f"\n📊 相関行列計算中... (数値変数のみ, 閾値: {CORRELATION_THRESHOLD})")
        
        # KEEP_COLSとカテゴリ変数を除外した数値変数のみ
        numeric_for_corr = [
            col for col in self.numeric_cols
            if col not in KEEP_COLS and col not in CATEGORICAL_COLS
        ]
        
        if len(numeric_for_corr) < 2:
            print("   ⚠️ 相関チェック対象の数値変数が2つ未満です。スキップします。")
            self.high_corr_pairs = []
            self.skipped_cols_for_corr = self.categorical_cols
            return
        
        print(f"   対象列数: {len(numeric_for_corr)}")
        print(f"   スキップ列（カテゴリ）: {len(self.categorical_cols)}")
        
        # 相関行列計算
        corr_matrix = self.X_train[numeric_for_corr].corr().abs()
        
        # 上三角行列のみを取得（対角線と下三角を除外）
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # 高相関ペアを抽出
        high_corr_pairs = []
        for col in upper_tri.columns:
            for idx in upper_tri.index:
                val = upper_tri.loc[idx, col]
                if pd.notna(val) and val > CORRELATION_THRESHOLD:
                    high_corr_pairs.append({
                        'feature_1': idx,
                        'feature_2': col,
                        'correlation': val
                    })
        
        self.high_corr_pairs = sorted(high_corr_pairs, key=lambda x: -x['correlation'])
        self.skipped_cols_for_corr = self.categorical_cols
        
        print(f"   高相関ペア (>{CORRELATION_THRESHOLD}): {len(self.high_corr_pairs)} 件")
        
        if self.high_corr_pairs:
            print("   高相関ペアリスト:")
            for pair in self.high_corr_pairs[:10]:
                print(f"      - {pair['feature_1']} ⟷ {pair['feature_2']}: {pair['correlation']:.4f}")
            if len(self.high_corr_pairs) > 10:
                print(f"      ... 他 {len(self.high_corr_pairs) - 10} 件")
    
    def generate_report(self):
        """分析レポートをMarkdownで出力"""
        print("\n📄 レポート生成中...")
        
        report_path = os.path.join(self.output_dir, "feature_selection_report.md")
        
        # Permutation Importance上位・下位
        perm_top10 = self.perm_importance_df.head(10)
        perm_bottom10 = self.perm_importance_df.tail(10).sort_values('importance_mean')
        
        # レポート内容
        report_lines = [
            "# 特徴量選択レポート",
            "",
            f"**実行日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## 設定",
            "",
            "| 項目 | 値 |",
            "|------|-----|",
            f"| 評価指標 | LogLoss |",
            f"| Permutation Importance 繰り返し回数 | {N_REPEATS} |",
            f"| ノイズ判定閾値 | importance <= {NOISE_THRESHOLD} |",
            f"| 相関係数閾値 | > {CORRELATION_THRESHOLD} |",
            f"| 削除禁止リスト | {', '.join(KEEP_COLS)} |",
            "",
            "---",
            "",
            "## ベースラインモデル",
            "",
            f"- **LogLoss**: {self.baseline_logloss:.6f}",
            "",
            "---",
            "",
            "## Permutation Importance（順列重要度）",
            "",
            "検証データで各特徴量をシャッフルし、LogLossの悪化度合いを測定しました。",
            "値が大きいほど重要、0以下はシャッフルしても精度が変わらない（ノイズ）。",
            "",
            "### 上位10特徴量（重要）",
            "",
            "| 特徴量 | Importance (LogLoss悪化量) | Std |",
            "|--------|---------------------------|-----|",
        ]
        
        for _, row in perm_top10.iterrows():
            report_lines.append(
                f"| {row['feature']} | {row['importance_mean']:.6f} | {row['importance_std']:.6f} |"
            )
        
        report_lines.extend([
            "",
            "### 下位10特徴量（削除候補）",
            "",
            "| 特徴量 | Importance (LogLoss悪化量) | Std | 削除推奨 |",
            "|--------|---------------------------|-----|----------|",
        ])
        
        for _, row in perm_bottom10.iterrows():
            is_noise = row['importance_mean'] <= NOISE_THRESHOLD
            is_protected = row['feature'] in KEEP_COLS
            if is_protected:
                status = "❌ 保護対象"
            elif is_noise:
                status = "✅ 推奨"
            else:
                status = "-"
            report_lines.append(
                f"| {row['feature']} | {row['importance_mean']:.6f} | {row['importance_std']:.6f} | {status} |"
            )
        
        report_lines.extend([
            "",
            f"### ノイズ候補一覧（importance <= {NOISE_THRESHOLD}）",
            "",
        ])
        
        if self.noise_features:
            report_lines.append("> [!WARNING]")
            report_lines.append("> 以下の特徴量は削除を検討してください。")
            report_lines.append("")
            for feat in self.noise_features:
                imp = self.perm_importance_df[self.perm_importance_df['feature'] == feat]['importance_mean'].values[0]
                report_lines.append(f"- `{feat}`: {imp:.6f}")
        else:
            report_lines.append("ノイズ候補はありませんでした。")
        
        report_lines.extend([
            "",
            "---",
            "",
            "## 多重共線性（相関行列）",
            "",
            f"**対象**: 数値変数のみ（カテゴリ変数はPearson相関が無意味なため除外）",
            "",
            f"### スキップした列（カテゴリ変数）",
            "",
        ])
        
        if self.skipped_cols_for_corr:
            for col in self.skipped_cols_for_corr:
                report_lines.append(f"- `{col}`")
        else:
            report_lines.append("なし")
        
        report_lines.extend([
            "",
            f"### 高相関ペア（相関係数 > {CORRELATION_THRESHOLD}）",
            "",
        ])
        
        if self.high_corr_pairs:
            report_lines.append("> [!IMPORTANT]")
            report_lines.append("> 以下のペアは片方を削除することを検討してください。")
            report_lines.append("")
            report_lines.append("| 特徴量1 | 特徴量2 | 相関係数 |")
            report_lines.append("|---------|---------|----------|")
            for pair in self.high_corr_pairs:
                report_lines.append(
                    f"| {pair['feature_1']} | {pair['feature_2']} | {pair['correlation']:.4f} |"
                )
        else:
            report_lines.append("高相関ペアはありませんでした。")
        
        report_lines.extend([
            "",
            "---",
            "",
            "## 推奨アクション",
            "",
            "1. **ノイズ候補の削除**: 上記のノイズ候補リストから、ドメイン知識に基づいて削除する列を決定してください。",
            "2. **高相関ペアの整理**: 高相関ペアがある場合、片方を削除するか、主成分分析（PCA）等で統合することを検討してください。",
            "3. **再学習**: 特徴量を削除後、モデルを再学習してLogLossやAUCの変化を確認してください。",
            "",
        ])
        
        # ファイル出力
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"   レポート出力: {report_path}")
        
        # CSVも出力
        perm_csv_path = os.path.join(self.output_dir, "permutation_importance.csv")
        self.perm_importance_df.to_csv(perm_csv_path, index=False, encoding='utf-8-sig')
        print(f"   Permutation Importance CSV: {perm_csv_path}")
        
        fi_csv_path = os.path.join(self.output_dir, "feature_importance.csv")
        self.feature_importance_df.to_csv(fi_csv_path, index=False, encoding='utf-8-sig')
        print(f"   Feature Importance CSV: {fi_csv_path}")
        
        return report_path
    
    def run(self):
        """メイン実行"""
        start = datetime.now()
        
        self.load_data()
        self.train_baseline()
        self.calculate_permutation_importance()
        self.calculate_correlation_matrix()
        report_path = self.generate_report()
        
        elapsed = (datetime.now() - start).total_seconds()
        
        print("\n" + "=" * 60)
        print("✅ 完了！")
        print(f"   実行時間: {elapsed:.1f}秒")
        print(f"   レポート: {report_path}")
        print("=" * 60)
        
        return {
            'noise_features': self.noise_features,
            'high_corr_pairs': self.high_corr_pairs,
            'baseline_logloss': self.baseline_logloss,
            'report_path': report_path
        }


if __name__ == "__main__":
    selector = FeatureSelector()
    selector.run()
