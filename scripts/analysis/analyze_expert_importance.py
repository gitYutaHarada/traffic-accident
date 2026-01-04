"""
Expert A vs Generalist 特徴量重要度比較スクリプト
===============================================
目的:
MoEモデルにおける「Expert A (Urban)」と「Generalist (Non-Urban)」が
それぞれどの特徴量を重視しているかを比較・分析する。

出力:
- 比較レポート (expert_importance_comparison.md)
- 重要度プロット (expert_importance_comparison.png)

実行方法:
    python scripts/analysis/analyze_expert_importance.py
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class ExpertImportanceAnalyzer:
    def __init__(
        self,
        moe_ckpt_dir="results/moe_stage2/checkpoints",
        output_dir="results/moe_stage2",
        n_folds=5
    ):
        self.moe_ckpt_dir = moe_ckpt_dir
        self.output_dir = output_dir
        self.n_folds = n_folds

    def extract_importance(self, expert_name):
        """指定されたExpertの重要度を抽出 (Fold平均)"""
        print(f"📊 {expert_name} の重要度を抽出中...")
        
        lgb_importances = []
        cat_importances = []
        feature_names = None

        for fold in range(self.n_folds):
            fold_dir = os.path.join(self.moe_ckpt_dir, f"{expert_name}_fold{fold}")
            lgb_path = os.path.join(fold_dir, "lgb_model.pkl")
            cat_path = os.path.join(fold_dir, "cat_model.pkl")

            # LightGBM
            if os.path.exists(lgb_path):
                lgb_model = joblib.load(lgb_path)
                imp = lgb_model.feature_importances_
                # 正規化 (sum=1)
                imp = imp / imp.sum()
                lgb_importances.append(imp)
                if feature_names is None:
                    feature_names = lgb_model.feature_name_
            
            # CatBoost
            if os.path.exists(cat_path):
                cat_model = joblib.load(cat_path)
                imp = cat_model.get_feature_importance()
                # 正規化
                imp = imp / imp.sum()
                cat_importances.append(imp)
                # CatBoostのfeature names取得 (LightGBMと順序が違う可能性に注意だが、今回はデータフレーム渡しなのでカラム名は一致するはず)
                # 安全のためLightGBMの名前を基準にする

        if not lgb_importances:
            print(f"⚠️ {expert_name} のモデルが見つかりません。")
            return None

        # Fold平均をとる
        avg_lgb_imp = np.mean(lgb_importances, axis=0)
        avg_cat_imp = np.mean(cat_importances, axis=0) if cat_importances else np.zeros_like(avg_lgb_imp)

        # 統合重要度 (LGBM + CatBoost) 
        # 重みづけはアンサンブル比率に合わせるのが筋だが、ここでは単純平均で傾向を見る
        combined_imp = (avg_lgb_imp + avg_cat_imp) / 2
        
        return pd.DataFrame({
            'feature': feature_names,
            f'{expert_name}_importance': combined_imp
        }).set_index('feature')

    def run(self):
        # 1. 重要度抽出
        df_urban = self.extract_importance("ExpertA_Urban")
        df_general = self.extract_importance("Generalist_NonUrban")

        if df_urban is None or df_general is None:
            return

        # 2. 結合
        df_merged = df_urban.join(df_general, how='outer').fillna(0)
        
        # 3. 分析指標計算
        # 差分: Expert A - Generalist (Aがどれだけより重視しているか)
        df_merged['diff'] = df_merged['ExpertA_Urban_importance'] - df_merged['Generalist_NonUrban_importance']
        # 比率: Expert A / Generalist (ゼロ除算回避)
        df_merged['ratio'] = (df_merged['ExpertA_Urban_importance'] + 1e-6) / (df_merged['Generalist_NonUrban_importance'] + 1e-6)

        # ソート: Expert Aで重要な順
        df_merged = df_merged.sort_values('ExpertA_Urban_importance', ascending=False)

        # 4. 可視化
        # 日本語フォント設定 (Windows向け)
        plt.rcParams['font.family'] = 'MS Gothic'
        
        plt.figure(figsize=(12, 10))
        
        # Top 20 Features (Expert A基準)
        top_features = df_merged.head(20).index
        plot_data = df_merged.loc[top_features, ['ExpertA_Urban_importance', 'Generalist_NonUrban_importance']]
        
        plot_data.plot(kind='barh', figsize=(10, 12), width=0.8)
        plt.title('Feature Importance Comparison: Urban(A) vs Generalist(B)')
        plt.xlabel('Normalized Importance')
        plt.gca().invert_yaxis() # 上位を上に
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "expert_feature_importance_comparison.png"))
        plt.close()

        # 5. レポート生成 (to_markdownはtabulate依存なのでto_stringで代用)
        report = "# Expert A vs Generalist 特徴量重要度比較レポート\n\n"
        
        # Expert A Top 10
        report += "## 🏙️ Expert A (Urban) が重視するトップ特徴量\n"
        report += df_merged[['ExpertA_Urban_importance', 'Generalist_NonUrban_importance', 'diff']].head(10).to_string()
        report += "\n\n"

        # Generalist Top 10 (参考)
        report += "## 🏞️ Generalist (Non-Urban) が重視するトップ特徴量\n"
        report += df_merged.sort_values('Generalist_NonUrban_importance', ascending=False)[['ExpertA_Urban_importance', 'Generalist_NonUrban_importance', 'diff']].head(10).to_string()
        report += "\n\n"

        # Expert A 特有の特徴量 (Diffが大きいもの)
        report += "## 🔍 Expert A 特有の注目ポイント (Diff上位)\n"
        report += "GeneralistよりもExpert Aで重要度が大きく上昇している特徴量です。\n"
        report += df_merged.sort_values('diff', ascending=False).head(10)[['ExpertA_Urban_importance', 'Generalist_NonUrban_importance', 'diff', 'ratio']].to_string()
        report += "\n\n"
        
        # 考察メモ
        report += "## 考察\n"
        report += "* **Diff上位の特徴量** に注目してください。これらがExpert Aが「市街地での誤検知」を防ぐために見ている鍵です。\n"
        report += "* もしここに「Stage 1 予測値 (prob_stage1)」以外の物理的な特徴量（道路幅、施設数など）が入っていれば、それを交差特徴量として強化するのが有効です。\n"

        output_path = os.path.join(self.output_dir, "expert_importance_comparison.md")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        print(f"\n✅ レポート保存完了: {output_path}")

if __name__ == "__main__":
    analyzer = ExpertImportanceAnalyzer()
    analyzer.run()
