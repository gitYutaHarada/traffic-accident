import argparse
import lightgbm as lgb
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from utils.data_loader import DataLoader
from utils.experiment_runner import ExperimentRunner

def main():
    parser = argparse.ArgumentParser(description='交通死亡事故予測モデルの実験実行スクリプト')
    parser.add_argument('--method', type=str, default='weighted', choices=['weighted', 'smote'], help='実験手法: weighted (重み付け) または smote (オーバーサンプリング)')
    parser.add_argument('--force_reload', action='store_true', help='キャッシュを使用せず生データを再読み込みする')
    args = parser.parse_args()

    print(f"🚀 実験開始: Method={args.method}, ForceReload={args.force_reload}")

    # データ読み込み
    # スクリプトの実行場所に関わらずパスを解決できるように調整（簡易的）
    # 基本的にプロジェクトルートから実行することを想定: python scripts/run_experiment.py
    raw_data_path = 'data/raw/honhyo_all_shishasuu_binary.csv'
    
    loader = DataLoader(raw_data_path)
    X, y = loader.load_data(force_reload=args.force_reload)

    experiment_name = f"lgbm_{args.method}"
    runner = ExperimentRunner(X, y, experiment_name=experiment_name)
    
    if args.method == 'weighted':
        # 重み付けの計算
        pos_count = y.sum()
        neg_count = len(y) - pos_count
        scale_pos_weight = neg_count / pos_count
        print(f"⚖️ Calculated scale_pos_weight: {scale_pos_weight:.2f}")

        def model_factory():
            return lgb.LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                verbosity=-1,
                boosting_type='gbdt',
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight
            )
            
    elif args.method == 'smote':
         def model_factory():
            return Pipeline([
                ('smote', SMOTE(random_state=42)),
                ('lgbm', lgb.LGBMClassifier(
                    objective='binary',
                    metric='binary_logloss',
                    verbosity=-1,
                    boosting_type='gbdt',
                    n_estimators=1000,
                    learning_rate=0.05,
                    num_leaves=31,
                    random_state=42,
                    n_jobs=-1
                ))
            ])

    runner.run_cv(model_factory)
    print("✨ 全工程完了")

if __name__ == "__main__":
    main()
