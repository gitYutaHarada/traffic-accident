"""
既存のstudy_history.csvからOptunaのSQLite studyを再構築するスクリプト

使用方法:
    python scripts/analysis/import_existing_trials.py
"""
import json
from pathlib import Path

import optuna
import pandas as pd

# パス設定
BASE_DIR = Path(__file__).resolve().parent.parent.parent
EXISTING_RESULTS_DIR = BASE_DIR / "results" / "tuning" / "tuning_20251209_011713"
STUDY_HISTORY_PATH = EXISTING_RESULTS_DIR / "study_history.csv"
BEST_PARAMS_PATH = EXISTING_RESULTS_DIR / "best_params.json"

# SQLiteストレージの設定
TUNING_DIR = BASE_DIR / "results" / "tuning"
STORAGE_PATH = TUNING_DIR / "lightgbm_tuning.db"
STORAGE_NAME = f"sqlite:///{STORAGE_PATH}"
STUDY_NAME = "lightgbm_pr_auc_optimization"

RANDOM_STATE = 42


def import_trials_from_csv():
    """既存のstudy_history.csvからtrialをインポート"""
    
    print("=" * 80)
    print("既存の試行データをSQLiteにインポート")
    print("=" * 80)
    
    # 既存のCSVを読み込み
    if not STUDY_HISTORY_PATH.exists():
        raise FileNotFoundError(f"study_history.csvが見つかりません: {STUDY_HISTORY_PATH}")
    
    print(f"\n[LOAD] {STUDY_HISTORY_PATH}")
    df = pd.read_csv(STUDY_HISTORY_PATH)
    print(f"[OK] {len(df)} 件の試行データを読み込みました")
    
    # 新しいstudyを作成（既存があれば上書き警告）
    print(f"\n[CREATE] Optuna studyを作成...")
    print(f"  Storage: {STORAGE_NAME}")
    print(f"  Study name: {STUDY_NAME}")
    
    # 既存のstudyがある場合は削除するか確認
    try:
        existing_study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=STORAGE_NAME
        )
        print(f"\n[WARNING] 既存のstudy '{STUDY_NAME}' が見つかりました")
        print(f"  既存の試行数: {len(existing_study.trials)}")
        print(f"  このstudyを削除して新しくインポートします...")
        optuna.delete_study(study_name=STUDY_NAME, storage=STORAGE_NAME)
        print(f"[OK] 既存のstudyを削除しました")
    except KeyError:
        print(f"[OK] 新規studyを作成します")
    
    # 新しいstudyを作成
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_NAME,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        load_if_exists=False
    )
    
    # パラメータ名のマッピング（CSVのカラム名 → Optunaのパラメータ名）
    param_columns = [
        "params_learning_rate",
        "params_num_leaves",
        "params_max_depth",
        "params_min_child_samples",
        "params_subsample",
        "params_colsample_bytree",
        "params_reg_alpha",
        "params_reg_lambda",
        "params_min_child_weight",
        "params_min_split_gain",
        "params_path_smooth",
        "params_scale_pos_weight",
    ]
    
    # 各trialをインポート
    print(f"\n[IMPORT] 試行データをインポート中...")
    imported_count = 0
    failed_count = 0
    
    for idx, row in df.iterrows():
        try:
            # stateの確認
            state_str = row.get("state", "COMPLETE")
            if state_str == "COMPLETE":
                state = optuna.trial.TrialState.COMPLETE
            elif state_str == "PRUNED":
                state = optuna.trial.TrialState.PRUNED
            else:
                state = optuna.trial.TrialState.COMPLETE
            
            # パラメータの抽出
            params = {}
            for col in param_columns:
                if col in df.columns:
                    param_name = col.replace("params_", "")
                    params[param_name] = row[col]
            
            # valueの取得
            value = row.get("value", None)
            if pd.isna(value):
                continue
            
            # trialを手動で作成
            trial = optuna.trial.create_trial(
                params=params,
                distributions={
                    "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.2, log=True),
                    "num_leaves": optuna.distributions.IntDistribution(31, 255),
                    "max_depth": optuna.distributions.IntDistribution(5, 15),
                    "min_child_samples": optuna.distributions.IntDistribution(20, 300),
                    "subsample": optuna.distributions.FloatDistribution(0.6, 1.0),
                    "colsample_bytree": optuna.distributions.FloatDistribution(0.6, 1.0),
                    "reg_alpha": optuna.distributions.FloatDistribution(0.0, 10.0),
                    "reg_lambda": optuna.distributions.FloatDistribution(0.0, 10.0),
                    "min_child_weight": optuna.distributions.FloatDistribution(0.001, 10.0, log=True),
                    "min_split_gain": optuna.distributions.FloatDistribution(0.0, 1.0),
                    "path_smooth": optuna.distributions.FloatDistribution(0.0, 10.0),
                    "scale_pos_weight": optuna.distributions.FloatDistribution(0.0, 1000.0),
                },
                values=[value],
                state=state,
            )
            
            # user_attributesの追加（評価指標）
            user_attr_cols = [
                "user_attrs_mean_Accuracy",
                "user_attrs_mean_Precision",
                "user_attrs_mean_Recall",
                "user_attrs_mean_F1",
                "user_attrs_mean_ROC_AUC",
                "user_attrs_mean_PR_AUC",
            ]
            for col in user_attr_cols:
                if col in df.columns and not pd.isna(row[col]):
                    attr_name = col.replace("user_attrs_", "")
                    trial.set_user_attr(attr_name, row[col])
            
            # studyにtrialを追加
            study.add_trial(trial)
            imported_count += 1
            
            if (imported_count % 20 == 0):
                print(f"  進捗: {imported_count}/{len(df)} 試行をインポート...")
        
        except Exception as e:
            print(f"  Warning: Trial {idx} のインポートに失敗: {e}")
            failed_count += 1
    
    print(f"\n[DONE] インポート完了")
    print(f"  成功: {imported_count} 試行")
    print(f"  失敗: {failed_count} 試行")
    print(f"  最良のPR-AUC: {study.best_value:.4f}")
    
    # 最良パラメータの表示
    print(f"\n[BEST] 最良パラメータ:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study


def verify_study():
    """インポートしたstudyを検証"""
    print("\n" + "=" * 80)
    print("インポートしたstudyの検証")
    print("=" * 80)
    
    try:
        study = optuna.load_study(
            study_name=STUDY_NAME,
            storage=STORAGE_NAME
        )
        
        n_trials = len(study.trials)
        n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        
        print(f"\n[OK] Studyの読み込み成功")
        print(f"  総試行数: {n_trials}")
        print(f"  完了試行数: {n_complete}")
        print(f"  最良のPR-AUC: {study.best_value:.4f}")
        print(f"\n[SUCCESS] これで継続実行スクリプトが使用できます！")
        print(f"\n次のコマンドで残り{200 - n_complete}試行を実行:")
        print(f"  python scripts/analysis/lightgbm_optuna_tuning_resume.py")
        
    except Exception as e:
        print(f"\n[ERROR] Studyの読み込みに失敗: {e}")


def main():
    """メイン処理"""
    try:
        # 既存の試行データをインポート
        study = import_trials_from_csv()
        
        # 検証
        verify_study()
        
    except Exception as e:
        print(f"\n[ERROR] エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
