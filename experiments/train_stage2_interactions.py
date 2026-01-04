"""
Implementation Plan に基づく Stage 2 相互作用特徴量実験スクリプト

このスクリプトは既存の run_interaction_experiment.py のラッパーとして機能し、
Implementation Plan で定義された実験手順を明確に実行します。

使用方法:
    python experiments/train_stage2_interactions.py

目的:
    - Precision @ Recall 99% の改善
    - AUC, LogLoss の評価
    - Feature Importance による新特徴量の効果確認
"""

import os
import sys
import subprocess
from datetime import datetime

# プロジェクトルートをパスに追加
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# 設定
INPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "honhyo_clean_with_features.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "processed", "honhyo_with_interactions.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "experiments", "interaction_features")

def print_header(text: str):
    """セクションヘッダーを表示"""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")

def step1_create_features():
    """
    手順1: 特徴量生成
    scripts/features/create_interaction_features.py を実行し、
    相互作用特徴量を追加したデータセットを生成
    """
    print_header("Step 1: 相互作用特徴量の生成")
    
    script_path = os.path.join(BASE_DIR, "scripts", "features", "create_interaction_features.py")
    
    if not os.path.exists(script_path):
        print(f"ERROR: スクリプトが見つかりません: {script_path}")
        return False
    
    cmd = [
        sys.executable, script_path,
        "--input", INPUT_PATH,
        "--output", OUTPUT_PATH
    ]
    
    print(f"実行コマンド: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: 特徴量生成に失敗しました")
        print(result.stderr)
        return False
    
    print(result.stdout)
    print("✓ 特徴量生成完了")
    return True

def step2_train_model():
    """
    手順2: LightGBM モデル学習 (CV 5-fold)
    既存の run_interaction_experiment.py を使用
    """
    print_header("Step 2: LightGBM モデル学習 (5-fold CV)")
    
    # 既存の実験スクリプトを使用
    script_path = os.path.join(BASE_DIR, "experiments", "run_interaction_experiment.py")
    
    if not os.path.exists(script_path):
        print(f"ERROR: 実験スクリプトが見つかりません: {script_path}")
        return False
    
    print(f"実験スクリプト: {script_path}")
    print("このスクリプトは以下を実行します:")
    print("  - LightGBM 5-fold CV学習")
    print("  - Feature Importance (gain/split) 分析")
    print("  - SHAP分析")
    print("  - TP vs FP 分布比較")
    
    # 実行は別途行う
    print("\n[INFO] 以下のコマンドで実行してください:")
    print(f"  python {script_path}")
    
    return True

def step3_evaluate():
    """
    手順3: 結果評価
    評価指標を確認
    """
    print_header("Step 3: 評価指標の確認ポイント")
    
    print("重点指標:")
    print("  1. Precision @ Recall 99% (最重要)")
    print("  2. AUC (全体性能)")
    print("  3. LogLoss (確率キャリブレーション)")
    print()
    print("Feature Importance 確認ポイント:")
    print("  - 新特徴量が上位に来ているか")
    print("  - 実装済みの相互作用特徴量:")
    print("    * stop_sign_interaction (一時停止規制の相互作用)")
    print("    * speed_reg_diff (速度規制の差分)")
    print("    * road_shape_terrain (道路形状×地形)")
    print("    * signal_road_shape (信号機×道路形状)")
    print("    * night_road_condition (昼夜×路面状態)")
    print("    * is_safe_night_urban (安全な夜の市街地)")
    print("    * is_night_truck (夜のトラック)")
    
    return True

def main():
    """メイン実行フロー"""
    print("\n" + "#" * 60)
    print("#  Implementation Plan: 特徴量エンジニアリング強化 実験")
    print("#" * 60)
    print(f"\n開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"入力データ: {INPUT_PATH}")
    print(f"出力データ: {OUTPUT_PATH}")
    print(f"結果出力先: {RESULTS_DIR}")
    
    # 手順1: 特徴量生成
    if not step1_create_features():
        print("\nERROR: Step 1 failed. Aborting.")
        return
    
    # 手順2: モデル学習（情報表示のみ）
    step2_train_model()
    
    # 手順3: 評価（情報表示のみ）
    step3_evaluate()
    
    print("\n" + "#" * 60)
    print("#  実験準備完了")
    print("#" * 60)
    print(f"\n完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n次のステップ:")
    print("  1. python experiments/run_interaction_experiment.py を実行")
    print(f"  2. {RESULTS_DIR} 内の結果を確認")

if __name__ == "__main__":
    main()
