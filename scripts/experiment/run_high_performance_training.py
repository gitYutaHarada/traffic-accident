"""
最高性能モデル構築パイプライン (チェックポイント・ログ機能付き)
=====================================================
学習データを2018-2022年の5年間に拡大し、最高性能モデルを構築する。

特徴:
- チェックポイント対応（途中から再開可能）
- ログファイル保存（長時間学習でも履歴を保持）
- Atomic Write（クラッシュ時のファイル破損を防止）
- PYTHONUNBUFFERED（リアルタイム出力）
- Intel Core Ultra 9 285K (24コア) / 64GB RAM 最大活用

実行方法:
    python scripts/experiment/run_high_performance_training.py

再開方法（途中で止まった場合）:
    同じコマンドを再実行するだけで、完了済みステップはスキップされます。

完全にやり直す場合:
    python scripts/experiment/run_high_performance_training.py --force-all
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# ========================================
# 定数
# ========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "high_performance_pipeline"
CHECKPOINT_FILE = CHECKPOINT_DIR / "pipeline_state.json"
LOG_DIR = CHECKPOINT_DIR / "logs"

# 新しいデータ分割設定
NEW_TRAIN_YEARS = "2018,2022"
NEW_VAL_YEARS = "2023,2023"
NEW_TEST_YEARS = "2024,2024"


class PipelineState:
    """パイプライン状態管理クラス (Atomic Write対応)"""
    
    def __init__(self, checkpoint_path: Path = CHECKPOINT_FILE):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
    
    def _load_state(self) -> Dict:
        """状態ファイルを読み込む（破損対策付き）"""
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                print("⚠️ 状態ファイルが破損しています。新規作成します。")
        return {
            "started_at": None,
            "completed_steps": [],
            "current_step": None,
            "last_updated": None,
        }
    
    def _save_state(self):
        """状態ファイルを保存 (Atomic Write)"""
        self.state["last_updated"] = datetime.now().isoformat()
        temp_path = self.checkpoint_path.with_suffix('.tmp')
        try:
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, ensure_ascii=False, indent=2)
            temp_path.replace(self.checkpoint_path)  # 原子的な置換
        except Exception as e:
            print(f"⚠️ 状態保存エラー: {e}")
    
    def is_step_completed(self, step_name: str) -> bool:
        """ステップが完了しているか確認"""
        return step_name in self.state["completed_steps"]
    
    def start_step(self, step_name: str):
        """ステップ開始を記録"""
        if self.state["started_at"] is None:
            self.state["started_at"] = datetime.now().isoformat()
        self.state["current_step"] = step_name
        self._save_state()
    
    def complete_step(self, step_name: str):
        """ステップ完了を記録"""
        if step_name not in self.state["completed_steps"]:
            self.state["completed_steps"].append(step_name)
        self.state["current_step"] = None
        self._save_state()
    
    def reset(self):
        """状態をリセット"""
        self.state = {
            "started_at": datetime.now().isoformat(),
            "completed_steps": [],
            "current_step": None,
            "last_updated": None,
        }
        self._save_state()


def run_step(
    state: PipelineState,
    step_name: str,
    command: List[str],
    description: str,
    cwd: Path = PROJECT_ROOT,
) -> bool:
    """
    1ステップを実行する（ログ保存・リアルタイム出力対応）
    
    Returns:
        True: 成功, False: 失敗
    """
    # 完了済みチェック
    if state.is_step_completed(step_name):
        print(f"✅ [{step_name}] 完了済み - スキップ")
        return True
    
    print("\n" + "=" * 70)
    print(f"🚀 [{step_name}] {description}")
    print(f"   コマンド: {' '.join(command)}")
    print("=" * 70)
    
    state.start_step(step_name)
    
    # ログファイルの準備
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"{step_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # 環境変数設定 (バッファリング無効化 + UTF-8エンコーディング強制)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"  # Windows cp932エラー回避
    
    try:
        with open(log_file, "w", encoding="utf-8", errors="replace") as f_log:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                encoding='utf-8',
                errors='replace',  # Windowsの非UTF-8文字を置換
                env=env,
            )
            
            # リアルタイムで出力を表示＆ログ保存
            for line in process.stdout:
                print(line, end='')
                f_log.write(line)
            
            process.wait()
        
        if process.returncode == 0:
            state.complete_step(step_name)
            print(f"\n✅ [{step_name}] 完了! (Log: {log_file.name})")
            return True
        else:
            print(f"\n❌ [{step_name}] 失敗 (exit code: {process.returncode})")
            print(f"   詳細はログを確認してください: {log_file}")
            return False
            
    except Exception as e:
        print(f"\n❌ [{step_name}] エラー: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="最高性能モデル構築パイプライン（チェックポイント・ログ機能付き）"
    )
    parser.add_argument(
        '--force-all', 
        action='store_true',
        help='全ステップを強制的に再実行（チェックポイントを無視）'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='実行せずにコマンドを表示するのみ'
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎯 最高性能モデル構築パイプライン")
    print(f"   Train: {NEW_TRAIN_YEARS} (5年間)")
    print(f"   Val:   {NEW_VAL_YEARS}")
    print(f"   Test:  {NEW_TEST_YEARS}")
    print(f"   チェックポイント: {CHECKPOINT_FILE}")
    print(f"   ログ: {LOG_DIR}")
    print("=" * 70)
    
    state = PipelineState()
    
    if args.force_all:
        print("\n⚠️ --force-all が指定されました。全履歴をリセットします。")
        state.reset()
    
    # ========================================
    # パイプライン定義
    # ========================================
    steps = [
        {
            "name": "step1_preprocess",
            "description": "データ前処理 (2018-2022 → Train, 2023 → Val, 2024 → Test)",
            "command": [
                sys.executable,
                "scripts/spatio_temporal/preprocess_spatio_temporal.py",
                "--train-years", NEW_TRAIN_YEARS,
                "--val-years", NEW_VAL_YEARS,
                "--test-years", NEW_TEST_YEARS,
            ],
        },
        {
            "name": "step2_single_stage",
            "description": "Single-Stage 4モデル学習 (LightGBM, CatBoost, MLP, TabNet)",
            "command": [
                sys.executable,
                "scripts/modeling/train_stage2_4models_spatiotemporal.py",
                "--force-retrain",
            ],
        },
        {
            "name": "step3_two_stage",
            "description": "Two-Stage 4モデル学習 (Hard Sample特化)",
            "command": [
                sys.executable,
                "scripts/modeling/train_stage2_4models_spatiotemporal_twostage.py",
                "--force-retrain",
            ],
        },
        {
            "name": "step4_stacking",
            "description": "Stage 3 Stackingメタモデル学習",
            "command": [
                sys.executable,
                "scripts/modeling/train_stage3_stacking.py",
            ],
        },
    ]
    
    if args.dry_run:
        print("\n📋 ドライラン: 以下のコマンドが実行されます")
        for step in steps:
            status = "✅ 完了済み" if state.is_step_completed(step["name"]) else "⏳ 未実行"
            print(f"\n{status} [{step['name']}] {step['description']}")
            print(f"   {' '.join(step['command'])}")
        return
    
    # ========================================
    # パイプライン実行
    # ========================================
    start_time = datetime.now()
    
    try:
        for step in steps:
            success = run_step(
                state=state,
                step_name=step["name"],
                command=step["command"],
                description=step["description"],
            )
            
            if not success:
                print("\n" + "=" * 70)
                print("❌ パイプラインが中断されました。")
                print("   再実行するには、同じコマンドを再度実行してください。")
                print("   完了済みステップはスキップされます。")
                print("=" * 70)
                sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n🛑 ユーザーにより中断されました。")
        print("   再開するには、同じコマンドを再度実行してください。")
        sys.exit(130)
    
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    
    print("\n" + "=" * 70)
    print("🎉 パイプライン完了!")
    print(f"   総実行時間: {elapsed:.1f} 分")
    print(f"   結果: {CHECKPOINT_DIR}")
    print(f"   ログ: {LOG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
