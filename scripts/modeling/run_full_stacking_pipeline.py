"""
Full Stacking Pipeline (統合実行スクリプト)
==========================================
以下の3ステップを順次実行:
1. Single-Stage Spatio-Temporal 4-Model Ensemble (OOF再生成)
2. Two-Stage Spatio-Temporal 4-Model Ensemble (OOF再生成)
3. Stage 3 Stacking Meta-Model

実行方法:
    python scripts/modeling/run_full_stacking_pipeline.py

所要時間: 約60-70分
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# プロジェクトルートをPATHに追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_single_stage():
    """Step 1: Single-Stage Spatio-Temporal 4-Model Ensemble"""
    print("\n" + "=" * 80)
    print("🚀 Step 1/3: Single-Stage Spatio-Temporal Ensemble")
    print("=" * 80)
    
    from scripts.modeling.train_stage2_4models_spatiotemporal import SpatioTemporalEnsemble
    
    ensemble = SpatioTemporalEnsemble(
        data_dir="data/spatio_temporal",
        output_dir="results/spatio_temporal_ensemble",
        n_folds=5,
        random_state=42,
    )
    ensemble.run()
    
    print("✅ Step 1 完了: Single-Stage OOF再生成")


def run_two_stage():
    """Step 2: Two-Stage Spatio-Temporal 4-Model Ensemble"""
    print("\n" + "=" * 80)
    print("🚀 Step 2/3: Two-Stage Spatio-Temporal Ensemble")
    print("=" * 80)
    
    from scripts.modeling.train_stage2_4models_spatiotemporal_twostage import TwoStageSpatioTemporalEnsemble
    
    ensemble = TwoStageSpatioTemporalEnsemble(
        spatio_temporal_dir="data/spatio_temporal",
        stage1_oof_path="data/processed/stage1_oof_predictions.csv",
        stage1_test_path="data/processed/stage1_test_predictions.csv",
        output_dir="results/twostage_spatiotemporal_ensemble",
        n_folds=5,
        random_state=42,
        stage1_recall_target=0.98,
    )
    ensemble.run()
    
    print("✅ Step 2 完了: Two-Stage OOF再生成")


def run_stacking():
    """Step 3: Stacking Meta-Model"""
    print("\n" + "=" * 80)
    print("🚀 Step 3/3: Stacking Meta-Model")
    print("=" * 80)
    
    from scripts.modeling.train_stage3_stacking import StackingMetaModel
    
    stacking = StackingMetaModel(
        output_dir=Path("results/stage3_stacking"),
        n_folds=5,
        random_state=42,
    )
    stacking.run()
    
    print("✅ Step 3 完了: Stacking")


def main():
    """メインエントリポイント"""
    start_time = datetime.now()
    
    print("=" * 80)
    print("🏁 Full Stacking Pipeline 開始")
    print(f"   開始時刻: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    try:
        # Step 1: Single-Stage
        step1_start = time.time()
        run_single_stage()
        step1_time = (time.time() - step1_start) / 60
        print(f"   ⏱️ Step 1 所要時間: {step1_time:.1f}分")
        
        # Step 2: Two-Stage
        step2_start = time.time()
        run_two_stage()
        step2_time = (time.time() - step2_start) / 60
        print(f"   ⏱️ Step 2 所要時間: {step2_time:.1f}分")
        
        # Step 3: Stacking
        step3_start = time.time()
        run_stacking()
        step3_time = (time.time() - step3_start) / 60
        print(f"   ⏱️ Step 3 所要時間: {step3_time:.1f}分")
        
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 完了サマリー
    total_time = (datetime.now() - start_time).total_seconds() / 60
    
    print("\n" + "=" * 80)
    print("🎉 Full Stacking Pipeline 完了!")
    print("=" * 80)
    print(f"   合計所要時間: {total_time:.1f}分")
    print(f"   Step 1 (Single-Stage): {step1_time:.1f}分")
    print(f"   Step 2 (Two-Stage):    {step2_time:.1f}分")
    print(f"   Step 3 (Stacking):     {step3_time:.1f}分")
    print("\n📁 出力ファイル:")
    print("   - results/spatio_temporal_ensemble/oof_predictions.csv")
    print("   - results/twostage_spatiotemporal_ensemble/oof_predictions.csv")
    print("   - results/stage3_stacking/final_submission_stacking.csv")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
