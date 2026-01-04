"""
モデル詳細分析 一括実行スクリプト

使用方法:
    python scripts/analysis/model_deep_dive/run_all_analysis.py
"""

import sys, io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import subprocess
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
OUTPUT_BASE = Path("results/analysis/model_deep_dive")


def run_script(name):
    path = SCRIPT_DIR / name
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        )
        if result.returncode == 0:
            print(result.stdout)
            print(f"[OK] {name}")
            return True
        else:
            print(f"[FAIL] {name}")
            print(f"Error: {result.stderr[:500]}")
            return False
    except Exception as e:
        print(f"[ERROR] {name}: {e}")
        return False


def generate_master_report():
    print(f"\n{'='*60}")
    print("Generating master report...")
    print(f"{'='*60}")
    
    report = f"""# Model Deep Dive Analysis - Master Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Analysis Summary

This report covers 4 analysis areas for the Stage 3 Stacking model:

1. **Geospatial Error Analysis** - Where does the model make mistakes?
2. **Model Disagreement Analysis** - How do base models differ?
3. **SHAP Feature Analysis** - Which features matter most?
4. **Threshold Simulation** - What threshold to use in production?

## Output Directories

- `geospatial/` - Error distribution maps
- `disagreement/` - Model comparison reports
- `shap/` - Feature importance plots
- `threshold/` - Threshold optimization curves
"""
    
    output_path = OUTPUT_BASE / "master_report.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Model Deep Dive Analysis - Batch Execution")
    print("=" * 60)
    
    scripts = [
        "01_error_geospatial_analysis.py",
        "02_disagreement_analysis.py",
        "03_shap_analysis.py",
        "04_threshold_simulation.py",
    ]
    
    results = {}
    for s in scripts:
        results[s] = run_script(s)
    
    generate_master_report()
    
    print("\n" + "=" * 60)
    print("Execution Summary")
    print("=" * 60)
    for s, ok in results.items():
        status = "[OK]" if ok else "[FAIL]"
        print(f"   {status} {s}")
    
    success = sum(results.values())
    print(f"\n   Success: {success}/{len(results)}")
    print(f"   Output: {OUTPUT_BASE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
