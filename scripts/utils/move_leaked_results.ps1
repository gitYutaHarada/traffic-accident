# データリーク実験結果ファイル移動スクリプト
# PowerShellで実行

# 移動先ディレクトリを作成
New-Item -ItemType Directory -Path "data/data_leakage/results/analysis" -Force | Out-Null
New-Item -ItemType Directory -Path "data/data_leakage/results/experiments" -Force | Out-Null
New-Item -ItemType Directory -Path "data/data_leakage/results/visualizations" -Force | Out-Null

Write-Host "=== データリーク実験結果の移動 ===" -ForegroundColor Yellow

# Analysis ファイル（データリークあり）
$analysisFiles = @(
    "advanced_model_metrics.csv",           # LightGBM + SMOTE
    "feature_importance.csv",               # Random Forest (初期)
    "lgbm_weighted_feature_importance.csv", # LightGBM + Weight
    "lgbm_weighted_metrics.csv",            # LightGBM + Weight
    "optuna_best_params.json",              # Optuna
    "optuna_cv_metrics.csv",                # Optuna
    "optuna_feature_importance.csv",        # Optuna
    "optuna_study_history.csv",             # Optuna
    "refined_model_cv_metrics.csv",         # Random Forest Refined
    "upsampled_model_metrics.csv",          # Random Forest Upsampled
    "weighted_auc_score.txt",               # LightGBM + Weight
    "weighted_model_metrics.csv"            # LightGBM + Weight
)

Write-Host "`n[Analysis] データリーク実験結果を移動中..." -ForegroundColor Cyan
foreach ($file in $analysisFiles) {
    $source = "results/analysis/$file"
    $dest = "data/data_leakage/results/analysis/$file"
    if (Test-Path $source) {
        Move-Item -Path $source -Destination $dest -Force
        Write-Host "  ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ $file (見つかりません)" -ForegroundColor DarkGray
    }
}

# Experiments ファイル（データリークあり）
$experimentFiles = @(
    "advanced_model_experiment.md",    # LightGBM + SMOTE
    "optuna_tuning_experiment.md",     # Optuna
    "refined_model_experiment.md",     # Random Forest Refined
    "upsampling_experiment.md",        # Random Forest Upsampled
    "weighted_model_experiment.md"     # LightGBM + Weight
)

Write-Host "`n[Experiments] データリーク実験レポートを移動中..." -ForegroundColor Cyan
foreach ($file in $experimentFiles) {
    $source = "results/experiments/$file"
    $dest = "data/data_leakage/results/experiments/$file"
    if (Test-Path $source) {
        Move-Item -Path $source -Destination $dest -Force
        Write-Host "  ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ $file (見つかりません)" -ForegroundColor DarkGray
    }
}

# Visualizations ファイル（データリークあり）
$visualizationFiles = @(
    "confusion_matrix_advanced.png",              # LightGBM + SMOTE
    "confusion_matrix_upsampled.png",             # Random Forest Upsampled
    "confusion_matrix_weighted.png",              # LightGBM + Weight
    "feature_importance.png",                     # Random Forest (初期)
    "feature_importance_refined.png",             # Random Forest Refined
    "feature_importance_upsampled.png",           # Random Forest Upsampled
    "lgbm_weighted_confusion_matrix.png",         # LightGBM + Weight
    "lgbm_weighted_feature_importance.png",       # LightGBM + Weight
    "lgbm_weighted_pr_curve.png",                 # LightGBM + Weight
    "optuna_confusion_matrix.png",                # Optuna
    "optuna_feature_importance_top20.png",        # Optuna
    "optuna_pr_curve.png",                        # Optuna
    "pr_curve_advanced.png",                      # LightGBM + SMOTE
    "pr_curve_weighted.png"                       # LightGBM + Weight
)

Write-Host "`n[Visualizations] データリーク実験可視化を移動中..." -ForegroundColor Cyan
foreach ($file in $visualizationFiles) {
    $source = "results/visualizations/$file"
    $dest = "data/data_leakage/results/visualizations/$file"
    if (Test-Path $source) {
        Move-Item -Path $source -Destination $dest -Force
        Write-Host "  ✓ $file" -ForegroundColor Green
    } else {
        Write-Host "  ⚠ $file (見つかりません)" -ForegroundColor DarkGray
    }
}

Write-Host "`n=== 移動完了 ===" -ForegroundColor Yellow
Write-Host "`n移動先: data/data_leakage/results/" -ForegroundColor Green
