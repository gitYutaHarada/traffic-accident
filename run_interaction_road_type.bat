@echo off
REM 多交互作用検証を honhyo_clean_road_type.csv で実行
echo ========================================
echo 多交互作用検証を開始します
echo データ: honhyo_clean_road_type.csv
echo ========================================

python scripts/feature_engineering/run_interaction_analysis.py --data-path data/processed/honhyo_clean_road_type.csv --output-dir outputs/results/interaction_features_road_type

echo.
echo ========================================
echo 完了しました!
echo 結果は outputs/results/interaction_features_road_type に保存されました
echo ========================================
pause
