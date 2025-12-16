@echo off
REM 3モデル比較検証を honhyo_clean_road_type.csv で実行
echo ========================================
echo 3モデル比較検証を開始します
echo データ: honhyo_clean_road_type.csv
echo ========================================
echo.
echo 比較対象:
echo  1. ロジスティック回帰 (線形モデル)
echo  2. Random Forest (バギング)
echo  3. LightGBM (ブースティング)
echo.
echo 推定実行時間: 約2-3時間
echo ========================================
echo.

python scripts/model_comparison/compare_three_models.py

echo.
echo ========================================
echo 完了しました!
echo 結果は outputs/results/model_comparison に保存されました
echo ========================================
pause
