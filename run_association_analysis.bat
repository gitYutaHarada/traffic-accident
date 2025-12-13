@echo off
REM アソシエーション分析実行バッチファイル
REM 
REM 死亡事故に繋がる要因をアソシエーション分析で発見します

echo ========================================
echo アソシエーション分析実行
echo ========================================
echo.

REM 仮想環境をアクティベート
call .venv\Scripts\activate

echo [1/3] データ前処理を実行中...
python scripts\association_analysis\preprocess_for_association.py
if %errorlevel% neq 0 (
    echo エラー: データ前処理に失敗しました
    pause
    exit /b 1
)
echo.

echo [2/3] アソシエーション分析を実行中...
python scripts\association_analysis\run_association_analysis.py
if %errorlevel% neq 0 (
    echo エラー: アソシエーション分析に失敗しました
    pause
    exit /b 1
)
echo.

echo [3/3] 結果を可視化中...
python scripts\association_analysis\visualize_association_rules.py
if %errorlevel% neq 0 (
    echo エラー: 可視化に失敗しました
    pause
    exit /b 1
)
echo.

echo ========================================
echo 分析完了!
echo ========================================
echo.
echo 結果は results\association_analysis に保存されました
echo.

pause
