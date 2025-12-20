@echo off
echo ============================================
echo    BLUEGUARDIAN LIVE TRADING BOT (DRY-RUN)
echo ============================================
echo.
echo Starting in DRY-RUN mode - NO orders will be executed!
echo   - Daily DD Limit: 4%% (internal: 3.2%%)
echo   - Total DD Limit: 8%% (internal: 6.4%%)
echo   - Max Unrealised Loss: 2%% (internal: 1.6%%)
echo   - Profit Target: 8%%
echo.

REM Activate virtual environment if it exists
if exist "..\venv\Scripts\activate.bat" (
    call "..\venv\Scripts\activate.bat"
) else if exist "venv\Scripts\activate.bat" (
    call "venv\Scripts\activate.bat"
)

python live_mt5_eval_runner.py --mode eval --dry-run

pause
