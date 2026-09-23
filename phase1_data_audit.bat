@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

if "%OWN_AI_V8_DATA_DIRS%"=="" set "OWN_AI_V8_DATA_DIRS=E:\My Drive [Inuka Minsara]\OwnAI_Dataset"

echo ==============================================
echo       OWN AI v8 PHASE 1 DATA LAKE AUDIT
echo ==============================================
echo Data roots: %OWN_AI_V8_DATA_DIRS%
echo.

python data\prepare_lake_v8.py

echo.
pause