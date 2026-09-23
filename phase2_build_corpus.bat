@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

if "%OWN_AI_V8_DATA_ROOT%"=="" set "OWN_AI_V8_DATA_ROOT=E:\My Drive [Inuka Minsara]\OwnAI_Dataset"

echo ==============================================
echo       OWN AI v8 PHASE 2 CLEAN CORPUS
echo ==============================================
echo Data root: %OWN_AI_V8_DATA_ROOT%
echo.

python data\build_corpus_v8.py

echo.
pause
