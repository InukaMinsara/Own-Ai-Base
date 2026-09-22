@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found.
    echo Create it first with:
    echo py -3.12 -m venv .venv
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo ==============================================
echo          OWN AI v8 FULL PIPELINE
echo ==============================================
echo.
echo 1. Build balanced SFT dataset
echo 2. Build token shards from local cache/cloud sync
echo 3. Streaming pretrain v8
echo 4. Instruction SFT
echo.

python data\build_v8_dataset.py
if errorlevel 1 goto :error

python data\build_token_shards_v8.py
if errorlevel 1 goto :error

python training\train_v8_streaming.py
if errorlevel 1 goto :error

python training\train_sft_v8.py
if errorlevel 1 goto :error

echo.
echo ==============================================
echo          OWN AI v8 COMPLETE
echo ==============================================
echo.
pause
exit /b 0

:error
echo.
echo ==============================================
echo OWN AI v8 PIPELINE FAILED
echo ==============================================
pause
exit /b 1
