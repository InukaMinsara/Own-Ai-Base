@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found.
    echo Create it first:
    echo py -3.12 -m venv .venv
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo ==============================================
echo      OWN AI v8 MASS-DATA STREAMING TRAINING
echo ==============================================
echo.
echo Data roots:
echo   data\cache
echo   data\raw
echo   data\knowledge
echo.
echo The training pipeline uses token shards and memory maps.
echo It does NOT load the complete dataset into RAM.
echo.

python data\build_token_shards_v8.py
if errorlevel 1 goto :error

python training\train_v8_streaming.py
if errorlevel 1 goto :error

echo.
echo STREAMING PRETRAINING COMPLETED
echo.
pause
exit /b 0

:error
echo.
echo STREAMING TRAINING FAILED
pause
exit /b 1
