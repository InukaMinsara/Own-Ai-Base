@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo ==============================================
echo         OWN AI v8 SMOKE TEST
echo ==============================================
echo.
echo This runs only 25 pretraining steps.
echo It is for checking the full pipeline before a long run.
echo.

set OWN_AI_V8_STEPS=25
set OWN_AI_V8_EVAL_INTERVAL=10
set OWN_AI_V8_SFT_EPOCHS=1

python data\build_v8_dataset.py
if errorlevel 1 goto :error

python training\train_v8.py
if errorlevel 1 goto :error

echo.
echo Smoke pretraining completed.
echo Run training\train_sft_v8.py manually after checking the output.
pause
exit /b 0

:error
echo.
echo SMOKE TEST FAILED
pause
exit /b 1
