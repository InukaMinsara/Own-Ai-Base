@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\\Scripts\\python.exe" (
    echo [ERROR] .venv not found.
    pause
    exit /b 1
)

call ".venv\\Scripts\\activate.bat"

echo ==============================================
echo            OWN AI v7 PRETRAINING
echo ==============================================

python data\\build_instruction_dataset_v7.py
if errorlevel 1 goto :error

python training\\train_v7.py
if errorlevel 1 goto :error

echo.
echo PRETRAINING COMPLETED
echo.
echo Next SFT stage:
echo python training\\train_sft_v7.py
echo.
pause
exit /b 0

:error
echo.
echo PRETRAINING FAILED
pause
exit /b 1
