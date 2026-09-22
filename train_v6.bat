@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] .venv not found.
    echo Create it first with: py -3.12 -m venv .venv
    pause
    exit /b 1
)

echo ==============================================
echo            OWN AI v6 TRAINING
echo ==============================================
echo.

call ".venv\Scripts\activate.bat"

python data\build_instructions.py
if errorlevel 1 goto :error

python training\train_sft_v6.py
if errorlevel 1 goto :error

echo.
echo ==============================================
echo TRAINING COMPLETED
echo ==============================================
pause
exit /b 0

:error
echo.
echo ==============================================
echo TRAINING FAILED
echo ==============================================
pause
exit /b 1
