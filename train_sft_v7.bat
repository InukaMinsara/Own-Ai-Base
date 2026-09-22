@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\\Scripts\\python.exe" (
    echo [ERROR] .venv not found.
    pause
    exit /b 1
)

call ".venv\\Scripts\\activate.bat"
python training\\train_sft_v7.py

if errorlevel 1 (
    echo.
    echo SFT FAILED
    pause
    exit /b 1
)

echo.
echo SFT COMPLETED
pause
