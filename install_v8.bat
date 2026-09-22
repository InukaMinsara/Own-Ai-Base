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
echo       OWN AI v8 DEPENDENCIES
echo ==============================================

python -m pip install -r requirements-v8.txt

if errorlevel 1 (
    echo.
    echo INSTALL FAILED
    pause
    exit /b 1
)

echo.
echo INSTALL COMPLETED
pause
