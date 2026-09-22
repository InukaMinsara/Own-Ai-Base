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
echo              OWN AI WEB CHAT
echo ==============================================
echo.

call ".venv\Scripts\activate.bat"
python chat\server.py

echo.
echo Own AI stopped.
pause
