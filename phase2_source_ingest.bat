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

echo ============================================================
echo           OWN AI v8 PHASE 2 SOURCE INGESTION
echo ============================================================
echo.
echo Available approved sources:
echo   siwiki
echo   enwiki
echo   enwiktionary
echo.
echo Example:
echo   phase2_source_ingest.bat siwiki
echo   phase2_source_ingest.bat enwiki
echo   phase2_source_ingest.bat siwiki enwiki
echo.

python data\download_sources_v8.py %*

echo.
pause
