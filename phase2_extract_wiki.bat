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

set "SOURCE_DIR=%OWN_AI_V8_DATA_ROOT%\open_license\wikimedia"

echo ============================================================
echo         OWN AI v8 PHASE 2 WIKIMEDIA EXTRACTION
echo ============================================================
echo Data root: %OWN_AI_V8_DATA_ROOT%
echo.

if /I "%~1"=="siwiki" goto :siwiki
if /I "%~1"=="enwiki" goto :enwiki
if /I "%~1"=="enwiktionary" goto :enwiktionary

echo Usage:
echo   phase2_extract_wiki.bat siwiki
echo   phase2_extract_wiki.bat enwiki
echo   phase2_extract_wiki.bat enwiktionary
exit /b 1

:siwiki
python data\extract_wikimedia_v8.py "%SOURCE_DIR%\siwiki\siwiki-latest-pages-articles-multistream.xml.bz2" "%SOURCE_DIR%\siwiki\extracted"
goto :done

:enwiki
python data\extract_wikimedia_v8.py "%SOURCE_DIR%\enwiki\enwiki-latest-pages-articles-multistream.xml.bz2" "%SOURCE_DIR%\enwiki\extracted"
goto :done

:enwiktionary
python data\extract_wikimedia_v8.py "%SOURCE_DIR%\enwiktionary\enwiktionary-latest-pages-articles-multistream.xml.bz2" "%SOURCE_DIR%\enwiktionary\extracted"
goto :done

:done
echo.
echo Extraction command finished.
pause
