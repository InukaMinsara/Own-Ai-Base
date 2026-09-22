@echo off
setlocal

cd /d "%~dp0.."

echo ================================================
echo OWN AI v8 - FULL LOCAL TRAINING PIPELINE
echo ================================================
echo.
echo Phase 1: Build dataset + pretrain
python data\build_v8_dataset.py
if errorlevel 1 exit /b 1

python training\train_v8.py
if errorlevel 1 exit /b 1

echo.
echo Phase 2: True instruction fine-tuning
python training\train_sft_v8.py
if errorlevel 1 exit /b 1

echo.
echo ================================================
echo OWN AI v8 TRAINING COMPLETE
echo ================================================
echo.
echo Checkpoints:
echo   checkpoints\own_ai_v8_pretrain_best.pt
echo   checkpoints\own_ai_v8_sft_best.pt
echo.
pause
