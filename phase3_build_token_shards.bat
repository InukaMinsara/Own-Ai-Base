@echo off
setlocal

echo ============================================================
echo       OWN AI v8 PHASE 3 TOKEN SHARDS
echo ============================================================
echo.

python data\build_token_shards_v8.py

echo.
echo Phase 3 token shard build finished.
pause
endlocal
