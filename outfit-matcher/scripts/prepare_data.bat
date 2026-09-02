@echo off
REM End-to-end prep on the training PC: generate manifests from a render directory.
REM Usage: scripts\prepare_data.bat D:\renders\shirts D:\data\shirts

setlocal
set DATA_ROOT=%1
set OUT_DIR=%2

python -m outfit_matcher.data.prepare_data --data-root %DATA_ROOT% --out-dir %OUT_DIR% --val-fraction 0.05 --seed 42
if errorlevel 1 goto :error
echo Manifests ready in %OUT_DIR%
goto :eof

:error
echo PREPARE FAILED - check that %DATA_ROOT% contains one folder per garment with front/back/left/right images.
exit /b 1
