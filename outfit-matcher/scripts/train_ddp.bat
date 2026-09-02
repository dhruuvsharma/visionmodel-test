@echo off
REM Launch 2-GPU DDP training on the dual-RTX-5090 PC (Windows, gloo backend).
REM Usage: scripts\train_ddp.bat D:\data\shirts  [config path]
REM Prereqs: torch with CUDA support; manifests already generated (scripts\prepare_data.bat).

setlocal
set CONFIG=%2
if "%CONFIG%"=="" set CONFIG=configs\shirts.yaml
set DATA_DIR=%1

echo [launcher] 2-GPU DDP: config=%CONFIG% data=%DATA_DIR%
python scripts\launch_ddp.py --nproc 2 --config %CONFIG% --data-override %DATA_DIR%
if errorlevel 1 goto :error
echo Training finished. Check runs\ for checkpoints + metrics.
goto :eof

:error
echo TRAINING FAILED - see error above.
exit /b 1
