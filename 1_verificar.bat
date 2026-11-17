@echo off
setlocal
cd /d "%~dp0"
title RXTorax - Verificacion previa al entrenamiento

echo ================================================================
echo  VERIFICACION PREVIA  -  no entrena nada en serio, tarda ~5 min
echo ================================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
  echo  No hay entorno virtual. Corre primero 0_instalar_cuda.bat
  goto :error
)
echo [1/6] Activando entorno virtual...
call .venv\Scripts\activate.bat
echo.

echo [2/6] Comprobando GPU y build de PyTorch...
python scripts\check_env.py
if errorlevel 1 goto :error
echo.

echo [3/6] Logica central (politicas de incertidumbre, split, metricas)...
python scripts\verify_core.py
if errorlevel 1 goto :error
echo.

echo [4/6] Tests unitarios...
python -m pytest tests\ -q
if errorlevel 1 goto :error
echo.

echo [5/6] Pipeline completo sobre datos sinteticos.
echo       Valida el codigo nuevo: channels_last, batch 64, prefetch y --resume.
if exist "data_synthetic\" goto :hay_sinteticos
python scripts\make_synthetic_data.py --out data_synthetic --patients 40
if errorlevel 1 goto :error
:hay_sinteticos
echo       - corrida inicial de 1 epoca
python scripts\train.py --config configs\e1_baseline.yaml --skip-test ^
  --set data.root=data_synthetic train.epochs=1 data.num_workers=0 ^
        model.pretrained=false data.internal_val_fraction=0.25 ^
        output.results_dir=results_smoke output.checkpoint_dir=checkpoints_smoke
if errorlevel 1 goto :error
echo       - reanudacion: deberia arrancar en la epoca 2, no en la 1
python scripts\train.py --config configs\e1_baseline.yaml --resume --skip-test ^
  --set data.root=data_synthetic train.epochs=2 data.num_workers=0 ^
        model.pretrained=false data.internal_val_fraction=0.25 ^
        output.results_dir=results_smoke output.checkpoint_dir=checkpoints_smoke
if errorlevel 1 goto :error
echo.

echo [6/6] CALIBRACION sobre el dataset real: 300 batches de 64.
echo       Anota los segundos que reporta la epoca 1.
echo.
python scripts\train.py --config configs\e1_baseline.yaml --skip-test ^
  --set train.epochs=1 train.max_train_batches=300 train.max_eval_batches=50 ^
        output.results_dir=results_calib output.checkpoint_dir=checkpoints_calib
if errorlevel 1 goto :error

echo.
echo ================================================================
echo  LISTO. Todo verificado.
echo.
echo  Tomá los segundos de la epoca 1 y multiplicalos por 9,4:
echo  ese es el tiempo real de una epoca completa.
echo.
echo     menos de 15 min  ..  vas comodo, se puede subir a 12 epocas
echo     entre 15 y 30    ..  dejar 8 epocas, no tocar nada
echo     mas de 30 min    ..  revisar si el dataset esta en un SSD
echo.
echo  Cuando decidas, corre  2_entrenar_todo.bat
echo ================================================================
pause
exit /b 0

:error
echo.
echo ================================================================
echo  FALLO en el paso anterior. No sigas hasta resolverlo.
echo  Copiá el error completo y pasamelo.
echo ================================================================
pause
exit /b 1
