@echo off
setlocal
cd /d "%~dp0"
title RXTorax - Evaluar mejoras de ensemble

echo ================================================================
echo  MEDIR MEJORAS SIN REENTRENAR (ensemble ponderado, por patologia, TTA)
echo ================================================================
echo.
echo  Corre las 6 configuraciones sobre el conjunto oficial y elige la
echo  mejor combinacion. Con GPU tarda unos minutos; en CPU, bastante mas.
echo.

if not exist ".venv\Scripts\activate.bat" (
  echo  No hay entorno virtual .venv en esta carpeta.
  echo  Carpeta actual: %CD%
  goto :fin
)
call .venv\Scripts\activate.bat

rem Verificar que este entorno tenga CUDA; si no, avisar (igual funciona, mas lento).
python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"

echo.
python scripts\evaluar_mejoras.py %*

echo.
echo  Si quedo results\ensemble_config.json, la demo ya usa la mejor combinacion.

:fin
echo.
pause
