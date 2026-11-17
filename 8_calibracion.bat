@echo off
setlocal
cd /d "%~dp0"
title RXTorax - Calibracion y curva riesgo-cobertura

echo ================================================================
echo  CALIBRACION + CURVA RIESGO-COBERTURA
echo ================================================================
echo.
echo  Corre DESPUES de que termine el entrenamiento (usa la GPU).
echo  Ajusta la temperatura, mide el ECE antes/despues y evalua la
echo  derivacion selectiva. Deja figuras en results\figuras\.
echo.

if not exist ".venv\Scripts\activate.bat" (
  echo  No hay entorno virtual .venv en esta carpeta.
  goto :fin
)
call .venv\Scripts\activate.bat
python -c "import matplotlib" 2>nul || python -m pip install matplotlib -q
python scripts\calibracion_riesgo.py %*

:fin
echo.
pause
