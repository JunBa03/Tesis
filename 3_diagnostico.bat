@echo off
setlocal
cd /d "%~dp0"
title RXTorax - Diagnostico de carga de imagenes

echo ================================================================
echo  DIAGNOSTICO: por que la carga de imagenes va lenta
echo ================================================================
echo.
echo  La calibracion dio 29 img/s cuando la GPU puede con unas 400.
echo  Esto separa disco, decodificacion y transformaciones para saber
echo  cual de las tres es la lenta. Tarda menos de un minuto.
echo.

if not exist ".venv\Scripts\activate.bat" (
  echo  No hay entorno virtual en esta carpeta.
  echo  Carpeta actual: %CD%
  goto :fin
)
call .venv\Scripts\activate.bat

python scripts\bench_io.py --n 200

:fin
echo.
pause
