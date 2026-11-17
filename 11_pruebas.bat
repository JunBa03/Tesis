@echo off
setlocal
cd /d "%~dp0"
title RXTorax - Pruebas

REM ======================================================================
REM  Pruebas del sistema. Tarda ~5 minutos.
REM   1  Pruebas unitarias (pytest) del pipeline
REM   2  Robustez frente a formatos de otros equipos
REM   3  Radiografias externas reales (carpeta ..\radiografias_externas)
REM ======================================================================

if not exist ".venv\Scripts\activate.bat" (
  echo No hay entorno virtual en %CD%
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
if not exist "results\pruebas" mkdir "results\pruebas"

echo.
echo === 1/3 Pruebas unitarias ===
python -m pytest tests -v --junitxml=results\pruebas\pytest.xml > results\pruebas\pytest.txt 2>&1
type results\pruebas\pytest.txt

echo.
echo === 2/3 Robustez frente a formatos externos ===
python scripts\prueba_robustez.py --n 10

echo.
echo === 3/3 Radiografias externas reales ===
python scripts\probar_externas.py --carpeta ..\radiografias_externas --salida results\pruebas\externas.html

echo.
echo ================================================================
echo  TERMINADO. Resultados en results\pruebas\  - avisale a Claude.
echo ================================================================
pause
