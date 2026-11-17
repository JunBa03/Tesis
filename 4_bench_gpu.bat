@echo off
setlocal
cd /d "%~dp0"
title RXTorax - Benchmark del paso de GPU
if not exist ".venv\Scripts\activate.bat" (
  echo  No hay entorno virtual en %CD%
  goto :fin
)
call .venv\Scripts\activate.bat
python scripts\bench_gpu.py
:fin
echo.
pause
