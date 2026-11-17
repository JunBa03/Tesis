@echo off
setlocal
cd /d "%~dp0"
title RXTorax
if not exist ".venv\Scripts\activate.bat" (
  echo  Falta la puesta en marcha. Corre primero instalar_demo.bat
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
echo  Abriendo la demostracion. Para cerrarla: Ctrl+C en esta ventana.
echo.
streamlit run app\Inicio.py
pause
