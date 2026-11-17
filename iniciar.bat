@echo off
setlocal
cd /d "%~dp0"
title RXTorax

if not exist ".venv\Scripts\activate.bat" (
  echo Falta la instalacion. Ejecutar primero instalar.bat.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat

echo Abriendo RXTorax en el navegador. Para cerrarla, Ctrl+C en esta ventana.
streamlit run app\Inicio.py
pause
