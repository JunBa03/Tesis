@echo off
setlocal
cd /d "%~dp0"
title RXTorax - Demostracion

if not exist ".venv\Scripts\activate.bat" (
  echo  No hay entorno virtual en %CD%
  goto :fin
)
call .venv\Scripts\activate.bat

python -c "import streamlit" 2>nul
if errorlevel 1 (
  echo Instalando streamlit, una sola vez...
  python -m pip install streamlit -q
)

dir /b checkpoints\*.pt >nul 2>&1
if errorlevel 1 (
  echo.
  echo  Todavia no hay ningun modelo entrenado en checkpoints\
  echo  Esperá a que 2_entrenar_todo.bat termine al menos E1.
  goto :fin
)

echo.
echo  Abriendo la demostracion en el navegador.
echo  Para cerrarla: Ctrl+C en esta ventana.
echo.
rem App multipagina nueva. Plan B minimo: streamlit run app\demo.py
streamlit run app\Inicio.py

:fin
echo.
pause
