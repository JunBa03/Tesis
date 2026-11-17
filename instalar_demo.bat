@echo off
setlocal
cd /d "%~dp0"
title Puesta en marcha de la demostracion

echo ================================================================
echo  PUESTA EN MARCHA  -  correr una sola vez en esta maquina
echo ================================================================
echo.

python --version 2>nul
if errorlevel 1 (
  echo  No se encontro Python. Instalalo desde python.org y volve a correr esto.
  goto :error
)

if not exist ".venv\Scripts\activate.bat" (
  echo [1/4] Creando entorno virtual...
  python -m venv .venv
  if errorlevel 1 goto :error
) else (
  echo [1/4] Entorno virtual ya existente.
)
call .venv\Scripts\activate.bat

echo.
echo [2/4] Instalando PyTorch para CPU (unos 200 MB)...
echo       Si esta maquina tiene placa NVIDIA y queres usarla, cambia
echo       la linea de abajo por el indice cu124.
python -m pip install --upgrade pip -q
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :error

echo.
echo [3/4] Resto de dependencias...
python -m pip install -r requirements-demo.txt -q
if errorlevel 1 goto :error

echo.
echo [4/4] Verificando que haya un modelo...
dir /b models\*.pt >nul 2>&1
if errorlevel 1 (
  echo.
  echo  ATENCION: no hay ningun modelo en models\
  echo  Exportarlos y copiarlos en models\:
  echo      python scripts\export_demo_model.py
  goto :error
)
for %%F in (models\*.pt) do echo       %%~nxF  (%%~zF bytes^)

echo.
echo ================================================================
echo  LISTO. Para abrir la demostracion:  demo.bat
echo ================================================================
pause
exit /b 0

:error
echo.
echo  FALLO. Revisa el mensaje de arriba.
pause
exit /b 1
