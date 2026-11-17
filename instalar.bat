@echo off
setlocal
cd /d "%~dp0"
title RXTorax - Instalacion

python --version >nul 2>&1
if errorlevel 1 (
  echo No se encontro Python 3. Instalarlo desde python.org.
  goto error
)

if not exist ".venv\Scripts\activate.bat" (
  echo Creando el entorno virtual...
  python -m venv .venv || goto error
)
call .venv\Scripts\activate.bat

echo Instalando dependencias...
python -m pip install --upgrade pip -q
rem PyTorch para CPU. Con placa NVIDIA puede usarse el indice cu124.
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu || goto error
python -m pip install -r requirements-demo.txt -q || goto error

if not exist "models\*.pt" (
  echo.
  echo No se encontraron modelos en models\. Copiarlos antes de iniciar la aplicacion.
)

echo.
echo Instalacion completa. Para abrir la aplicacion: iniciar.bat
pause
exit /b 0

:error
echo.
echo La instalacion no se completo. Revisar el mensaje anterior.
pause
exit /b 1
