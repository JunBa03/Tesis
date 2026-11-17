@echo off
setlocal
cd /d "%~dp0"
title RXTorax - Instalacion

set "URL_MODELOS=https://github.com/JunBa03/Tesis/releases/download/v1.0"

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

if not exist "models" mkdir models
if not exist "models\E7_combinado.pt" (
  echo Descargando los modelos entrenados, unos 160 MB...
  curl -fL --retry 2 -o "%TEMP%\modelos_rxtorax.zip" "%URL_MODELOS%/modelos.zip"
  if errorlevel 1 (
    echo No se pudieron descargar los modelos. Revisar la conexion o descargar
    echo modelos.zip a mano desde %URL_MODELOS% y descomprimirlo en models\
    goto error
  )
  powershell -NoProfile -Command "Expand-Archive -Force '%TEMP%\modelos_rxtorax.zip' 'models'" || goto error
  del "%TEMP%\modelos_rxtorax.zip" >nul 2>&1
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
