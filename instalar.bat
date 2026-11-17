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
echo Descargando los modelos entrenados (unos 160 MB)...
for %%M in (E1_baseline_densenet121_uzeros_bce E2_uones E3_uignore E4_weighted_bce E5_focal E6_resnet50 E7_combinado) do (
  if not exist "models\%%M.pt" (
    echo   %%M.pt
    curl -fL --retry 2 -s -o "models\%%M.pt" "%URL_MODELOS%/%%M.pt"
    if errorlevel 1 (
      del "models\%%M.pt" >nul 2>&1
      echo   No se pudo descargar %%M.pt. Revisar la conexion o descargarlo a mano desde
      echo   %URL_MODELOS%
      goto error
    )
  )
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
