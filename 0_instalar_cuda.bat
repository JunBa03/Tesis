@echo off
setlocal
cd /d "%~dp0"
title RXTorax - Instalar PyTorch con CUDA

echo ================================================================
echo  INSTALAR PYTORCH CON SOPORTE CUDA
echo ================================================================
echo.
echo  El torch que hay instalado es la build solo-CPU: entrena en el
echo  procesador, no en la 3060 Ti. Este script lo reemplaza.
echo.
echo  Descarga unos 2.5 GB. Puede tardar varios minutos.
echo.
pause

if not exist ".venv\Scripts\activate.bat" (
  echo Creando entorno virtual...
  python -m venv .venv || goto :error
)
call .venv\Scripts\activate.bat

echo.
echo === Driver NVIDIA ===
nvidia-smi
if errorlevel 1 (
  echo.
  echo  No se pudo ejecutar nvidia-smi. Falta el driver NVIDIA o no esta en el PATH.
  echo  Instalalo desde https://www.nvidia.com/Download/index.aspx antes de seguir.
  goto :error
)

echo.
echo === Quitando la build solo-CPU ===
python -m pip uninstall -y torch torchvision torchaudio

echo.
echo === Instalando la build CUDA 12.4 ===
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 goto :error

echo.
echo === Resto de dependencias ===
python -m pip install -r requirements.txt -q
if errorlevel 1 goto :error

echo.
echo === Comprobacion ===
python scripts\check_env.py
if errorlevel 1 goto :error

echo.
echo ================================================================
echo  LISTO. Ahora corre  1_verificar.bat
echo ================================================================
pause
exit /b 0

:error
echo.
echo ================================================================
echo  FALLO. Copiame el error completo.
echo.
echo  Si el problema es la version de CUDA, fijate cual te corresponde
echo  en https://pytorch.org/get-started/locally/ y cambia cu124 por
echo  esa en la linea del pip install de este archivo.
echo ================================================================
pause
exit /b 1
