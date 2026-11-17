@echo off
setlocal
cd /d "%~dp0"
title RXTorax - Entrenar modelo aplicado E7 (320px)

echo ================================================================
echo  ENTRENAR MODELO APLICADO E7  -  ResNet-50 + U-Ones a 320px
echo ================================================================
echo.
echo  Es el modelo mas preciso para la demo. NO reemplaza la ablacion
echo  (E1-E6 a 224px), que se mantiene como estudio controlado.
echo.
echo  A 320px y batch 32 tarda mas que los de 224 (aprox 3-4 horas en
echo  una 3060 Ti). Se puede cortar y retomar: vuelve a correr este .bat.
echo.

if not exist ".venv\Scripts\activate.bat" (
  echo  No hay entorno virtual .venv en esta carpeta.
  goto :fin
)
call .venv\Scripts\activate.bat

python -c "import torch; print('CUDA disponible:', torch.cuda.is_available())"

echo.
python scripts\train.py --config configs\e7_combinado.yaml --resume
if errorlevel 1 goto :error

echo.
echo ================================================================
echo  LISTO. Ahora medi si mejora y actualiza la demo:
echo      python scripts\evaluar_mejoras.py
echo  (o el .bat 6_evaluar_mejoras.bat)
echo ================================================================
goto :fin

:error
echo.
echo  FALLO el entrenamiento. Revisa el mensaje de arriba.

:fin
echo.
pause
