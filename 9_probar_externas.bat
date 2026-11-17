@echo off
setlocal
cd /d "%~dp0"
title Prueba con radiografias externas - RXTorax
if not exist ".venv\Scripts\activate.bat" (
  echo  Falta la puesta en marcha. Corre primero instalar_demo.bat
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
if "%~1"=="" (
  set /p CARPETA="Arrastra aca la carpeta con radiografias y presiona Enter: "
) else (
  set "CARPETA=%~1"
)
python scripts\probar_externas.py --carpeta "%CARPETA%" --salida resultado_externas.html
if exist resultado_externas.html start "" resultado_externas.html
pause
