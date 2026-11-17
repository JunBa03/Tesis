@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title RXTorax - Entrenamiento E1 a E6

REM Numero de epocas por configuracion. Cambialo aca segun lo que haya dado
REM la calibracion de 1_verificar.bat (8 por defecto, 12 si sobra tiempo).
set EPOCAS=12

echo ================================================================
echo  ENTRENAMIENTO COMPLETO  -  E1 a E6, %EPOCAS% epocas cada una
echo ================================================================
echo.
echo  Estimado: ~12 horas (medido: 9 min/epoca DenseNet, 7 min ResNet-50).
echo.
echo  Se puede cortar en cualquier momento con Ctrl+C y volver a correr
echo  este mismo archivo: cada configuracion retoma donde quedo, y las
echo  que ya terminaron se saltean solas.
echo.
echo  Mientras corre, en otra ventana:
echo      python scripts\make_dashboard.py --open
echo.
pause

if not exist ".venv\Scripts\activate.bat" (
  echo No hay entorno virtual. Corre primero 1_verificar.bat
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat

echo.
echo === Split por paciente ===
python scripts\prepare_splits.py --config configs\e1_baseline.yaml || goto :error

for %%C in (e1_baseline e2_uones e3_uignore e4_weighted_bce e5_focal e6_resnet50) do (
  echo.
  echo ================================================================
  echo  %%C   -   %date% %time%
  echo ================================================================
  python scripts\train.py --config configs\%%C.yaml --resume --set train.epochs=%EPOCAS%
  if errorlevel 1 (
    echo.
    echo  *** %%C fallo. Se sigue con la siguiente. ***
    echo  *** Revisa results\*\train.log y avisame.  ***
    echo.
  )
  REM Panel actualizado despues de cada configuracion.
  python scripts\make_dashboard.py
)

echo.
echo === Consolidando resultados ===
python scripts\compare_results.py --results-dir results
python scripts\make_figures.py
python scripts\make_dashboard.py

echo.
echo ================================================================
echo  TERMINADO  -  %date% %time%
echo.
echo   results\panel.html                    panel de experimentos
echo   results\comparison\comparison_table.md tabla E1-E6
echo   results\figuras\                       figuras de resultados
echo ================================================================
pause
exit /b 0

:error
echo.
echo  FALLO antes de empezar a entrenar. Copiame el error.
pause
exit /b 1
