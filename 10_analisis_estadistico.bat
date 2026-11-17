@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title RXTorax - Analisis estadistico

REM ======================================================================
REM  Analisis estadistico de los resultados. Requiere placa de video;
REM  conviene dejarlo corriendo de noche.
REM
REM   Paso 1  Predicciones por imagen de E1-E7 en el test oficial     ~5 min
REM   Paso 2  Latencia en CPU y en GPU             ~10 min
REM   Paso 3  E1 y E6 con dos semillas extra (43 y 44)                ~6 h
REM   Paso 4  Intervalos de confianza y analisis estadistico          ~1 min
REM
REM  Se puede cortar con Ctrl+C y volver a correr: las semillas retoman
REM  donde quedaron y las ya terminadas se saltean.
REM ======================================================================

if not exist ".venv\Scripts\activate.bat" (
  echo No hay entorno virtual en %CD%
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat

echo.
echo === Paso 1/4: predicciones por imagen ===
if exist "results\predicciones\test.npz" (
  echo   Ya existen, se saltea.
) else (
  python scripts\exportar_predicciones.py || goto :error
)

echo.
echo === Paso 2/4: latencia ===
python scripts\medir_latencia.py --device cpu --n 20
python scripts\medir_latencia.py --device cuda --n 20

echo.
echo === Paso 3/4: semillas extra de E1 y E6 ===
for %%S in (43 44) do (
  for %%C in (e1_baseline:E1_baseline_densenet121_uzeros_bce e6_resnet50:E6_resnet50) do (
    for /f "tokens=1,2 delims=:" %%a in ("%%C") do (
      if exist "results\%%b_seed%%S\results.json" (
        echo   %%b semilla %%S ya terminada, se saltea.
      ) else (
        echo.
        echo ----------------------------------------------------------------
        echo  %%b  semilla %%S   -   %date% %time%
        echo ----------------------------------------------------------------
        python scripts\train.py --config configs\%%a.yaml --resume --set train.epochs=12 experiment.seed=%%S data.split_seed=42 experiment.name=%%b_seed%%S output.checkpoint_dir=./checkpoints_semillas
        if errorlevel 1 echo  *** %%b semilla %%S fallo, se sigue con la siguiente ***
      )
    )
  )
)

echo.
echo === Paso 4/4: analisis estadistico ===
python scripts\analisis_estadistico.py || goto :error

echo.
echo ================================================================
echo  TERMINADO  -  %date% %time%
echo   results\estadistica\tabla.md       intervalos de confianza
echo   results\latencia_cpu.json          tiempos en CPU
echo   results\latencia_cuda.json         tiempos en GPU
echo  Avisale a Claude que termino.
echo ================================================================
pause
exit /b 0

:error
echo.
echo  FALLO. Copia el mensaje de arriba y pasaselo a Claude.
pause
exit /b 1
