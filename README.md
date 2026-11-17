# RXTórax

Sistema de asistencia para la lectura de radiografías de tórax. Estima 14 observaciones
por estudio con un ensamblado de siete redes convolucionales entrenadas sobre CheXpert,
prioriza la cola de estudios por urgencia, señala con un mapa de calor dónde mirar y
avisa cuándo su propia lectura no es confiable. No emite diagnósticos.

Proyecto Final de Ingeniería en Informática, UADE (2026).
Autor: Juan Bautista Felizia. Tutor: Damián Andrés Del Campo.

> Este repositorio se armó para la entrega final, con una versión más limpia y ordenada
> del código desarrollado durante el proyecto.

## Funciones

- **Análisis de un estudio:** lectura de conjunto con la probabilidad de cada hallazgo,
  la concordancia entre los siete modelos, una recomendación (revisar o derivar) y un
  mapa de atención Grad-CAM++ recortado a los campos pulmonares.
- **Priorización:** carga de un lote de estudios y cola ordenada por urgencia,
  exportable a CSV.
- **Comparación por modelo:** la lectura y el mapa de cada configuración por separado.
- **Informe exportable** en HTML y **registro de correcciones** del especialista.
- Acepta JPG, PNG y DICOM, y funciona sin conexión. El asistente conversacional es
  opcional y es lo único que usa internet.

## Modelos

Seis configuraciones forman un estudio de ablación controlado: cada una cambia una sola
variable respecto del baseline E1. Una séptima, E7, combina las mejores decisiones a
mayor resolución.

| Config. | Variable | Arquitectura | Incertidumbre | Pérdida |
|---|---|---|---|---|
| E1 | baseline | DenseNet-121 | U-Zeros | BCE |
| E2 | incertidumbre | DenseNet-121 | U-Ones | BCE |
| E3 | incertidumbre | DenseNet-121 | U-Ignore | BCE |
| E4 | pérdida | DenseNet-121 | U-Zeros | BCE ponderada |
| E5 | pérdida | DenseNet-121 | U-Zeros | Focal loss |
| E6 | arquitectura | ResNet-50 | U-Zeros | BCE |
| E7 | aplicada | ResNet-50, 320 px | U-Ones | BCE |

Sobre las 202 radiografías frontales del conjunto de validación oficial de CheXpert, el
ensamblado de las siete con test-time augmentation alcanza un AUC-ROC medio de 0,8905 en
las cinco patologías del benchmark. Con los umbrales de operación de la aplicación, la
sensibilidad media es de 0,86 y el valor predictivo negativo, de 0,93.

## Datos

Se usa CheXpert-v1.0-small (Stanford AIMI), disponible con su acuerdo de uso en
<https://aimi.stanford.edu/datasets/chexpert-chest-x-rays>. Las imágenes no se incluyen
en el repositorio. Por defecto se buscan en `../data/raw/CheXpert-v1.0-small/`; la ruta
se cambia en `data.root` de `configs/base.yaml`.

## Prueba rápida (Windows)

1. Descargar el repositorio (**Code → Download ZIP**) y descomprimirlo.
2. Ejecutar `instalar.bat` una sola vez. Crea el entorno, instala las dependencias y
   descarga los siete modelos entrenados (unos 160 MB) desde la sección
   [Releases](https://github.com/JunBa03/Tesis/releases) del repositorio.
3. Ejecutar `iniciar.bat`. La aplicación se abre en el navegador.
4. En **Análisis**, cargar una radiografía de tórax frontal en JPG, PNG o DICOM.

Requiere Python 3.10 o superior y conexión a internet durante la instalación. No necesita
placa de video; en una computadora sin GPU el análisis de un estudio tarda unos segundos.

Las imágenes de CheXpert no se incluyen porque su acuerdo de uso no permite
redistribuirlas, pero no hacen falta: sirve cualquier radiografía de tórax frontal. El
asistente conversacional es opcional y requiere una clave propia (ver más abajo); sin ella
la aplicación funciona igual.

Los modelos publicados se entrenaron con CheXpert y se ofrecen solo para uso académico y
de investigación, en las mismas condiciones que el conjunto de datos.

## Instalación manual

Para otros sistemas operativos o para entrenar:

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch se instala aparte, con la versión que corresponda al equipo
(<https://pytorch.org/get-started/locally/>). Los modelos se descargan de
[Releases](https://github.com/JunBa03/Tesis/releases) (`modelos.zip`) y se descomprimen
en `models/`.

## Uso

```bash
# aplicación
streamlit run app/Inicio.py

# entrenamiento de una configuración
python scripts/train.py --config configs/e1_baseline.yaml

# análisis de una imagen sin interfaz
python scripts/predict_image.py --imagen ruta/a/radiografia.jpg

# pruebas
python -m pytest -q
```

Para el asistente conversacional, copiar `.streamlit/secrets.toml.example` como
`.streamlit/secrets.toml` y completar la clave.

## Estructura

```
app/        aplicación Streamlit (pantallas, motor de asistencia e interfaz)
configs/    configuración YAML de cada experimento (heredan de base.yaml)
results/    configuración del ensamblado, calibración, resumen y figuras
scripts/    entrenamiento, evaluación, análisis y utilidades
src/        datos, modelos, pérdidas, métricas e inferencia
tests/      pruebas automatizadas (pytest)
```

## Alcance

Herramienta de investigación, no de uso clínico. Fue entrenada y evaluada solo con
CheXpert y no tiene validación clínica sobre población local. Sus resultados no
reemplazan el juicio del profesional.
