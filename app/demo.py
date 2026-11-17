"""Demostración mínima: cargar una radiografía y ver qué predice un modelo.

    streamlit run app/demo.py

El modelo se carga una sola vez y queda en caché. Devuelve probabilidades por
observación; no emite diagnósticos. La aplicación completa es app/Inicio.py.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import pandas as pd
import streamlit as st
from PIL import Image

from src.config import load_config
from src.data.io import read_chexpert_csv, resolve_dataset_root
from src.data.labels import COMPETITION_TASKS, OBSERVATIONS
from src.inference import Clasificador, buscar_modelos, cargar_umbrales

st.set_page_config(page_title="Análisis de radiografías de tórax",
                   layout="wide", initial_sidebar_state="expanded")

AMBAR = "#d97706"
TINTA = "#1f2937"

st.markdown("""
<style>
  .bloque { border:1px solid #e5e7eb; border-radius:10px; padding:14px 16px; margin-bottom:10px; }
  .obs { display:flex; justify-content:space-between; align-items:center; padding:7px 0;
         border-bottom:1px solid #f3f4f6; }
  .obs:last-child { border-bottom:none; }
  .obs .nombre { font-size:14px; }
  .obs .bench { font-weight:700; }
  .obs .valor { font-variant-numeric:tabular-nums; font-weight:600; font-size:14px; }
  .barra { height:7px; background:#f3f4f6; border-radius:4px; overflow:hidden; margin-top:3px; }
  .barra > div { height:100%; border-radius:4px; }
  .aviso { background:#fef3e2; border:1px solid #f6d9a8; border-radius:8px;
           padding:11px 14px; font-size:13px; color:#8a5108; }
  .meta { color:#6b7280; font-size:12.5px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Cargando el modelo…")
def cargar_clasificador(ruta: str) -> Clasificador:
    clf = Clasificador(ruta, umbrales=cargar_umbrales(Path(ruta).stem, RAIZ / "results"))
    clf.calentar()          # que la primera predicción real no tarde
    return clf


@st.cache_data(show_spinner=False)
def ejemplos_validacion(n: int = 12) -> list[dict]:
    """Imágenes de ejemplo con sus etiquetas de consenso.

    Busca primero en demo_samples/ y, si no está, usa el conjunto de validación
    completo.
    """
    portatil = RAIZ / "demo_samples" / "manifiesto.json"
    if portatil.exists():
        try:
            import json
            entradas = json.loads(portatil.read_text(encoding="utf-8"))
            return [
                {"ruta": str(RAIZ / "demo_samples" / e["archivo"]),
                 "nombre": e["archivo"],
                 "verdad": e["etiquetas"]}
                for e in entradas
                if (RAIZ / "demo_samples" / e["archivo"]).exists()
            ]
        except Exception:
            pass

    try:
        cfg = load_config(RAIZ / "configs" / "e1_baseline.yaml")
        root = resolve_dataset_root(cfg["data"]["root"])
        df = read_chexpert_csv(root, cfg["data"]["valid_csv"])
        df = df[df["Frontal/Lateral"].str.strip().str.lower() == "frontal"]
        muestra = df.sample(min(n, len(df)), random_state=7)
        return [
            {"ruta": str(Path(root) / fila["Path"]),
             "nombre": "/".join(Path(fila["Path"]).parts[-3:]),
             "verdad": {o: float(fila[o]) for o in OBSERVATIONS if pd.notna(fila[o])}}
            for _, fila in muestra.iterrows()
        ]
    except Exception:
        return []


def barra(prob: float, umbral: float, destacado: bool) -> str:
    color = AMBAR if prob >= umbral else ("#9ca3af" if not destacado else "#cbd5e1")
    return (f'<div class="barra"><div style="width:{prob*100:.1f}%;'
            f'background:{color}"></div></div>')


def panel_observaciones(pred, titulo: str, observaciones: list[str],
                        verdad: dict | None = None) -> None:
    st.markdown(f"**{titulo}**")
    filas = []
    for obs in observaciones:
        p = pred.probabilidades[obs]
        u = pred.umbral(obs)
        bench = obs in COMPETITION_TASKS
        etiqueta = f'<span class="nombre {"bench" if bench else ""}">{obs}</span>'
        if verdad is not None and obs in verdad:
            marca = "positivo" if verdad[obs] == 1.0 else "negativo"
            etiqueta += f' <span class="meta">· radiólogos: {marca}</span>'
        filas.append(
            f'<div class="obs"><div style="flex:1">{etiqueta}{barra(p, u, bench)}</div>'
            f'<div class="valor" style="width:64px;text-align:right">{p:.3f}</div></div>'
        )
    st.markdown(f'<div class="bloque">{"".join(filas)}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ barra lateral

st.sidebar.title("Configuración")

checkpoints = buscar_modelos(RAIZ)
if not checkpoints:
    st.title("Análisis automatizado de radiografías de tórax")
    st.error(
        "No hay ningún modelo disponible.\n\n"
        "Entrenar con `scripts/train.py` o copiar los modelos exportados "
        "en `models/`."
    )
    st.stop()

nombres = [p.stem for p in checkpoints]
elegido = st.sidebar.selectbox(
    "Modelo", nombres, index=0,
    help="Cada checkpoint es una de las seis configuraciones del experimento.",
)
ruta_ckpt = str(checkpoints[nombres.index(elegido)])

try:
    clf = cargar_clasificador(ruta_ckpt)
except Exception as exc:
    st.error(f"No se pudo cargar el modelo:\n\n```\n{exc}\n```")
    st.stop()

st.sidebar.markdown(
    f"""<div class="meta">
    <b>Arquitectura</b><br>{clf.backbone}<br><br>
    <b>Etiquetas inciertas</b><br>{clf.incertidumbre}<br><br>
    <b>Función de pérdida</b><br>{clf.perdida}<br><br>
    <b>Época seleccionada</b><br>{clf.epoca or '—'}<br><br>
    <b>AUC-ROC en validación</b><br>
    {f'{clf.auc_validacion:.4f}' if clf.auc_validacion else '—'}<br><br>
    <b>Dispositivo</b><br>{clf.device}
    </div>""",
    unsafe_allow_html=True,
)

mostrar_todas = st.sidebar.checkbox("Mostrar las 14 observaciones", value=False)
mostrar_verdad = st.sidebar.checkbox(
    "Mostrar etiquetas de los radiólogos", value=False,
    help="Solo disponible para las imágenes de ejemplo del conjunto oficial.",
)

# --------------------------------------------------------------------- principal

st.title("Análisis automatizado de radiografías de tórax")
st.markdown(
    '<div class="aviso"><b>Herramienta de investigación, no de uso clínico.</b> '
    'Devuelve una probabilidad por observación y no emite diagnósticos. '
    'Entrenada y evaluada exclusivamente sobre CheXpert, sin validación clínica.'
    '</div>', unsafe_allow_html=True)
st.write("")

col_izq, col_der = st.columns([1, 1.15], gap="large")

with col_izq:
    st.subheader("Radiografía")
    hay_ejemplos = bool(ejemplos_validacion())
    opciones = ["Subir un archivo"] + (["Ejemplo del conjunto oficial"] if hay_ejemplos else [])
    origen = st.radio("Origen", opciones, horizontal=True,
                      label_visibility="collapsed")

    imagen = None
    verdad = None

    if origen == "Subir un archivo":
        subido = st.file_uploader("Radiografía de tórax, proyección frontal",
                                  type=["jpg", "jpeg", "png"])
        if subido is not None:
            imagen = Image.open(subido)
    else:
        ejemplos = ejemplos_validacion()
        if ejemplos:
            etiquetas = [e["nombre"] for e in ejemplos]
            idx = st.selectbox("Estudio", range(len(etiquetas)),
                               format_func=lambda i: etiquetas[i])
            imagen = Image.open(ejemplos[idx]["ruta"])
            if mostrar_verdad:
                verdad = ejemplos[idx]["verdad"]

    if imagen is not None:
        st.image(imagen, use_container_width=True)

with col_der:
    st.subheader("Observaciones detectadas")
    if imagen is None:
        st.info("Cargá una radiografía o elegí un ejemplo para ver las predicciones.")
    else:
        with st.spinner("Analizando…"):
            pred = clf.predecir(imagen)

        panel_observaciones(
            pred, "Patologías del benchmark",
            [o for o, _ in pred.ordenadas(solo_benchmark=True)], verdad,
        )
        if mostrar_todas:
            resto = [o for o, _ in pred.ordenadas() if o not in COMPETITION_TASKS]
            panel_observaciones(pred, "Resto de las observaciones", resto, verdad)

        st.markdown(
            '<div class="meta">La barra se pinta en ámbar cuando la probabilidad '
            'supera el umbral calibrado para esa observación sobre la validación '
            'interna. El umbral es por patología, no uno global: cada hallazgo tiene '
            'una prevalencia distinta.</div>',
            unsafe_allow_html=True,
        )
