"""Carga de modelos y ejemplos con caché de Streamlit.

Separa el costo (cargar 85 MB de pesos, leer el conjunto de validación) de las
páginas, que solo piden lo que necesitan. Todo lo que devuelve queda cacheado por el
servidor, de modo que abrir otra pantalla no vuelve a cargar los modelos.
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from . import theme  # asegura el bootstrap de sys.path

from src.config import load_config
from src.data.io import read_chexpert_csv, resolve_dataset_root
from src.data.labels import OBSERVATIONS
from src.inference import Clasificador, cargar_umbrales

from app.core import registry
from app.core.gradcam import GradCAM

RAIZ = theme.RAIZ


@st.cache_resource(show_spinner="Cargando el modelo…")
def clasificador(ruta: str) -> Clasificador:
    """Un clasificador listo para predecir, calentado para que la primera vez no tarde."""
    clf = Clasificador(ruta, umbrales=cargar_umbrales(Path(ruta).stem, RAIZ / "results"))
    clf.calentar()
    return clf


@st.cache_resource(show_spinner="Preparando el mapa de calor…")
def gradcam(ruta: str) -> GradCAM:
    """Grad-CAM sobre una instancia propia del modelo, para aislar los ganchos."""
    clf = Clasificador(ruta, umbrales=cargar_umbrales(Path(ruta).stem, RAIZ / "results"))
    return GradCAM(clf)


def modelos_disponibles() -> dict[str, Path]:
    """Mapa id -> ruta de las configuraciones entrenadas que existan."""
    return registry.descubrir(RAIZ)


def gradcams_disponibles() -> dict:
    """Un Grad-CAM por configuración disponible. Cada uno contiene su clasificador.

    Reutilizar el clasificador interno del Grad-CAM (``.clf``) para predecir evita
    cargar el mismo modelo dos veces cuando una pantalla necesita tanto la predicción
    como el mapa de calor.
    """
    return {eid: gradcam(str(ruta)) for eid, ruta in modelos_disponibles().items()}


@st.cache_data(show_spinner=False)
def datos_desempeno() -> dict:
    """Reúne las métricas guardadas para la ficha técnica.

    Lee, si existen, ``results/comparison/summary.json`` (AUC por configuración y
    patología), ``results/ensemble_config.json`` (AUC del ensemble) y
    ``results/calibracion.json`` (calibración y riesgo-cobertura). Devuelve lo que
    encuentre; la página se adapta a lo disponible.
    """
    def _leer(rel):
        f = RAIZ / rel
        if f.exists():
            try:
                return json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    return {
        "summary": _leer("results/comparison/summary.json"),
        "ensemble": _leer("results/ensemble_config.json"),
        "calibracion": _leer("results/calibracion.json"),
        "aplicado": _leer("results/E7_combinado/results.json"),
    }


@st.cache_data(show_spinner=False)
def config_ensemble() -> dict:
    """Configuración del mejor ensemble, si `scripts/evaluar_mejoras.py` ya corrió.

    Devuelve ``{}`` cuando no existe, en cuyo caso la demo usa el promedio uniforme sin
    TTA. La clave ``pesos`` es ``{observacion: {id_modelo: peso}}`` y ``usar_tta`` un
    booleano.
    """
    f = RAIZ / "results" / "ensemble_config.json"
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


@st.cache_data(show_spinner=False)
def ejemplos(n: int = 12) -> list[dict]:
    """Estudios de ejemplo con sus etiquetas de consenso.

    Busca primero ``demo_samples/`` y, si no está, usa el conjunto de validación
    completo. Cada ítem trae la ruta de la
    imagen y las etiquetas de los radiólogos.
    """
    portatil = RAIZ / "demo_samples" / "manifiesto.json"
    if portatil.exists():
        try:
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
        import pandas as pd

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
