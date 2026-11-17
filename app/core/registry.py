"""Catálogo de las seis configuraciones del experimento y descubrimiento de modelos.

Es la única fuente de verdad sobre qué es cada configuración: qué eje del diseño de
ablación modifica, qué variante usa y cuánto rindió. Las pantallas leen de acá en
lugar de repetir esa información, así que agregar una configuración nueva (una E7)
es sumar una entrada a este diccionario y nada más.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.inference import buscar_modelos


@dataclass(frozen=True)
class Experimento:
    """Metadatos de una configuración del diseño experimental."""

    id: str            # "E1"
    nombre: str        # "Baseline"
    eje: str           # eje de ablación que modifica respecto al baseline
    variante: str      # descripción corta de la variante
    backbone: str      # arquitectura
    incertidumbre: str # política de etiquetas inciertas
    perdida: str       # función de pérdida
    color: str         # color con el que se identifica en gráficos y tarjetas
    auc: Optional[float] = None  # AUC-ROC medio en el conjunto oficial


# Los AUC medios provienen de la evaluación final sobre el conjunto de validación
# oficial de CheXpert.
CATALOGO: dict[str, Experimento] = {
    "E1": Experimento("E1", "Baseline", "—", "U-Zeros / BCE",
                      "DenseNet-121", "U-Zeros", "BCE", "#6b7280", 0.863),
    "E2": Experimento("E2", "U-Ones", "Incertidumbre", "Inciertos como positivos",
                      "DenseNet-121", "U-Ones", "BCE", "#d97706", 0.877),
    "E3": Experimento("E3", "U-Ignore", "Incertidumbre", "Inciertos enmascarados",
                      "DenseNet-121", "U-Ignore", "BCE", "#f59e0b", 0.864),
    "E4": Experimento("E4", "BCE ponderada", "Función de pérdida", "Peso por frecuencia",
                      "DenseNet-121", "U-Zeros", "BCE ponderada", "#2563eb", 0.860),
    "E5": Experimento("E5", "Focal loss", "Función de pérdida", "Enfoca casos difíciles",
                      "DenseNet-121", "U-Zeros", "Focal", "#3b82f6", 0.861),
    "E6": Experimento("E6", "ResNet-50", "Arquitectura", "Backbone alternativo",
                      "ResNet-50", "U-Zeros", "BCE", "#059669", 0.882),
    "E7": Experimento("E7", "Combinado (aplicado)", "—",
                      "ResNet-50 + U-Ones a 320px", "ResNet-50", "U-Ones", "BCE",
                      "#7c3aed", None),
}

# Los tres ejes de comparación del diseño. Cada eje incluye al baseline E1 como
# punto de referencia, porque el efecto se mide siempre contra él.
EJES: dict[str, list[str]] = {
    "Incertidumbre": ["E1", "E2", "E3"],
    "Función de pérdida": ["E1", "E4", "E5"],
    "Arquitectura": ["E1", "E6"],
}

# Nombres en español de las 14 observaciones, para la interfaz. El dataset las nombra
# en inglés; se traducen para que la lectura no dependa del término técnico en otro
# idioma.
NOMBRES_ES: dict[str, str] = {
    "No Finding": "Sin hallazgos",
    "Enlarged Cardiomediastinum": "Mediastino ensanchado",
    "Cardiomegaly": "Cardiomegalia",
    "Lung Opacity": "Opacidad pulmonar",
    "Lung Lesion": "Lesión pulmonar",
    "Edema": "Edema",
    "Consolidation": "Consolidación",
    "Pneumonia": "Neumonía",
    "Atelectasis": "Atelectasia",
    "Pneumothorax": "Neumotórax",
    "Pleural Effusion": "Derrame pleural",
    "Pleural Other": "Otra alteración pleural",
    "Fracture": "Fractura",
    "Support Devices": "Dispositivos de soporte",
}


def es(observacion: str) -> str:
    """Nombre en español de una observación (o el original si no está mapeado)."""
    return NOMBRES_ES.get(observacion, observacion)


def id_de_ruta(ruta: Path) -> Optional[str]:
    """Extrae el identificador de configuración (E1..E6) del nombre del archivo.

    Los checkpoints se llaman ``E2_uones.pt`` o
    ``E1_baseline_densenet121_uzeros_bce.pt``; en ambos casos el prefijo antes del
    primer guion bajo es el identificador.
    """
    prefijo = ruta.stem.split("_", 1)[0].upper()
    return prefijo if prefijo in CATALOGO else None


def descubrir(raiz: str | Path = ".") -> dict[str, Path]:
    """Mapa id -> ruta del modelo, para las configuraciones que estén disponibles.

    Prioriza los modelos livianos de ``models/`` y, si no están, usa ``checkpoints/``.
    Devuelve solo lo que existe, de modo que la aplicación funciona aunque haya una
    sola configuración entrenada.
    """
    encontrados: dict[str, Path] = {}
    for ruta in buscar_modelos(raiz):
        eid = id_de_ruta(ruta)
        if eid and eid not in encontrados:
            encontrados[eid] = ruta
    # Orden estable E1..E6.
    return {k: encontrados[k] for k in CATALOGO if k in encontrados}


def info(eid: str) -> Experimento:
    """Metadatos de una configuración por su identificador."""
    return CATALOGO[eid]
