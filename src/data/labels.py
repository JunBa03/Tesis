"""Definicion de las 14 observaciones de CheXpert y aplicacion de las politicas de incertidumbre."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

# Orden exacto de las columnas de observacion en train.csv / valid.csv de CheXpert.
OBSERVATIONS: list[str] = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# Las cinco patologias sobre las que el benchmark oficial calcula el AUC-ROC promedio
# (Irvin et al., 2019).
COMPETITION_TASKS: list[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

COMPETITION_INDICES: list[int] = [OBSERVATIONS.index(t) for t in COMPETITION_TASKS]

# Nombres cortos para las tablas comparativas del informe.
SHORT_NAMES = {
    "Atelectasis": "Ate.",
    "Cardiomegaly": "Card.",
    "Consolidation": "Con.",
    "Edema": "Ede.",
    "Pleural Effusion": "Der.",
}

# Valores de referencia reportados por Irvin et al. (2019) sobre el conjunto de
# validacion oficial (234 imagenes, consenso de tres radiologos). Se usan como
# punto de comparacion en compare_results.py.
REFERENCE_AUC = {
    "DenseNet-121 + U-Zeros (Irvin et al., 2019)": {
        "Atelectasis": 0.811, "Cardiomegaly": 0.840, "Consolidation": 0.932,
        "Edema": 0.929, "Pleural Effusion": 0.931, "mean": 0.889,
    },
    "DenseNet-121 + U-Ones (Irvin et al., 2019)": {
        "Atelectasis": 0.858, "Cardiomegaly": 0.832, "Consolidation": 0.899,
        "Edema": 0.941, "Pleural Effusion": 0.934, "mean": 0.893,
    },
    "Radiologos (comite)": {
        "Atelectasis": 0.821, "Cardiomegaly": 0.888, "Consolidation": 0.898,
        "Edema": 0.941, "Pleural Effusion": 0.967, "mean": 0.903,
    },
}

UNCERTAIN = -1.0
POSITIVE = 1.0
NEGATIVE = 0.0


def build_label_matrix(
    df: pd.DataFrame,
    policy: str,
    u_ignore_mode: str = "row",
    blank_as_negative: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Convierte las columnas de observacion en matrices de etiquetas y de mascara.

    CheXpert codifica cada observacion con cuatro valores posibles: 1.0 (positivo),
    0.0 (negativo), -1.0 (incierto) y celda vacia (no mencionado). Esta funcion
    resuelve los dos ultimos segun la politica configurada.

    Args:
        df: DataFrame con al menos las 14 columnas de OBSERVATIONS.
        policy: 'u_zeros' (incierto -> 0), 'u_ones' (incierto -> 1) o 'u_ignore'.
        u_ignore_mode: para 'u_ignore', 'row' descarta la fila completa cuando alguna
            de las cinco patologias del benchmark es incierta (protocolo de Irvin et al.);
            'mask' conserva la fila y excluye solo esa etiqueta del calculo de la perdida.
        blank_as_negative: trata las celdas vacias como negativas (practica estandar).

    Returns:
        labels: matriz float32 (N, 14) con valores en {0.0, 1.0}.
        mask: matriz float32 (N, 14) con 1.0 donde la etiqueta contribuye a la perdida.
        df_out: el DataFrame filtrado (identico a df salvo en u_ignore/'row').
    """
    if policy not in {"u_zeros", "u_ones", "u_ignore"}:
        raise ValueError(f"Politica desconocida: {policy!r}")

    missing = [c for c in OBSERVATIONS if c not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas de observacion en el CSV: {missing}")

    df_out = df

    if policy == "u_ignore" and u_ignore_mode == "row":
        # Se descartan las filas con -1 en alguna de las cinco patologias evaluadas.
        comp = df_out[COMPETITION_TASKS].to_numpy(dtype=np.float32)
        keep = ~np.any(comp == UNCERTAIN, axis=1)
        df_out = df_out.loc[keep].reset_index(drop=True)

    raw = df_out[OBSERVATIONS].to_numpy(dtype=np.float32)

    # Celdas vacias -> NaN en el CSV.
    blank = np.isnan(raw)
    uncertain = raw == UNCERTAIN

    labels = np.nan_to_num(raw, nan=0.0).astype(np.float32)
    mask = np.ones_like(labels, dtype=np.float32)

    if not blank_as_negative:
        mask[blank] = 0.0
    labels[blank] = 0.0

    if policy == "u_zeros":
        labels[uncertain] = NEGATIVE
    elif policy == "u_ones":
        labels[uncertain] = POSITIVE
    else:  # u_ignore
        labels[uncertain] = NEGATIVE  # valor irrelevante: queda enmascarado
        mask[uncertain] = 0.0

    return labels, mask, df_out


def positive_counts(labels: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Cuenta positivos y negativos efectivos por observacion (respetando la mascara)."""
    pos = ((labels == 1.0) & (mask == 1.0)).sum(axis=0)
    neg = ((labels == 0.0) & (mask == 1.0)).sum(axis=0)
    return pos.astype(np.int64), neg.astype(np.int64)


def prevalence_report(labels: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """Tabla de prevalencia por observacion, util para justificar la ponderacion de clases."""
    pos, neg = positive_counts(labels, mask)
    total = pos + neg
    with np.errstate(divide="ignore", invalid="ignore"):
        prev = np.where(total > 0, pos / np.maximum(total, 1), 0.0)
    return pd.DataFrame(
        {
            "observation": OBSERVATIONS,
            "positives": pos,
            "negatives": neg,
            "effective_n": total,
            "prevalence": prev.round(5),
            "in_benchmark": [o in COMPETITION_TASKS for o in OBSERVATIONS],
        }
    )
