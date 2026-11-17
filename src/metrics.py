"""Metricas de evaluacion: AUC-ROC por observacion segun el protocolo de CheXpert."""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

from .data.labels import COMPETITION_TASKS, OBSERVATIONS

logger = logging.getLogger("chexpert")


def auc_per_observation(
    y_true: np.ndarray,
    y_score: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Optional[float]]:
    """AUC-ROC calculado de forma independiente para cada una de las 14 observaciones.

    Devuelve None para las columnas donde el conjunto de evaluacion contiene una
    sola clase: el AUC no esta definido en ese caso y forzar un valor distorsiona
    el promedio. Esto ocurre en el valid.csv oficial para algunas observaciones
    raras, porque son solo 234 imagenes.
    """
    out: Dict[str, Optional[float]] = {}
    for i, name in enumerate(OBSERVATIONS):
        yt, ys = y_true[:, i], y_score[:, i]
        if mask is not None:
            sel = mask[:, i] > 0
            yt, ys = yt[sel], ys[sel]
        if yt.size == 0 or len(np.unique(yt)) < 2:
            out[name] = None
            continue
        out[name] = float(roc_auc_score(yt, ys))
    return out


def mean_competition_auc(auc: Dict[str, Optional[float]]) -> Optional[float]:
    """Promedio sobre las cinco patologias del benchmark oficial.

    Es la metrica principal de comparacion entre configuraciones y la que permite
    contrastar con los valores de referencia de Irvin et al. (2019).
    """
    vals = [auc[t] for t in COMPETITION_TASKS if auc.get(t) is not None]
    return float(np.mean(vals)) if vals else None


def mean_all_auc(auc: Dict[str, Optional[float]]) -> Optional[float]:
    vals = [v for v in auc.values() if v is not None]
    return float(np.mean(vals)) if vals else None


def threshold_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, float]]:
    """Precision, recall y F1 por observacion a un umbral fijo.

    Son secundarias respecto al AUC-ROC porque dependen del umbral, pero permiten
    analizar el recall en las patologias graves.
    """
    out: Dict[str, Dict[str, float]] = {}
    y_pred = (y_score >= threshold).astype(int)
    for i, name in enumerate(OBSERVATIONS):
        yt, yp, ys = y_true[:, i], y_pred[:, i], y_score[:, i]
        if mask is not None:
            sel = mask[:, i] > 0
            yt, yp, ys = yt[sel], yp[sel], ys[sel]
        if yt.size == 0:
            continue
        entry = {
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "n_positives": int(yt.sum()),
            "n": int(yt.size),
        }
        entry["average_precision"] = (
            float(average_precision_score(yt, ys)) if len(np.unique(yt)) > 1 else float("nan")
        )
        out[name] = entry
    return out


def best_threshold_per_observation(
    y_true: np.ndarray,
    y_score: np.ndarray,
    grid: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Umbral que maximiza el F1 por observacion.

    Se calcula sobre la validacion interna, nunca sobre el conjunto oficial: elegir
    el umbral con los datos de test invalidaria la comparacion con la literatura.
    """
    grid = np.arange(0.05, 0.96, 0.05) if grid is None else grid
    out: Dict[str, float] = {}
    for i, name in enumerate(OBSERVATIONS):
        yt, ys = y_true[:, i], y_score[:, i]
        if len(np.unique(yt)) < 2:
            out[name] = 0.5
            continue
        scores = [f1_score(yt, (ys >= t).astype(int), zero_division=0) for t in grid]
        out[name] = float(grid[int(np.argmax(scores))])
    return out


def summarize(y_true: np.ndarray, y_score: np.ndarray,
              mask: Optional[np.ndarray] = None) -> Dict:
    auc = auc_per_observation(y_true, y_score, mask)
    return {
        "auc_per_observation": auc,
        "auc_competition_mean": mean_competition_auc(auc),
        "auc_all_mean": mean_all_auc(auc),
        "auc_competition": {t: auc.get(t) for t in COMPETITION_TASKS},
    }
