"""Traduce el foco del mapa de calor a una zona anatómica en lenguaje radiológico.

En vez de un punto de píxeles, ubica la atención dentro de una grilla de campos pulmonares
 (superior, medio, inferior por cada hemitórax) y devuelve el nombre de la zona
dominante. Esa granularidad es la que el método permite: el mapa no localiza con
precisión, pero sí indica en qué zona se concentra la atención.

Convención radiológica: en la imagen, el lado izquierdo corresponde al hemitórax DERECHO del
paciente (las radiografías se leen como si se mirara al paciente de frente).
"""
from __future__ import annotations

import numpy as np

FILAS = ["superior", "medio", "inferior"]


def zona_dominante(cam: np.ndarray, mascara: np.ndarray):
    """Nombre de la zona torácica con mayor atención, o ``None`` si no hay máscara.

    Ubica el centro de gravedad de la atención dentro del recuadro del tórax. En vertical
    define el campo (superior/medio/inferior). En horizontal, si cae cerca de la línea
    media lo reporta como región central (mediastino o silueta cardíaca, donde se
    manifiesta la cardiomegalia); si no, como hemitórax derecho o izquierdo. Recordar que
    en la imagen la mitad izquierda es el hemitórax DERECHO del paciente.
    """
    if mascara is None or not mascara.any():
        return None
    ys, xs = np.where(mascara)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

    c = cam * mascara
    total = float(c.sum())
    if total <= 0:
        return None
    yy, xx = np.mgrid[0:cam.shape[0], 0:cam.shape[1]]
    cy = float((c * yy).sum()) / total
    cx = float((c * xx).sum()) / total
    fy = (cy - y0) / max(y1 - y0, 1)
    fx = (cx - x0) / max(x1 - x0, 1)

    fila = FILAS[0] if fy < 1 / 3 else (FILAS[1] if fy < 2 / 3 else FILAS[2])
    if 0.38 <= fx <= 0.62:
        return f"campo {fila}, región central (mediastino / silueta cardíaca)"
    hemi = "hemitórax derecho" if fx < 0.5 else "hemitórax izquierdo"
    return f"campo {fila}, {hemi}"
