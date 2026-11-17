"""Calibración de probabilidades y evaluación de derivación selectiva.

Solo numpy: son funciones puras que reciben predicciones y etiquetas y devuelven
números. La orquestación (correr los modelos, guardar figuras) vive en el script
``scripts/calibracion_riesgo.py``.

Dos herramientas:

  - Temperature scaling: ajusta un único escalar T que divide a los logits antes de la
    sigmoide, para que las probabilidades reflejen la frecuencia real de los eventos. No
    altera el orden (ni el AUC); solo la escala.
  - Curva riesgo-cobertura: mide la derivación selectiva. Ordena las decisiones por
    confianza y reporta el error sobre la fracción que el sistema decide responder.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    """Inversa de la sigmoide, con recorte para no dividir por cero."""
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoide(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _validos(prob, label, mask):
    """Aplana a los pares (predicción, etiqueta) que participan de la métrica."""
    if mask is None:
        mask = np.ones_like(prob)
    sel = mask > 0
    return prob[sel].ravel(), label[sel].ravel()


def ece(prob: np.ndarray, label: np.ndarray, mask=None, n_bins: int = 10) -> float:
    """Expected Calibration Error sobre las entradas válidas.

    Reparte las predicciones en ``n_bins`` intervalos de confianza; en cada uno compara la
    confianza promedio con la proporción real de positivos, y promedia la brecha pesando
    por cuántas caen en cada intervalo.
    """
    p, y = _validos(prob, label, mask)
    if p.size == 0:
        return float("nan")
    bordes = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        en_bin = (p >= bordes[i]) & (p < bordes[i + 1] if i < n_bins - 1 else p <= bordes[i + 1])
        if not en_bin.any():
            continue
        conf = p[en_bin].mean()
        acc = y[en_bin].mean()
        total += (en_bin.sum() / p.size) * abs(acc - conf)
    return float(total)


def bce(prob: np.ndarray, y: np.ndarray) -> float:
    prob = np.clip(prob, EPS, 1 - EPS)
    return float(-(y * np.log(prob) + (1 - y) * np.log(1 - prob)).mean())


def ajustar_temperatura(prob: np.ndarray, label: np.ndarray, mask=None) -> float:
    """Busca la temperatura T que minimiza la entropía cruzada sobre validación.

    Recibe probabilidades (recupera los logits internamente). Devuelve T: valores mayores
    a 1 suavizan predicciones sobreconfiadas; menores a 1 las agudizan.
    """
    p, y = _validos(prob, label, mask)
    if p.size == 0:
        return 1.0
    z = logit(p)
    # Búsqueda en una grilla logarítmica y refinamiento local; la pérdida es suave en T.
    Ts = np.geomspace(0.3, 6.0, 120)
    perdidas = [bce(sigmoide(z / T), y) for T in Ts]
    T0 = Ts[int(np.argmin(perdidas))]
    finos = np.linspace(max(0.3, T0 * 0.7), T0 * 1.3, 60)
    perdidas_f = [bce(sigmoide(z / T), y) for T in finos]
    return float(finos[int(np.argmin(perdidas_f))])


def aplicar_temperatura(prob: np.ndarray, T: float) -> np.ndarray:
    """Reescala probabilidades por temperatura: sigmoide(logit(p) / T)."""
    return sigmoide(logit(prob) / T)


def curva_riesgo_cobertura(confianza: np.ndarray, correcto: np.ndarray):
    """Curva riesgo-cobertura a partir de una confianza y un acierto por decisión.

    ``confianza`` es un puntaje (mayor = más seguro); ``correcto`` es 1 si la decisión fue
    acertada, 0 si no. Devuelve ``(cobertura, riesgo, aurc)``: al bajar la cobertura (al
    responder solo las decisiones más confiables) el riesgo debería caer. ``aurc`` es el
    área bajo la curva (menor es mejor).
    """
    orden = np.argsort(-confianza)          # de más a menos confiable
    acierto_ordenado = correcto[orden].astype(float)
    n = len(correcto)
    aciertos_acumulados = np.cumsum(acierto_ordenado)
    k = np.arange(1, n + 1)
    cobertura = k / n
    riesgo = 1.0 - aciertos_acumulados / k  # error entre las k más confiables
    # Área bajo la curva por regla del trapecio, sin depender de np.trapz/np.trapezoid
    # (cuyo nombre cambió entre versiones de numpy).
    aurc = float(np.sum(np.diff(cobertura) * (riesgo[1:] + riesgo[:-1]) / 2))
    return cobertura, riesgo, aurc


def riesgo_a_cobertura(cobertura: np.ndarray, riesgo: np.ndarray, c: float) -> float:
    """Riesgo cuando el sistema responde una fracción ``c`` de los casos."""
    idx = int(np.clip(np.searchsorted(cobertura, c), 0, len(riesgo) - 1))
    return float(riesgo[idx])
