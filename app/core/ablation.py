"""Comparación de las configuraciones sobre una misma imagen y ensamblado.

Cada configuración difiere del baseline en una sola variable (estrategia de
incertidumbre, función de pérdida o arquitectura), así que la diferencia entre sus
predicciones para una imagen muestra el efecto de esa variable. El ensamblado combina
todas en una única lectura y usa su desacuerdo como señal de incertidumbre.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from src.data.labels import COMPETITION_TASKS, OBSERVATIONS

from . import registry


@dataclass
class FilaComparacion:
    """Predicción de una configuración para una observación puntual."""

    eid: str
    probabilidad: float
    delta_vs_baseline: float  # diferencia respecto a E1 (positivo = más probable)


def predecir_todos(clasificadores: dict, imagen: Image.Image) -> dict:
    """Corre cada clasificador disponible sobre la imagen. Devuelve id -> Prediccion."""
    return {eid: clf.predecir(imagen) for eid, clf in clasificadores.items()}


@dataclass
class Ensemble:
    """Respuesta combinada de todas las configuraciones sobre una imagen.

    Promedia las probabilidades y los umbrales de los modelos individuales. Expone la
    misma interfaz que una predicción suelta (``probabilidades``, ``umbral`` y
    ``ordenadas``) para poder reutilizar los mismos componentes de la interfaz. La
    combinación por promedio es la forma más simple de ensemble y suele mejorar la
    estabilidad respecto a cualquier modelo individual, porque promedia sus errores.
    """

    probabilidades: dict[str, float]
    umbrales: dict[str, float]
    individuales: dict = field(default_factory=dict)  # id -> Prediccion
    operacion: frozenset = frozenset()  # observaciones con umbral de operación propio

    def umbral(self, obs: str) -> float:
        return self.umbrales.get(obs, 0.5)

    def ordenadas(self, solo_benchmark: bool = False) -> list[tuple[str, float]]:
        items = self.probabilidades.items()
        if solo_benchmark:
            items = [(k, v) for k, v in items if k in COMPETITION_TASKS]
        return sorted(items, key=lambda kv: kv[1], reverse=True)

    def acuerdo(self, obs: str) -> int:
        """Cuántos modelos individuales señalan esa observación (superan su umbral).

        Si la observación tiene umbral de operación del sistema, cada modelo se compara
        contra ese mismo umbral, para que el conteo sea coherente con la lectura de conjunto.
        """
        if obs in self.operacion:
            u = self.umbral(obs)
            return sum(1 for p in self.individuales.values()
                       if p.probabilidades.get(obs, 0.0) >= u)
        return sum(1 for p in self.individuales.values()
                   if p.probabilidades.get(obs, 0.0) >= p.umbral(obs))

    def dispersion(self, obs: str) -> float:
        """Desviación estándar de las probabilidades de los modelos para una observación.

        Es una medida de incertidumbre del propio sistema: si las seis configuraciones
        coinciden, la dispersión es baja y la respuesta es firme; si discrepan, la
        dispersión es alta y conviene tomar la predicción con cautela. 
        """
        vals = [p.probabilidades.get(obs, 0.0) for p in self.individuales.values()]
        if len(vals) < 2:
            return 0.0
        media = sum(vals) / len(vals)
        return (sum((v - media) ** 2 for v in vals) / len(vals)) ** 0.5

    def nivel_confianza(self, obs: str) -> str:
        """Traduce la dispersión a una franja legible: alta, media o baja concordancia."""
        d = self.dispersion(obs)
        if d < 0.08:
            return "alta"
        if d < 0.18:
            return "media"
        return "baja"


def predecir_ensemble(clasificadores: dict, imagen: Image.Image,
                      pesos: dict | None = None, tta: bool = False,
                      umbrales_operacion: dict | None = None) -> Ensemble:
    """Combina las predicciones de todas las configuraciones en una sola respuesta.

    ``pesos`` es un diccionario opcional ``{observacion: {id_modelo: peso}}`` que permite
    ponderar a cada modelo según su desempeño en validación, o incluso rutear cada
    patología al modelo que mejor la resuelve (peso 1 a uno, 0 al resto). Si no se pasa,
    el promedio es uniforme. ``tta`` activa test-time augmentation en cada modelo.
    ``umbrales_operacion`` (``{observacion: umbral}``) fija el punto de operación del
    sistema; para las observaciones que no figuran se usa el promedio de los umbrales de
    los modelos (máximo F1 en validación interna).
    """
    individuales = {eid: clf.predecir(imagen, tta=tta)
                    for eid, clf in clasificadores.items()}

    def combinar(valor_de, obs):
        w = (pesos or {}).get(obs)
        if w:
            pares = [(w[e], valor_de(p)) for e, p in individuales.items()
                     if w.get(e, 0) > 0]
            total = sum(pi for pi, _ in pares)
            if total > 0:
                return sum(pi * vi for pi, vi in pares) / total
        # sin pesos (o todos en cero): promedio uniforme
        vals = [valor_de(p) for p in individuales.values()]
        return sum(vals) / len(vals)

    probs, umbrales = {}, {}
    for obs in OBSERVATIONS:
        probs[obs] = combinar(lambda p: p.probabilidades.get(obs, 0.0), obs)
        umbrales[obs] = combinar(lambda p: p.umbral(obs), obs)
    operacion = frozenset()
    if umbrales_operacion:
        operacion = frozenset(o for o in umbrales_operacion if o in umbrales)
        for o in operacion:
            umbrales[o] = float(umbrales_operacion[o])
    return Ensemble(probs, umbrales, individuales, operacion)


def comparar_observacion(predicciones: dict, observacion: str) -> dict:
    """Agrupa por eje la probabilidad de una observación en cada configuración.

    Devuelve ``{eje: [FilaComparacion, ...]}`` usando E1 como referencia para el
    delta. Solo incluye las configuraciones que estén cargadas.
    """
    base = None
    if "E1" in predicciones:
        base = predicciones["E1"].probabilidades.get(observacion)

    salida: dict[str, list[FilaComparacion]] = {}
    for eje, ids in registry.EJES.items():
        filas = []
        for eid in ids:
            if eid not in predicciones:
                continue
            p = predicciones[eid].probabilidades.get(observacion, 0.0)
            delta = (p - base) if base is not None else 0.0
            filas.append(FilaComparacion(eid, p, delta))
        if len(filas) >= 2:  # un eje con una sola configuración no compara nada
            salida[eje] = filas
    return salida


def resumen_benchmark(predicciones: dict) -> dict:
    """Probabilidad de cada patología del benchmark por configuración.

    Estructura pensada para una tabla: ``{patologia: {eid: prob}}``. Es la vista que
    permite ver de un golpe dónde coinciden y dónde discrepan las configuraciones.
    """
    tabla: dict[str, dict[str, float]] = {}
    for obs in COMPETITION_TASKS:
        tabla[obs] = {
            eid: pred.probabilidades.get(obs, 0.0)
            for eid, pred in predicciones.items()
        }
    return tabla
