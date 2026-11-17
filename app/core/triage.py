"""Priorización de la cola de estudios por probabilidad de hallazgo grave.

Ordena los estudios para que los de mayor sospecha de patología grave se lean primero.
No diagnostica: solo reordena la cola. El puntaje pondera cada patología por su
gravedad clínica, de modo que el neumotórax y el derrame pesan más que un hallazgo
crónico estable.
"""
from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

# Peso de gravedad por observación. El neumotórax es la urgencia que apareció de forma
# independiente en las tres entrevistas; las patologías agudas pesan más que las
# crónicas. Las observaciones no listadas no aportan al puntaje de urgencia.
GRAVEDAD: dict[str, float] = {
    "Pneumothorax": 1.00,
    "Edema": 0.85,
    "Consolidation": 0.80,
    "Pleural Effusion": 0.70,
    "Pneumonia": 0.70,
    "Lung Lesion": 0.65,
    "Atelectasis": 0.50,
    "Cardiomegaly": 0.45,
    "Enlarged Cardiomediastinum": 0.40,
    "Lung Opacity": 0.40,
    "Fracture": 0.55,
}


@dataclass
class ItemCola:
    """Un estudio en la cola, con su puntaje de urgencia y el hallazgo que lo motiva."""

    nombre: str
    imagen: Image.Image
    urgencia: float                 # 0..100
    motivo: str                     # patología que más contribuye al puntaje
    prob_motivo: float              # su probabilidad
    probabilidades: dict[str, float]
    verdad: dict | None = None      # etiquetas de radiólogos, si se conocen

    @property
    def nivel(self) -> str:
        """Franja de prioridad para el color de la interfaz."""
        if self.urgencia >= 60:
            return "alta"
        if self.urgencia >= 30:
            return "media"
        return "baja"


def puntaje(probabilidades: dict[str, float]) -> tuple[float, str, float]:
    """Urgencia de un estudio a partir de sus probabilidades.

    Devuelve ``(urgencia_0a100, patologia_dominante, su_probabilidad)``. La urgencia
    es la contribución máxima (probabilidad por gravedad) entre las patologías
    consideradas: basta un hallazgo grave probable para subir el estudio en la cola,
    aunque el resto esté limpio.
    """
    contribuciones = {
        obs: probabilidades.get(obs, 0.0) * peso
        for obs, peso in GRAVEDAD.items()
    }
    motivo = max(contribuciones, key=contribuciones.get)
    urgencia = contribuciones[motivo] * 100.0
    return urgencia, motivo, probabilidades.get(motivo, 0.0)


def construir_cola(estudios: list[dict], clasificador) -> list[ItemCola]:
    """Evalúa una lista de estudios y los devuelve ordenados por urgencia descendente.

    Cada estudio es un dict con ``nombre``, ``imagen`` (PIL) y opcionalmente
    ``verdad``. El orden resultante es la cola priorizada que vería el profesional.
    """
    items: list[ItemCola] = []
    for e in estudios:
        pred = clasificador.predecir(e["imagen"])
        urgencia, motivo, prob = puntaje(pred.probabilidades)
        items.append(ItemCola(
            nombre=e["nombre"],
            imagen=e["imagen"],
            urgencia=urgencia,
            motivo=motivo,
            prob_motivo=prob,
            probabilidades=pred.probabilidades,
            verdad=e.get("verdad"),
        ))
    items.sort(key=lambda it: it.urgencia, reverse=True)
    return items
