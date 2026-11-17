"""Resumen en lenguaje natural de la lectura del conjunto.

Traduce las probabilidades y la concordancia entre modelos a una frase orientativa:
describe qué señala el sistema y con cuánta firmeza, sin emitir un diagnóstico.
"""
from __future__ import annotations

from . import registry

_CONCORDANCIA = {
    "alta": "concordancia alta entre modelos",
    "media": "concordancia media entre modelos",
    "baja": "concordancia baja entre modelos",
}


def resumen_clinico(ensemble, senaladas: list[tuple[str, float]]) -> str:
    """Frase orientativa a partir de las observaciones señaladas y su concordancia.

    ``senaladas`` es la lista ``(observacion, probabilidad)`` ya filtrada por umbral y
    ordenada de mayor a menor.
    """
    if not senaladas:
        nf = ensemble.probabilidades.get("No Finding")
        cola = f" La probabilidad de un estudio sin hallazgos es {nf:.2f}." if nf else ""
        return ("El conjunto no identifica observaciones por encima del umbral en esta "
                "imagen." + cola)

    partes = []
    for obs, p in senaladas:
        conf = _CONCORDANCIA[ensemble.nivel_confianza(obs)]
        partes.append(f"{registry.es(obs).lower()} (probabilidad {p:.2f}, {conf})")

    if len(partes) == 1:
        cuerpo = partes[0]
    else:
        cuerpo = ", ".join(partes[:-1]) + " y " + partes[-1]

    return (f"El conjunto señala {cuerpo}. "
            "Es una orientación para priorizar la lectura, no un diagnóstico: la decisión "
            "clínica queda en manos del profesional.")


def detalle_confianza(ensemble, obs: str) -> str:
    """Texto corto de confianza para una observación (para chips o notas)."""
    return _CONCORDANCIA[ensemble.nivel_confianza(obs)]
