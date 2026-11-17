"""Derivación selectiva: decidir cuándo el sistema responde y cuándo deriva.

Es el diferencial del proyecto frente a las herramientas comerciales. En lugar de emitir
siempre una respuesta (y sumar falsos positivos, el problema documentado de esas
herramientas) el sistema clasifica cada estudio en tres estados:

  - hallazgo confiable: al menos una observación supera su umbral con holgura y con buena
    concordancia entre modelos;
  - normal confiable: ninguna observación se acerca al umbral y hay acuerdo;
  - derivar al radiólogo: el caso cae en la zona de decisión (probabilidades cerca del
    umbral) o los modelos discrepan, de modo que no conviene que el sistema decida.

Este comportamiento (''saber cuándo no sabe'' y abstenerse) es la *selective
prediction* que la literatura señala como frontera y que casi ningún producto comercial
implementa. Solo es válido si la confianza está bien fundada, y acá se apoya en el
desacuerdo entre las seis configuraciones del ensemble.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Recomendacion:
    """Resultado de la derivación selectiva para un estudio."""

    estado: str                       # "hallazgo" | "normal" | "derivar"
    firmes: list = field(default_factory=list)    # [(obs, prob)] hallazgos confiables
    dudosas: list = field(default_factory=list)   # observaciones en zona de decisión
    senaladas: list = field(default_factory=list)  # [(obs, prob)] por umbral

    @property
    def deriva(self) -> bool:
        return self.estado == "derivar"


def evaluar(ensemble, senaladas, considerar,
            banda: float = 0.08, piso: float = 0.25) -> Recomendacion:
    """Decide el estado de un estudio.

    ``senaladas`` es ``[(obs, prob)]`` filtrado por umbral y ordenado; ``considerar`` es
    la lista de observaciones patológicas (sin ``No Finding`` ni ``Support Devices``).

    ``banda`` es el ancho de la zona de decisión alrededor del umbral. ``piso`` es la
    probabilidad mínima para que una observación cuente como indecisa: una patología con
    probabilidad muy baja, aunque quede cerca de un umbral bajo, no es motivo para derivar
    (no es un candidato real). Sin ese piso el sistema derivaba casi siempre, porque
    con catorce observaciones y seis modelos casi siempre hay alguna en zona ambigua.
    """
    marcadas = {o for o, _ in senaladas}

    # Observaciones indecisas: cerca del umbral con probabilidad no despreciable, o un
    # hallazgo ya señalado en el que los modelos discrepan.
    dudosas = []
    for o in considerar:
        p = ensemble.probabilidades[o]
        u = ensemble.umbral(o)
        cerca_del_umbral = abs(p - u) < banda and p >= piso
        senal_discrepante = o in marcadas and ensemble.nivel_confianza(o) == "baja"
        if cerca_del_umbral or senal_discrepante:
            dudosas.append(o)

    # Hallazgos firmes: superan el umbral con holgura y sin baja concordancia.
    firmes = [(o, p) for o, p in senaladas
              if (p - ensemble.umbral(o)) >= banda
              and ensemble.nivel_confianza(o) != "baja"]

    if firmes:
        estado = "hallazgo"
    elif dudosas:
        estado = "derivar"
    else:
        estado = "normal"

    return Recomendacion(estado=estado, firmes=firmes, dudosas=dudosas,
                         senaladas=list(senaladas))
