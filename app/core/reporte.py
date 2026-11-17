"""Arma el informe con formato radiológico: Técnica, Hallazgos, Impresión, Recomendación.

Convierte la salida del sistema (probabilidades, concordancia, derivación) en la
estructura de un informe clínico, la forma en que un radiólogo comunica su lectura. La
misma estructura alimenta la vista en pantalla y el informe descargable.
"""
from __future__ import annotations

from . import registry


def construir(ensemble, senaladas, recomendacion_estado: str,
              zona: str | None = None, meta: dict | None = None) -> dict:
    """Devuelve las cuatro secciones del informe como texto.

    ``recomendacion_estado`` es ``"hallazgo"``, ``"normal"`` o ``"derivar"``.
    """
    proy = ""
    if meta and meta.get("proyeccion"):
        proy = f", proyección {meta['proyeccion']}"
    tecnica = (f"Radiografía de tórax{proy}. Análisis automatizado asistido por "
               "inteligencia artificial (uso de investigación).")

    hallazgos = []
    if senaladas:
        for i, (obs, p) in enumerate(senaladas):
            conc = ensemble.nivel_confianza(obs)
            linea = f"{registry.es(obs)}: probabilidad {p:.2f}, concordancia {conc}"
            if i == 0 and zona:
                linea += f", atención en {zona}"
            hallazgos.append(linea + ".")
    else:
        nf = ensemble.probabilidades.get("No Finding")
        extra = f" Probabilidad de estudio sin hallazgos: {nf:.2f}." if nf is not None else ""
        hallazgos.append("No se identifican observaciones por encima del umbral "
                         "calibrado." + extra)

    if recomendacion_estado == "hallazgo":
        top = registry.es(senaladas[0][0])
        impresion = f"Hallazgo más probable: {top}."
        if len(senaladas) > 1:
            impresion += " Se señalan además otras observaciones (ver hallazgos)."
        recomendacion = "Se sugiere revisión dirigida del hallazgo señalado."
    elif recomendacion_estado == "normal":
        impresion = "Estudio sin hallazgos por encima del umbral."
        recomendacion = "Sin indicación de acción adicional por parte del sistema."
    else:  # derivar
        impresion = ("Incertidumbre diagnóstica: probabilidades en zona de decisión o "
                     "desacuerdo entre modelos.")
        recomendacion = "Se recomienda derivar el estudio a un radiólogo para su lectura."
    recomendacion += (" El resultado es orientativo; la decisión clínica corresponde al "
                      "profesional.")

    return {"tecnica": tecnica, "hallazgos": hallazgos,
            "impresion": impresion, "recomendacion": recomendacion}


def a_html(rep: dict) -> str:
    """Informe estructurado como bloque HTML (para pantalla y para la descarga)."""
    items = "".join(f"<li>{h}</li>" for h in rep["hallazgos"])
    return (
        '<div class="reporte">'
        f'<div class="sec"><div class="rot">Técnica</div><div>{rep["tecnica"]}</div></div>'
        f'<div class="sec"><div class="rot">Hallazgos</div><ul>{items}</ul></div>'
        f'<div class="sec"><div class="rot">Impresión</div><div>{rep["impresion"]}</div></div>'
        f'<div class="sec"><div class="rot">Recomendación</div>'
        f'<div>{rep["recomendacion"]}</div></div>'
        '</div>')
