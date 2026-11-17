"""Ciclo de mejora continua: registro de correcciones del radiólogo.

Materializa el aprendizaje ``human-in-the-loop``. Cuando el sistema deriva un caso o se
equivoca, el profesional indica cuáles son las patologías realmente presentes; esa
corrección (una etiqueta confiable, de experto) se acumula en un archivo. El conjunto
de correcciones es el material de entrenamiento de la próxima iteración: el sistema aprende
de sus errores sin arriesgarse a aprender el ruido de las etiquetas automáticas, porque la
señal proviene de un radiólogo.

Este módulo solo registra y resume; el reentrenamiento periódico con las correcciones es la
etapa siguiente del proyecto.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


def _ruta(raiz) -> Path:
    return Path(raiz) / "correcciones" / "correcciones.jsonl"


def registrar(raiz, estudio: str, probabilidades: dict, umbrales: dict,
              presentes, considerar, recomendacion: str = "",
              origen: str = "nueva") -> dict:
    """Guarda una corrección y devuelve el registro escrito.

    ``presentes`` son las observaciones que el radiólogo marca como reales. Se compara con
    lo que el sistema había señalado por umbral para dejar registrado si coincidieron.

    ``origen`` distingue el uso de la corrección. ``"nueva"``: imagen nueva, cuya etiqueta
    aportó un radiólogo; es apta para el reentrenamiento futuro. ``"ejemplo_oficial"``: una
    imagen del conjunto de validación oficial, cuya etiqueta ya conocíamos; solo sirve para
    monitoreo y NUNCA para reentrenar, porque entrenar sobre el conjunto de prueba
    invalidaría la comparación con la literatura.
    """
    ruta = _ruta(raiz)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    senalo = [o for o in considerar if probabilidades.get(o, 0.0) >= umbrales.get(o, 0.5)]
    registro = {
        "fecha": time.strftime("%Y-%m-%d %H:%M:%S"),
        "estudio": estudio,
        "origen": origen,
        "apta_reentrenamiento": origen == "nueva",
        "recomendacion": recomendacion,
        "sistema_senalo": senalo,
        "radiologo_presentes": list(presentes),
        "coincide": set(senalo) == set(presentes),
        "probabilidades": {o: round(float(probabilidades.get(o, 0.0)), 3)
                           for o in considerar},
    }
    with open(ruta, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(registro, ensure_ascii=False) + "\n")
    return registro


def leer(raiz) -> list:
    ruta = _ruta(raiz)
    if not ruta.exists():
        return []
    salida = []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        linea = linea.strip()
        if linea:
            try:
                salida.append(json.loads(linea))
            except json.JSONDecodeError:
                pass
    return salida


def resumen(raiz):
    """Devuelve ``(total, discrepancias, aptas_reentrenamiento)``.

    ``discrepancias``: en cuántas el sistema no coincidió con el radiólogo (los errores).
    ``aptas_reentrenamiento``: cuántas provienen de imágenes nuevas y, por lo tanto, pueden
    usarse para reentrenar (las de ejemplos oficiales quedan excluidas).
    """
    registros = leer(raiz)
    discrepancias = sum(1 for r in registros if not r.get("coincide", True))
    aptas = sum(1 for r in registros if r.get("apta_reentrenamiento", True))
    return len(registros), discrepancias, aptas
