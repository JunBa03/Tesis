"""Asistente que explica el análisis actual, sobre la API de Grok (xAI).

Es una capa opcional de interpretabilidad: NO diagnostica ni agrega información nueva.
Solo explica, en lenguaje natural, la salida que el sistema ya calculó y los conceptos
del propio sistema (probabilidad, umbral, concordancia, derivación, mapa de calor).

Diseño:
- Lógica pura, sin Streamlit: recibe la clave y el contexto desde la capa de UI, que es
  la que sabe leer ``st.secrets``. Así el núcleo no depende de la interfaz.
- Nunca se envía la imagen a la nube: solo los números y el texto ya derivados.
- Candados por prompt de sistema: alcance acotado, sin diagnóstico, siempre deriva al
  profesional.
"""
from __future__ import annotations

import requests

from app.core import registry


def proveedor(clave, modelo_cfg=None):
    """Detecta el proveedor por el prefijo de la clave y devuelve (url, modelo).

    Las claves de Groq (groq.com, inferencia rápida) empiezan con ``gsk_``; las de xAI
    (Grok) con ``xai-``. Ambos exponen una API compatible con OpenAI. El modelo se puede
    forzar desde la configuración; si no, se usa un valor por defecto por proveedor.
    """
    m = (modelo_cfg or "").strip()
    if clave and clave.startswith("gsk_"):
        # Groq. Si el modelo configurado es de xAI (empieza con "grok"), lo ignoramos:
        # es un resto de plantilla y no existe en Groq. Usamos el default de Groq.
        if not m or m.startswith("grok"):
            m = "llama-3.3-70b-versatile"
        return ("https://api.groq.com/openai/v1/chat/completions", m)
    # xAI/Grok. Si el modelo configurado es de otro proveedor, usamos el default de xAI.
    if not m or m.startswith(("llama", "mixtral", "gemma", "openai/", "qwen")):
        m = "grok-2-latest"
    return ("https://api.x.ai/v1/chat/completions", m)

SISTEMA = (
    "Sos un asistente que EXPLICA la salida de RXTórax, un sistema de investigación que "
    "analiza radiografías de tórax. Tu único rol es aclarar, en español y de forma breve "
    "(2 a 5 oraciones), los resultados que se te pasan en el contexto y los conceptos del "
    "propio sistema: probabilidad, umbral calibrado, concordancia entre modelos, derivación "
    "selectiva y mapa de calor por Grad-CAM++.\n\n"
    "Reglas estrictas e inviolables:\n"
    "1. NO emitís diagnósticos ni recomendaciones clínicas, ni sugerís tratamientos.\n"
    "2. NO inventás hallazgos, cifras ni información que no esté en el contexto. Si algo no "
    "está, decí que el sistema no lo provee.\n"
    "3. Recordás, cuando corresponda, que es una herramienta de investigación —no de uso "
    "clínico— y que la decisión diagnóstica es del profesional.\n"
    "4. Si te preguntan algo fuera de este alcance (diagnosticar, interpretar otra imagen, "
    "dar conducta médica), lo aclarás con amabilidad y no lo respondés.\n"
    "5. Te apoyás siempre en los números del contexto; no los contradecís. Sé coherente con "
    "la recomendación: la observación que la motiva es la de mayor probabilidad entre las "
    "SEÑALADAS, que puede estar fuera de las 5 del benchmark. Si te preguntan por la "
    "probabilidad más alta, considerá TODAS las observaciones del contexto, no solo las 5 "
    "del benchmark."
)

ESTADO_ES = {
    "hallazgo": "revisar (hallazgo probable por encima del umbral)",
    "normal": "sin hallazgos (ninguna observación se acerca al umbral)",
    "derivar": "derivar al radiólogo (caso incierto: probabilidades en zona de decisión "
               "o desacuerdo entre modelos)",
}


def construir_contexto(pred, senaladas, rec, zona=None) -> str:
    """Arma el texto de contexto con la salida real del sistema (sin la imagen).

    ``senaladas`` son TODAS las observaciones (de las 14) que superan su umbral: son las
    que motivan la recomendación y pueden estar fuera de las 5 del benchmark. Es clave
    incluirlas para que el asistente no contradiga a la recomendación.
    """
    lineas = ["Salida del sistema para la radiografía actual (no es un diagnóstico):"]
    lineas.append(f"- Recomendación de triage: {ESTADO_ES.get(rec.estado, rec.estado)}.")

    if senaladas:
        lineas.append("- Observaciones SEÑALADAS (probabilidad por encima de su umbral; "
                      "son las que motivan la recomendación, y la más alta de estas es la "
                      "más relevante del análisis):")
        for obs, p in senaladas:
            lineas.append(f"    · {registry.es(obs)}: {p:.2f} (umbral {pred.umbral(obs):.2f}); "
                          f"concordancia {pred.nivel_confianza(obs)}")
    else:
        lineas.append("- Ninguna observación superó su umbral (estudio sin hallazgos "
                      "por encima del umbral).")

    if getattr(rec, "dudosas", None):
        lineas.append("- En zona de decisión (cerca del umbral o con desacuerdo entre "
                      "modelos): " + ", ".join(registry.es(o) for o in rec.dudosas))

    lineas.append("- Referencia: probabilidad de las 5 patologías del benchmark oficial "
                  "(pueden ser todas bajas aunque haya otra observación señalada):")
    for obs, p in pred.ordenadas(solo_benchmark=True):
        u = pred.umbral(obs)
        estado = "supera el umbral" if p >= u else "por debajo del umbral"
        lineas.append(f"    · {registry.es(obs)}: {p:.2f} (umbral {u:.2f}, {estado})")

    if zona:
        lineas.append(f"- Zona de mayor atención del mapa de calor: {zona} (es una pista de "
                      "atención, no una localización validada).")
    return "\n".join(lineas)


def responder(mensajes, contexto, clave, modelo, url, timeout=30) -> str:
    """Devuelve la respuesta del asistente. Lanza excepción si falla la conexión/clave."""
    payload = {
        "model": modelo,
        "temperature": 0.2,
        "messages": [
            {"role": "system",
             "content": SISTEMA + "\n\nCONTEXTO DEL ANÁLISIS ACTUAL:\n" + contexto},
            *mensajes,
        ],
    }
    r = requests.post(url, json=payload, timeout=timeout, headers={
        "Authorization": f"Bearer {clave}", "Content-Type": "application/json"})
    if r.status_code >= 400:
        # Superficie el motivo real (clave inválida, modelo inexistente, etc.).
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()["choices"][0]["message"]["content"].strip()
