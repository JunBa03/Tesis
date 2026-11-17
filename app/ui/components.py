"""Componentes visuales reutilizables: barras de probabilidad, tarjetas y chips.

Cada función devuelve o dibuja una pieza que varias páginas comparten, para no
repetir el mismo HTML en cada pantalla.
"""
from __future__ import annotations

import streamlit as st

from src.data.labels import COMPETITION_TASKS, OBSERVATIONS

from . import theme
from app.core import registry


def veredicto(pred) -> None:
    """Resumen por umbral: qué observaciones señala el modelo, si es que alguna.

    Un modelo multilabel devuelve una probabilidad para cada observación, y muchas
    quedan en valores intermedios. Mostrar todas esas barras da la impresión falsa de
    que el modelo "diagnostica de todo". La lectura correcta es por umbral: una
    observación se considera señalada solo si supera el umbral calibrado para ella. El
    resto son probabilidades por debajo del punto de decisión, no hallazgos.
    """
    considerar = [o for o in OBSERVATIONS if o not in ("No Finding", "Support Devices")]
    senaladas = sorted(
        ((o, pred.probabilidades[o]) for o in considerar
         if pred.probabilidades[o] >= pred.umbral(o)),
        key=lambda kv: kv[1], reverse=True)
    nf = pred.probabilidades.get("No Finding")
    ref = f' Probabilidad de "sin hallazgos": {nf:.2f}.' if nf is not None else ""

    if not senaladas:
        st.markdown(
            f'<div class="veredicto v-limpio"><b>Sin hallazgos por encima del umbral.</b>'
            f' El modelo no señala ninguna observación como positiva para esta imagen.'
            f'{ref}</div>', unsafe_allow_html=True)
    else:
        lista = ", ".join(f"{o} ({p:.2f})" for o, p in senaladas)
        st.markdown(
            f'<div class="veredicto v-hallazgo"><b>Observaciones señaladas '
            f'({len(senaladas)}):</b> {lista}.{ref}</div>', unsafe_allow_html=True)


def _color_barra(prob: float, umbral: float, destacado: bool) -> str:
    if prob >= umbral:
        return theme.AMBAR
    return "#cbd5e1" if destacado else "#9ca3af"


def fila_observacion(obs: str, prob: float, umbral: float,
                     verdad: dict | None = None) -> str:
    """HTML de una fila: nombre, barra y valor. Resalta las patologías del benchmark."""
    bench = obs in COMPETITION_TASKS
    etiqueta = f'<span class="nombre {"bench" if bench else ""}">{registry.es(obs)}</span>'
    if verdad is not None and obs in verdad and verdad[obs] in (0.0, 1.0):
        marca = "positivo" if verdad[obs] == 1.0 else "negativo"
        etiqueta += f' <span class="meta">· radiólogos: {marca}</span>'
    barra = (f'<div class="barra"><div style="width:{prob*100:.1f}%;'
             f'background:{_color_barra(prob, umbral, bench)}"></div></div>')
    return (f'<div class="obs"><div style="flex:1">{etiqueta}{barra}</div>'
            f'<div class="valor" style="width:64px;text-align:right">{prob:.3f}</div></div>')


def panel_observaciones(pred, titulo: str, observaciones: list[str],
                        verdad: dict | None = None) -> None:
    """Bloque con una fila por observación."""
    st.markdown(f"**{titulo}**")
    filas = "".join(
        fila_observacion(o, pred.probabilidades[o], pred.umbral(o), verdad)
        for o in observaciones
    )
    st.markdown(f'<div class="bloque">{filas}</div>', unsafe_allow_html=True)


def chip(texto: str, color: str) -> str:
    """Etiqueta redondeada de color (para el eje o el nivel de urgencia)."""
    return f'<span class="chip" style="background:{color}">{texto}</span>'


def tarjeta_modelo(exp, extra: str = "") -> str:
    """Tarjeta con los metadatos de una configuración."""
    return (
        f'<div class="tarjeta-modelo">'
        f'<h4>{chip(exp.id, exp.color)} &nbsp;{exp.nombre}</h4>'
        f'<div class="meta">{exp.backbone} · {exp.incertidumbre} · {exp.perdida}</div>'
        f'{extra}</div>'
    )
