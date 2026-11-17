"""Punto de entrada de la demostración.

    streamlit run app/Inicio.py

Página de bienvenida: presenta el producto y deja los accesos a mano. Las pantallas de
trabajo viven en la carpeta ``pages/`` y Streamlit arma la navegación a partir de ella.
"""
from __future__ import annotations

# Streamlit pone la carpeta del script (app/) en sys.path, no la raíz del repo. Para
# poder importar los paquetes app.* y src.* hay que agregar la raíz antes que nada.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ui import theme

import streamlit as st

theme.aplicar("RXTórax — Análisis de radiografías de tórax", sidebar="expanded")

# Centra verticalmente el contenido y quita el recuadro flotante, solo en esta página.
st.markdown(
    "<style>"
    ".block-container{min-height:84vh;display:flex;flex-direction:column;"
    "justify-content:center;padding-top:2rem;padding-bottom:2rem;}"
    "[data-testid='stVerticalBlockBorderWrapper']{box-shadow:none !important;"
    "border:none !important;background:transparent !important;}"
    "</style>",
    unsafe_allow_html=True)

# --- bienvenida
st.markdown(
    '<div class="landing">'
    '<div style="margin:0 auto 26px;width:104px;'
    'filter:drop-shadow(0 10px 26px rgba(37,99,235,0.28))">'
    '<svg width="104" height="104" viewBox="0 0 64 64" style="display:block">'
    '<rect x="4" y="4" width="56" height="56" rx="15" fill="#111a2b" '
    'stroke="#2f6bd8" stroke-width="1.6"/>'
    '<g stroke="#5b9bff" stroke-width="2.4" fill="none" stroke-linecap="round">'
    '<line x1="32" y1="17" x2="32" y2="40"/>'
    '<path d="M32 23 Q 44 24 45 33"/><path d="M32 30 Q 45 31 46 40"/>'
    '<path d="M32 23 Q 20 24 19 33"/><path d="M32 30 Q 19 31 18 40"/></g>'
    '<path d="M14 46 h9 l3 -6 4 12 3 -6 h17" stroke="#5b9bff" stroke-width="2.2" '
    'fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg></div>'
    '<div class="welcome-tag">Bienvenido</div>'
    '<h1 class="hero-title">RXTórax</h1>'
    '<div class="hero-tagline">Apoyo a la lectura y priorización de radiografías '
    'de tórax</div>'
    '<div class="hero-desc">Un asistente de investigación que estima hallazgos, prioriza '
    'la cola de estudios y señala regiones de interés sobre la imagen.</div>'
    '</div>',
    unsafe_allow_html=True)

st.write("")
st.markdown('<div class="landing-label">Explorar</div>', unsafe_allow_html=True)

# --- accesos (degradan a una pista si la versión de Streamlit no los soporta)
_accesos = [
    ("pages/1_Analisis.py", "Análisis"),
    ("pages/2_Priorizacion.py", "Priorización"),
    ("pages/3_Comparacion_por_modelo.py", "Comparación"),
]
try:
    _cols = st.columns(3, gap="medium")
    for _col, (_destino, _etiqueta) in zip(_cols, _accesos):
        _col.page_link(_destino, label=_etiqueta, use_container_width=True)
except Exception:
    st.markdown(
        '<div class="hero-hint">Elegí una sección en el menú de la izquierda '
        'para comenzar.</div>', unsafe_allow_html=True)

# --- pie sobrio
st.write("")
st.markdown(
    '<div class="landing-foot">Prueba de concepto · Proyecto Final de Ingeniería · '
    'Juan Bautista Felizia<br>Herramienta de investigación sobre CheXpert — no es un '
    'producto clínico ni reemplaza la lectura del especialista.</div>',
    unsafe_allow_html=True)
