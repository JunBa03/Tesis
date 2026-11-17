"""Comparación por modelo: la misma imagen a través de cada configuración.

Cada configuración difiere del baseline en una sola variable, de modo que la
diferencia de probabilidad entre ellas muestra el efecto aislado de esa decisión.
"""
from __future__ import annotations

# Raíz del repo en sys.path antes de importar app.* / src.* (ver Inicio.py).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image

from app.ui import theme, loaders, components, entrada
from app.core import registry, ablation, overlay, anatomia

import streamlit as st

from src.data.labels import COMPETITION_TASKS, OBSERVATIONS

theme.aplicar("Comparación por modelo — RXTórax")
theme.encabezado("Comparación por modelo · comportamiento individual")
theme.aviso()
st.markdown(
    '<div class="meta" style="max-width:820px;margin-bottom:10px">La misma radiografía, '
    'evaluada por cada configuración por separado: el comportamiento individual de los '
    'modelos que la pantalla de análisis combina en una sola respuesta. Como cada '
    'configuración difiere del baseline en una única variable, la diferencia entre dos es '
    'atribuible a esa variable.</div>', unsafe_allow_html=True)

disponibles = loaders.modelos_disponibles()
if len(disponibles) < 2:
    st.warning("Se necesitan al menos dos configuraciones para comparar. "
               f"Hay {len(disponibles)} cargada(s).")
    st.stop()

# --- elegir imagen (selector compartido: persiste entre páginas)
sel_col, _ = st.columns([1.3, 1], gap="large")
with sel_col:
    imagen, verdad, _nombre = entrada.elegir_imagen()

if imagen is None:
    st.info("Elegí un ejemplo o subí una radiografía.")
    st.stop()

# --- Grad-CAM por configuración; su clasificador interno sirve para predecir.
gradcams = loaders.gradcams_disponibles()
clasificadores = {eid: g.clf for eid, g in gradcams.items()}
with st.spinner(f"Evaluando con {len(clasificadores)} configuraciones…"):
    predicciones = ablation.predecir_todos(clasificadores, imagen)

# Máscara del tórax para acotar los mapas (si torchxrayvision está disponible).
clave = hash(imagen.convert("L").resize((64, 64)).tobytes())
if st.session_state.get("_torax_clave_cmp") != clave:
    with st.spinner("Segmentando el tórax…"):
        st.session_state["_torax_mascara_cmp"] = anatomia.mascara_torax(imagen)
    st.session_state["_torax_clave_cmp"] = clave
mascara_torax = st.session_state.get("_torax_mascara_cmp")

# --- radiografía centrada + selector de observación (por defecto, la más probable)
col_img, col_obs = st.columns([1.3, 1], gap="large")
with col_img:
    st.image(imagen.convert("L"), use_container_width=True, caption="Estudio en análisis")
with col_obs:
    prob_media = {o: sum(p.probabilidades[o] for p in predicciones.values())
                  / len(predicciones) for o in COMPETITION_TASKS}
    default = max(prob_media, key=prob_media.get)
    obs = st.selectbox("Observación a comparar", OBSERVATIONS,
                       index=OBSERVATIONS.index(default), format_func=registry.es,
                       help="Por defecto, la patología del benchmark más probable en esta "
                            "imagen.")
    if verdad is not None and verdad.get(obs) in (0.0, 1.0):
        marca = "positivo" if verdad[obs] == 1.0 else "negativo"
        st.caption(f"Etiqueta de los radiólogos para {registry.es(obs)}: **{marca}**.")

# --- efecto por eje: tres columnas a lo ancho
st.markdown("#### Efecto por eje")
por_eje = ablation.comparar_observacion(predicciones, obs)
cols_eje = st.columns(len(por_eje) if por_eje else 1)
for col, (eje, filas) in zip(cols_eje, por_eje.items()):
    with col:
        st.markdown(f"**{eje}**")
        html = '<div class="bloque">'
        for f in filas:
            exp = registry.info(f.eid)
            delta = ("" if f.eid == "E1" else
                     f'<span class="meta"> · Δ {f.delta_vs_baseline:+.3f}</span>')
            html += (
                f'<div class="obs"><div style="flex:1">'
                f'{components.chip(exp.id, exp.color)} '
                f'<span class="nombre">{exp.nombre}</span>{delta}'
                f'<div class="barra"><div style="width:{f.probabilidad*100:.1f}%;'
                f'background:{exp.color}"></div></div></div>'
                f'<div class="valor" style="width:52px;text-align:right">'
                f'{f.probabilidad:.3f}</div></div>')
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

# --- mapa de calor de cada modelo por separado, para la observación elegida
st.divider()
st.subheader(f"Mapa de calor por modelo · {registry.es(obs)}")
st.markdown(
    '<div class="meta">Dónde mira cada configuración para esta observación. Permite '
    'ver que dos modelos pueden llegar a una probabilidad parecida atendiendo a zonas '
    'distintas, o coincidir en la región relevante.</div>', unsafe_allow_html=True)

por_fila = 3
items = list(gradcams.items())
for inicio in range(0, len(items), por_fila):
    fila = items[inicio:inicio + por_fila]
    cols = st.columns(por_fila)
    for col, (eid, g) in zip(cols, fila):
        exp = registry.info(eid)
        with st.spinner(f"{eid}…"):
            cam = g.generar(imagen, obs)
            vista = overlay.superponer(imagen, cam, alpha=0.5, lado=320,
                                       mascara=mascara_torax)
        p = predicciones[eid].probabilidades.get(obs, 0.0)
        col.markdown(components.chip(eid, exp.color) +
                     f' <b>{exp.nombre}</b> · {p:.3f}', unsafe_allow_html=True)
        col.image(vista, use_container_width=True)

# --- tabla resumen: todas las patologías del benchmark x configuración
st.divider()
st.subheader("Resumen: patologías del benchmark")
tabla = ablation.resumen_benchmark(predicciones)
ids = list(predicciones)
encabezado = "".join(
    f'<th style="padding:6px 10px;text-align:center">'
    f'{components.chip(e, registry.info(e).color)}</th>' for e in ids)
filas_html = ""
for obs in COMPETITION_TASKS:
    celdas = ""
    valores = tabla[obs]
    maximo = max(valores.values()) if valores else 0
    for e in ids:
        v = valores[e]
        peso = "700" if v == maximo and maximo > 0 else "400"
        celdas += (f'<td style="padding:6px 10px;text-align:center;'
                   f'font-variant-numeric:tabular-nums;font-weight:{peso}">{v:.3f}</td>')
    filas_html += (f'<tr><td style="padding:6px 10px;font-weight:600">'
                   f'{registry.es(obs)}</td>{celdas}</tr>')
st.markdown(
    f'<table style="border-collapse:collapse;width:100%;font-size:13.5px">'
    f'<tr><th style="text-align:left;padding:6px 10px">Patología</th>{encabezado}</tr>'
    f'{filas_html}</table>'
    '<div class="meta" style="margin-top:8px">En negrita, la configuración con mayor '
    'probabilidad para cada patología. No implica que sea la "correcta": es la lectura '
    'de cada modelo sobre esta imagen.</div>',
    unsafe_allow_html=True)
