"""Pantalla de priorización: la bandeja de estudios ordenada por urgencia.

Reproduce la función número uno que pidieron los médicos entrevistados: ante una cola de
estudios sin leer, ordenarlos para que los de mayor probabilidad de hallazgo grave se
atiendan primero. Se pueden cargar varios estudios propios (JPG, PNG o DICOM) o usar los
ejemplos oficiales. El sistema no diagnostica (solo reordena) pero actúa sobre el
problema descripto: bajo alta carga, la atención se reparte desigual.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from PIL import Image

from app.ui import theme, loaders, components, entrada
from app.core import registry, triage, dicom, ablation

import streamlit as st

from src.data.labels import COMPETITION_TASKS

theme.aplicar("Priorización de cola — RXTórax", sidebar="expanded")
theme.encabezado("Priorización · cola de estudios por urgencia")
theme.aviso()

disponibles = loaders.modelos_disponibles()
if not disponibles:
    st.error("No hay modelos cargados. Volvé a la página de inicio.")
    st.stop()

# ------------------------------------------------------------------ configuración
st.sidebar.title("Configuración")
ids = list(disponibles)
mejor = max(ids, key=lambda e: registry.info(e).auc or 0)
modo = st.sidebar.radio(
    "Modo de análisis", ["Un modelo", "Conjunto (todos combinados)"],
    help="Un modelo: usa una sola configuración. Conjunto: combina todos los modelos en "
         "una respuesta promedio, más robusta (es más lento porque corre todos).")
if modo == "Un modelo":
    elegido = st.sidebar.selectbox(
        "Modelo", ids, index=ids.index(mejor),
        format_func=lambda e: f"{e} · {registry.info(e).nombre}")
    clf = loaders.clasificador(str(disponibles[elegido]))
    _clasificadores = None
    etiqueta_modo = f"modelo {elegido}"
    clave_modo = elegido
else:
    clf = None
    _clasificadores = {eid: loaders.clasificador(str(ruta))
                       for eid, ruta in disponibles.items()}
    _pesos = loaders.config_ensemble().get("pesos")
    etiqueta_modo = f"conjunto ({len(_clasificadores)} modelos combinados)"
    clave_modo = "conjunto"
mostrar_verdad = st.sidebar.checkbox("Mostrar etiquetas de radiólogos", value=True,
                                     help="Solo disponible para los ejemplos oficiales.")

# ------------------------------------------------------------------ origen de estudios
ejemplos = loaders.ejemplos()
opciones = ["Subir estudios"] + (["Ejemplos oficiales"] if ejemplos else [])
origen = st.radio("Origen de los estudios", opciones, horizontal=True, key="prio_origen")

estudios = []
if origen == "Subir estudios":
    subidos = st.file_uploader(
        "Radiografías de tórax frontales (JPG, PNG o DICOM) · se pueden seleccionar varias",
        type=["jpg", "jpeg", "png", "dcm", "dicom"], accept_multiple_files=True,
        key="prio_upload")
    for f in subidos or []:
        if dicom.es_dicom(f.name):
            if not dicom.disponible():
                st.warning("Para leer DICOM instalá `pydicom`. Se omiten los .dcm.")
                continue
            img, _meta = dicom.leer(f)
            if img is None:
                st.warning(f"No se pudo leer {f.name}; se omite.")
                continue
        else:
            img = entrada.abrir_imagen(f)
            if img is None:
                st.warning(f"No se pudo leer {f.name} como imagen; se omite.")
                continue
        estudios.append({"nombre": f.name, "imagen": img, "verdad": None})
    if not estudios:
        st.info("Subí una o varias radiografías para armar la cola priorizada.")
        st.stop()
else:
    estudios = [{"nombre": e["nombre"], "imagen": Image.open(e["ruta"]),
                 "verdad": e.get("verdad")} for e in ejemplos]

# ------------------------------------------------------------------ construir la cola
# Se cachea en la sesión por (origen, modo, estudios) para no recalcular al filtrar.
clave = (origen, clave_modo, tuple(e["nombre"] for e in estudios), len(estudios))
if st.session_state.get("_cola_clave") != clave:
    barra = st.progress(0.0, text="Analizando estudios…")
    items = []
    umbrales_activos = clf.umbrales if clf is not None else {}
    for i, e in enumerate(estudios):
        if clf is not None:
            pred = clf.predecir(e["imagen"])
        else:
            pred = ablation.predecir_ensemble(_clasificadores, e["imagen"],
                                              pesos=_pesos, tta=False,
                                              umbrales_operacion=loaders.config_ensemble().get("umbrales_operacion"))
        if not umbrales_activos:
            umbrales_activos = pred.umbrales
        urgencia, motivo, prob = triage.puntaje(pred.probabilidades)
        items.append(triage.ItemCola(
            nombre=e["nombre"], imagen=e["imagen"], urgencia=urgencia, motivo=motivo,
            prob_motivo=prob, probabilidades=pred.probabilidades, verdad=e.get("verdad")))
        barra.progress((i + 1) / len(estudios),
                       text=f"Analizando estudios… {i+1}/{len(estudios)}")
    barra.empty()
    items.sort(key=lambda it: it.urgencia, reverse=True)
    st.session_state["_cola_items"] = items
    st.session_state["_cola_umbrales"] = umbrales_activos
    st.session_state["_cola_clave"] = clave
cola = st.session_state["_cola_items"]
umbrales_activos = st.session_state["_cola_umbrales"]

# ------------------------------------------------------------------ resumen + filtro
n = {"alta": 0, "media": 0, "baja": 0}
for it in cola:
    n[it.nivel] += 1
c1, c2, c3, c4 = st.columns(4)
c1.metric("Estudios", len(cola))
c2.metric("Prioridad alta", n["alta"])
c3.metric("Prioridad media", n["media"])
c4.metric("Prioridad baja", n["baja"])

fcol, dcol = st.columns([3, 1])
with fcol:
    niveles = st.multiselect("Mostrar", ["alta", "media", "baja"],
                             default=["alta", "media", "baja"],
                             format_func=lambda s: f"Prioridad {s}")
with dcol:
    # Espaciador que iguala la altura del rótulo "Mostrar", para que el botón quede
    # alineado con el recuadro del filtro de la izquierda y no arriba con la etiqueta.
    st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
    filas_csv = "nombre,urgencia,nivel,motivo,prob_motivo\n" + "\n".join(
        f'{it.nombre},{it.urgencia:.0f},{it.nivel},{registry.es(it.motivo)},'
        f'{it.prob_motivo:.3f}' for it in cola)
    st.download_button("Descargar cola (CSV)", data=filas_csv,
                       file_name="cola_priorizada.csv", mime="text/csv",
                       use_container_width=True)

st.caption(f"Cola ordenada por urgencia · {etiqueta_modo}")
st.write("")

# ------------------------------------------------------------------ worklist
visibles = [it for it in cola if it.nivel in niveles]
rango = 0
for it in cola:
    if it.nivel not in niveles:
        continue
    rango += 1
    color = theme.NIVEL_COLOR[it.nivel]
    izq, der = st.columns([1, 5])
    with izq:
        st.image(it.imagen.convert("L"), use_container_width=True)
    with der:
        st.markdown(
            f'<div class="cola-item" style="border-left-color:{color}">'
            f'<div style="flex:1">'
            f'<b>#{rango} · {it.nombre}</b>'
            f'<div class="meta">Motivo: {registry.es(it.motivo)} '
            f'(p = {it.prob_motivo:.3f})</div>'
            f'<div class="barra" style="max-width:260px;margin-top:6px">'
            f'<div style="width:{min(it.urgencia,100):.0f}%;background:{color}"></div></div>'
            f'</div>'
            f'<div style="text-align:right;min-width:70px">'
            f'{components.chip(it.nivel.upper(), color)}'
            f'<div class="valor" style="font-size:20px">{it.urgencia:.0f}</div>'
            f'<div class="meta">urgencia</div></div></div>',
            unsafe_allow_html=True)
        with st.expander("Ver observaciones del estudio"):
            class _P:  # adaptador mínimo para el panel de observaciones
                probabilidades = it.probabilidades
                umbral = staticmethod(lambda o: umbrales_activos.get(o, 0.5))
            orden = sorted(COMPETITION_TASKS,
                           key=lambda o: it.probabilidades.get(o, 0), reverse=True)
            verdad = it.verdad if mostrar_verdad else None
            components.panel_observaciones(_P(), "Patologías del benchmark", orden, verdad)

if not visibles:
    st.info("Ningún estudio en los niveles seleccionados.")

st.divider()
st.markdown(
    '<div class="meta">El puntaje de urgencia es una demostración del criterio, no una '
    'escala clínica validada. Pondera las patologías graves (neumotórax, edema, '
    'consolidación, derrame) por encima de los hallazgos crónicos estables, para no '
    'demorar la lectura de un hallazgo grave.</div>',
    unsafe_allow_html=True)
