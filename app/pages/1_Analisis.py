"""Pantalla de análisis: estación de lectura con respuesta única del conjunto.

El diseño sigue la convención de una estación de lectura radiológica: la imagen ocupa el
lugar principal a la izquierda y el informe estructurado va a la derecha. El sistema
procesa la radiografía con las seis configuraciones a la vez y entrega una sola
respuesta (el hallazgo más probable y un mapa de calor combinado). No hay panel de
configuración: la lectura es del conjunto, no de un modelo elegido a mano.
"""
from __future__ import annotations

# Raíz del repo en sys.path antes de importar app.* / src.* (ver Inicio.py).
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from PIL import Image

from app.ui import theme, loaders, components, entrada
from app.core import (ablation, overlay, registry, incertidumbre, informe,
                      derivacion, anatomia, zonas, correcciones, reporte, asistente)


def _cfg_grok():
    """Lee la clave y el modelo desde st.secrets o variables de entorno.

    Acepta claves de Groq (``gsk_``) o de xAI/Grok (``xai-``) y devuelve el endpoint y el
    modelo por proveedor. Ignora el valor de plantilla para no habilitar el panel en vano.
    """
    clave = modelo_cfg = None
    try:
        clave = (st.secrets.get("XAI_API_KEY") or st.secrets.get("GROQ_API_KEY")
                 or st.secrets.get("GROK_API_KEY"))
        modelo_cfg = st.secrets.get("GROK_MODEL") or st.secrets.get("GROQ_MODEL")
    except Exception:
        pass
    clave = (clave or os.environ.get("XAI_API_KEY") or os.environ.get("GROQ_API_KEY")
             or os.environ.get("GROK_API_KEY"))
    modelo_cfg = modelo_cfg or os.environ.get("GROK_MODEL") or os.environ.get("GROQ_MODEL")
    if not clave or clave.strip() in ("", "pega-aca-tu-api-key-de-grok"):
        return None, None, None
    url, modelo = asistente.proveedor(clave.strip(), modelo_cfg)
    return clave.strip(), url, modelo

import streamlit as st

from src.data.labels import COMPETITION_TASKS, OBSERVATIONS

theme.aplicar("Análisis — RXTórax")
theme.encabezado("Análisis · resultado combinado del conjunto")
theme.aviso()

disponibles = loaders.modelos_disponibles()
if not disponibles:
    st.error("No hay modelos cargados. Volvé a la página de inicio para ver cómo "
             "prepararlos.")
    st.stop()

gradcams = loaders.gradcams_disponibles()
clasificadores = {eid: g.clf for eid, g in gradcams.items()}
ids = list(clasificadores)

# ---------------------------------------------------------------- entrada
imagen, verdad_ejemplo, _nombre = entrada.elegir_imagen()

if imagen is None:
    st.info("Cargá una radiografía o elegí un ejemplo para ver el análisis.")
    st.stop()

# Reiniciar la conversación del asistente al cambiar de radiografía: cada estudio empieza
# con el chat limpio, para no seguir escribiendo sobre el análisis anterior.
if st.session_state.get("_chat_img") != _nombre:
    st.session_state["chat_msgs"] = []
    st.session_state["_chat_img"] = _nombre

# Metadatos del estudio DICOM, si los hay.
meta_estudio = st.session_state.get("meta_estudio")
if meta_estudio:
    partes = []
    if meta_estudio.get("proyeccion"):
        partes.append(f"proyección {meta_estudio['proyeccion']}")
    if meta_estudio.get("sexo"):
        partes.append(f"sexo {meta_estudio['sexo']}")
    if meta_estudio.get("edad"):
        partes.append(meta_estudio["edad"])
    if partes:
        st.caption("Estudio DICOM · " + " · ".join(partes))

# ---------------------------------------------------------------- procesamiento
cfg_ens = loaders.config_ensemble()
pesos = cfg_ens.get("pesos")
usar_tta = cfg_ens.get("usar_tta", False)
with st.spinner(f"Procesando con {len(ids)} configuraciones…"):
    pred = ablation.predecir_ensemble(clasificadores, imagen,
                                      pesos=pesos, tta=usar_tta,
                                      umbrales_operacion=cfg_ens.get("umbrales_operacion"))

if cfg_ens:
    metodo = cfg_ens.get("metodo", "promedio").replace("_", " ")
    tta_txt = " · TTA" if usar_tta else ""
    st.caption(f"Combinación: {metodo}{tta_txt} de las {len(ids)} configuraciones.")

considerar = [o for o in OBSERVATIONS if o not in ("No Finding", "Support Devices")]
senaladas = sorted(
    ((o, pred.probabilidades[o]) for o in considerar
     if pred.probabilidades[o] >= pred.umbral(o)),
    key=lambda kv: kv[1], reverse=True)
hay_hallazgo = bool(senaladas)

# Derivación selectiva: el sistema decide si responde o deriva al radiólogo. Se calcula
# acá (antes de las columnas) para que lo usen el informe, el asistente y la descarga.
rec = derivacion.evaluar(pred, senaladas, considerar)
resumen = incertidumbre.resumen_clinico(pred, senaladas)
metodo_txt = (cfg_ens.get("metodo", "promedio") + (" + TTA" if usar_tta else "")
              ).replace("_", " ")
if rec.estado == "hallazgo":
    recomendacion_txt = "Revisar: " + registry.es(rec.firmes[0][0])
elif rec.estado == "normal":
    recomendacion_txt = "Sin hallazgos (alta confianza)"
else:
    recomendacion_txt = "Derivar al radiólogo (caso incierto)"

st.write("")
visor, col_informe = st.columns([1.5, 1], gap="large")

# ---------------------------------------------------------------- visor
with visor:
    st.markdown("###### Visor")
    seleccion = []
    alpha = 0.55
    if hay_hallazgo:
        opciones_mapa = [o for o, _ in senaladas]
        seleccion = st.multiselect(
            "Mapa de calor de las patologías señaladas",
            opciones_mapa, default=[opciones_mapa[0]], format_func=registry.es,
            help="Agregá o quitá para ver una, varias, todas o ninguna.",
            placeholder="Elegí una o más patologías")
        alpha = st.slider("Intensidad del mapa de calor", 0.3, 0.9, 0.55, 0.05)

    # El mapa de calor sale de un único modelo fuerte (el de mayor resolución/AUC), no
    # del promedio de las siete arquitecturas: promediar mapas heterogéneos los emborrona.
    id_mapa = "E7" if "E7" in gradcams else max(
        gradcams, key=lambda e: registry.info(e).auc or 0)
    g_mapa = gradcams[id_mapa]

    # Máscara del tórax (segmentación anatómica), cacheada por imagen. Si torchxrayvision
    # no está instalado, queda en None y se cae a la máscara de cuerpo.
    clave = hash(imagen.convert("L").resize((64, 64)).tobytes())
    if st.session_state.get("_torax_clave") != clave:
        with st.spinner("Segmentando el tórax…"):
            st.session_state["_torax_mascara"] = anatomia.mascara_torax(imagen)
        st.session_state["_torax_clave"] = clave
    mascara_torax = st.session_state.get("_torax_mascara")

    zona = None
    if seleccion:
        with st.spinner("Generando el mapa de calor…"):
            cam = overlay.unir_mapas([g_mapa.generar(imagen, obs) for obs in seleccion])
            vista = overlay.superponer(imagen, cam, alpha=alpha, lado=680,
                                       mascara=mascara_torax)
            if mascara_torax is not None:
                zona = zonas.zona_dominante(cam, mascara_torax)
        pie = "Mapa de calor · " + ", ".join(registry.es(o) for o in seleccion)
    else:
        vista = overlay.escala_grises(imagen, lado=680)
        pie = ("Radiografía (sin hallazgos por encima del umbral)" if not hay_hallazgo
               else "Radiografía · ninguna patología seleccionada")

    with st.container(border=True):
        st.image(vista, use_container_width=True)

    if zona:
        st.markdown(
            f'<div class="meta"><b>Zona de atención:</b> {zona} '
            '<span style="opacity:0.7">(según la segmentación anatómica del tórax)</span>'
            '</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="meta">{pie}</div>', unsafe_allow_html=True)

    # Informe estructurado (necesita la zona del mapa) y botón de descarga, en esta columna.
    rep = reporte.construir(pred, senaladas, rec.estado, zona=zona, meta=meta_estudio)
    # verdad=None: el informe descargable no incluye la "condición real" (consenso de
    # radiólogos), para que sea igual para radiografías oficiales y externas.
    html_informe = informe.construir_html(
        imagen, vista, pred, senaladas, resumen, considerar,
        metodo=metodo_txt, verdad=None,
        recomendacion=recomendacion_txt, estado=rec.estado, reporte=rep)
    st.write("")
    st.download_button("Descargar informe (HTML)", data=html_informe,
                       file_name="informe_analisis.html", mime="text/html",
                       use_container_width=True)

    # Corrección del radiólogo (human-in-the-loop): solo con radiografías EXTERNAS. Las del
    # conjunto oficial ya tienen etiqueta y se autoevalúan, así que no se ofrece corregirlas.
    if verdad_ejemplo is None:
        with st.expander("Corrección del radiólogo (retroalimentación)"):
            st.markdown(
                '<div class="meta">Imagen nueva, sin etiqueta previa. Marcá las patologías '
                'realmente presentes: esa corrección de experto alimenta el reentrenamiento '
                'futuro, sin arriesgarse a aprender el ruido de las etiquetas automáticas.'
                '</div>', unsafe_allow_html=True)
            presentes = st.multiselect(
                "Patologías presentes según el radiólogo", considerar,
                default=[o for o, _ in senaladas], format_func=registry.es,
                key="correccion_presentes",
                placeholder="Seleccioná una o varias patologías")
            if st.button("Guardar corrección", use_container_width=True):
                reg = correcciones.registrar(
                    theme.RAIZ, _nombre or "imagen subida", pred.probabilidades,
                    pred.umbrales, presentes, considerar, recomendacion_txt,
                    origen="nueva")
                st.success("Corrección registrada para el reentrenamiento futuro.")
                # Devolución inmediata: en qué se equivocó el sistema respecto del radiólogo.
                omitidas = [o for o in reg["radiologo_presentes"]
                            if o not in reg["sistema_senalo"]]
                de_mas = [o for o in reg["sistema_senalo"]
                          if o not in reg["radiologo_presentes"]]
                acertadas = [o for o in reg["radiologo_presentes"]
                             if o in reg["sistema_senalo"]]
                if reg["coincide"]:
                    st.info("El sistema coincidió con el radiólogo en todas las patologías.")
                else:
                    filas = []
                    if acertadas:
                        filas.append("**Detectadas por el sistema:** "
                                     + ", ".join(registry.es(o) for o in acertadas))
                    if omitidas:
                        filas.append("**No detectadas (falsos negativos):** "
                                     + ", ".join(f"{registry.es(o)} "
                                                 f"({pred.probabilidades[o]:.0%})"
                                                 for o in omitidas))
                    if de_mas:
                        filas.append("**Señaladas de más (falsos positivos):** "
                                     + ", ".join(registry.es(o) for o in de_mas))
                    st.warning("  \n".join(filas))
            n_corr, discrep, aptas = correcciones.resumen(theme.RAIZ)
            if n_corr:
                s = "" if n_corr == 1 else "s"
                st.caption(f"{n_corr} registro{s} · {discrep} con desacuerdo · "
                           f"{aptas} de imágenes nuevas, apto{'' if aptas == 1 else 's'} "
                           "para reentrenar.")

# ---------------------------------------------------------------- informe
with col_informe:
    st.markdown("###### Informe")
    recomendacion_txt = ""
    if rec.estado == "hallazgo":
        top_obs, top_p = rec.firmes[0]
        ac = pred.acuerdo(top_obs)
        conc = pred.nivel_confianza(top_obs)
        otras = ", ".join(registry.es(o) for o, _ in rec.firmes[1:] + rec.senaladas
                          if o != top_obs)
        sub = (f"Probabilidad combinada {top_p:.2f} · concordancia {conc} · "
               f"{ac} de {len(ids)} modelos coinciden")
        if otras:
            sub += f"<br>También señaladas: {otras}"
        recomendacion_txt = f"Revisar: {registry.es(top_obs)}"
        st.markdown(
            f'<div class="resultado r-hallazgo"><div class="rot">Recomendación</div>'
            f'<div class="dx">Revisar · {registry.es(top_obs)}</div>'
            f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)
    elif rec.estado == "normal":
        nf = pred.probabilidades.get("No Finding")
        ref = f"Probabilidad de estudio sin hallazgos {nf:.2f}." if nf is not None else ""
        recomendacion_txt = "Sin hallazgos (alta confianza)"
        st.markdown(
            f'<div class="resultado r-limpio"><div class="rot">Recomendación</div>'
            f'<div class="dx">Sin hallazgos</div>'
            f'<div class="sub">Alta confianza: ninguna observación se acerca al umbral. '
            f'{ref}</div></div>', unsafe_allow_html=True)
    else:  # derivar
        motivos = ", ".join(registry.es(o) for o in rec.dudosas[:4]) or "señales límite"
        recomendacion_txt = "Derivar al radiólogo (caso incierto)"
        st.markdown(
            f'<div class="resultado r-derivar"><div class="rot">Recomendación</div>'
            f'<div class="dx">Derivar al radiólogo</div>'
            f'<div class="sub">Caso incierto: {motivos} en zona de decisión o con baja '
            f'concordancia entre modelos. El sistema se abstiene de decidir.</div></div>',
            unsafe_allow_html=True)

    # Informe estructurado con formato radiológico (Técnica/Hallazgos/Impresión/Recom.).
    rep = reporte.construir(pred, senaladas, rec.estado, zona=zona, meta=meta_estudio)
    st.markdown(reporte.a_html(rep), unsafe_allow_html=True)
    st.write("")

    resumen = incertidumbre.resumen_clinico(pred, senaladas)

    # La comparación con las etiquetas de radiólogos solo existe para ejemplos oficiales.
    if verdad_ejemplo is not None:
        ver_verdad = st.checkbox("Mostrar etiquetas de radiólogos en el detalle", value=True)
        verdad = verdad_ejemplo if ver_verdad else None
    else:
        verdad = None

    orden_bench = [o for o, _ in pred.ordenadas(solo_benchmark=True)]
    components.panel_observaciones(pred, "Patologías del benchmark", orden_bench, verdad)
    with st.expander("Ver las 14 observaciones"):
        rest = [o for o, _ in pred.ordenadas() if o not in COMPETITION_TASKS]
        components.panel_observaciones(pred, "Resto de las observaciones", rest, verdad)
    st.markdown(
        '<div class="meta">Probabilidad combinada (promedio de los seis modelos). La barra '
        'se pinta en ámbar cuando supera el umbral calibrado de esa patología; por debajo '
        'del umbral no es un hallazgo. La concordancia mide el acuerdo entre modelos.</div>',
        unsafe_allow_html=True)

    # La corrección del radiólogo se movió a la columna izquierda y solo aparece con
    # radiografías externas (ver el bloque en el visor).

# ---------------------------------------------------------------- asistente (a lo ancho)
st.write("")
clave_grok, url_grok, modelo_grok = _cfg_grok()
with st.container(border=True):
    st.markdown("###### Asistente · explicá este análisis")
    if not clave_grok:
        st.caption("Opcional. Cargá tu clave en `.streamlit/secrets.toml` (campo "
                   "`XAI_API_KEY`) para habilitar un asistente que explica este resultado "
                   "en lenguaje natural. No envía la imagen: solo los números del análisis.")
    else:
        st.session_state.setdefault("chat_msgs", [])
        _hist = st.container(height=280)
        _sug = ["¿Por qué la recomendación?", "¿Qué es la concordancia?",
                "¿Qué muestra el mapa?"]
        _click = None
        for _c, _s in zip(st.columns(len(_sug)), _sug):
            if _c.button(_s, key="sug_" + _s, use_container_width=True):
                _click = _s
        with st.form("form_chat", clear_on_submit=True):
            _fc1, _fc2 = st.columns([6, 1])
            _txt = _fc1.text_input("Pregunta", label_visibility="collapsed",
                                   placeholder="Preguntá sobre este resultado…")
            _enviar = _fc2.form_submit_button("Preguntar", use_container_width=True)
        _pregunta = _click or (_txt if (_enviar and _txt) else None)
        if _pregunta:
            st.session_state["chat_msgs"].append({"role": "user", "content": _pregunta})
            with st.spinner("Pensando…"):
                try:
                    _ctx = asistente.construir_contexto(pred, senaladas, rec, zona)
                    _resp = asistente.responder(st.session_state["chat_msgs"], _ctx,
                                                clave_grok, modelo_grok, url_grok)
                except Exception as _e:
                    _resp = ("No pude responder ahora (" + str(_e)[:160] + "). Revisá la "
                             "clave y el modelo; el asistente es opcional y el resto del "
                             "análisis funciona igual.")
            st.session_state["chat_msgs"].append({"role": "assistant", "content": _resp})
        with _hist:
            if not st.session_state["chat_msgs"]:
                st.caption("Preguntá sobre este resultado o tocá una de las sugerencias.")
            for _m in st.session_state["chat_msgs"]:
                st.chat_message(_m["role"]).write(_m["content"])
