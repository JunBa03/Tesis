"""Selector de imagen compartido por las pantallas, con estado persistente.

Todas las pantallas eligen la radiografía con esta misma función. Al usar claves de
``session_state`` idénticas, Streamlit conserva la selección al cambiar de página: si en
Análisis se elige un estudio, al pasar a Comparación por modelo sigue seleccionado el
mismo, sin volver a elegirlo.
"""
from __future__ import annotations

from PIL import Image

import streamlit as st

from . import loaders
from app.core import dicom


def abrir_imagen(archivo):
    """Abre un archivo subido como imagen o devuelve ``None`` si no es una imagen válida.

    ``Image.open`` solo lee el encabezado; ``load()`` fuerza la decodificación completa para
    detectar acá un archivo dañado o con extensión falsa, y no más adelante en el modelo.
    """
    try:
        imagen = Image.open(archivo)
        imagen.load()
        return imagen
    except Exception:
        return None


LADO_MINIMO = 256


def avisar_resolucion(imagen) -> None:
    """Advierte si la imagen es demasiado chica para una lectura confiable.

    La prueba de robustez mostró que, reducida a 160 px de lado, la imagen cambia el hallazgo
    principal en 6 de cada 10 estudios. Por debajo de 256 px se avisa al usuario.
    """
    if min(imagen.size) < LADO_MINIMO:
        st.warning(f"La imagen tiene baja resolución ({imagen.size[0]}×{imagen.size[1]} px). "
                   f"Por debajo de {LADO_MINIMO} px la lectura es menos confiable; si es "
                   "posible, cargá la radiografía en su resolución original.")


def elegir_imagen(label_visible: bool = False):
    """Muestra el selector de origen/estudio y devuelve ``(imagen, verdad, nombre)``.

    ``imagen`` es un ``PIL.Image`` o ``None`` si todavía no se cargó nada. ``verdad`` son
    las etiquetas de consenso (solo para ejemplos oficiales) o ``None``.
    """
    ejemplos = loaders.ejemplos()
    tiene_ejemplos = bool(ejemplos)

    opciones = (["Ejemplo del conjunto oficial", "Subir un archivo"]
                if tiene_ejemplos else ["Subir un archivo"])
    origen = st.radio("Origen de la imagen", opciones, horizontal=True,
                      key="sel_origen")

    vis = "visible" if label_visible else "collapsed"
    if origen == "Subir un archivo":
        subido = st.file_uploader(
            "Radiografía de tórax frontal (JPG, PNG o DICOM)",
            type=["jpg", "jpeg", "png", "dcm", "dicom"],
            key="sel_upload", label_visibility=vis,
            help="Admite radiografías externas al dataset. La imagen se normaliza "
                 "automáticamente (orientación, escala de grises, contraste y profundidad "
                 "de bits) antes del análisis. Para resultados válidos debe ser una "
                 "radiografía de tórax en proyección frontal.")
        st.session_state["meta_estudio"] = None
        if subido is None:
            return None, None, None
        if dicom.es_dicom(subido.name):
            if not dicom.disponible():
                st.warning("Para leer archivos DICOM instalá `pydicom`. "
                           "Por ahora subí un JPG o PNG.")
                return None, None, None
            imagen, meta = dicom.leer(subido)
            if imagen is None:
                st.error("No se pudo leer el archivo DICOM.")
                return None, None, None
            st.session_state["meta_estudio"] = meta
            return imagen, None, subido.name
        imagen = abrir_imagen(subido)
        if imagen is None:
            st.error(f"No se pudo leer «{subido.name}» como imagen. Verificá que sea una "
                     "radiografía en JPG, PNG o DICOM y que el archivo no esté dañado.")
            return None, None, None
        avisar_resolucion(imagen)
        return imagen, None, subido.name

    st.session_state["meta_estudio"] = None
    etiquetas = [e["nombre"] for e in ejemplos]
    idx = st.selectbox("Estudio", range(len(etiquetas)),
                       format_func=lambda i: etiquetas[i],
                       key="sel_ejemplo", label_visibility=vis)
    e = ejemplos[idx]
    return Image.open(e["ruta"]), e.get("verdad"), e["nombre"]
