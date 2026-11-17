"""Lectura de radiografías en formato DICOM (.dcm), el estándar clínico real.

Convierte el archivo DICOM en una imagen visible (aplicando la ventana radiológica y
corrigiendo la interpretación fotométrica) y extrae los metadatos del estudio (edad,
sexo, proyección). Es una dependencia opcional (``pydicom``): si no está instalada, las
funciones lo indican y la demo sigue aceptando JPG/PNG.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def es_dicom(nombre: str) -> bool:
    return nombre.lower().endswith((".dcm", ".dicom"))


def disponible() -> bool:
    try:
        import pydicom  # noqa: F401
        return True
    except Exception:
        return False


def _edad(valor) -> str | None:
    """Convierte 'PatientAge' tipo '045Y' a '45 años'."""
    s = str(valor).strip()
    if not s:
        return None
    digitos = "".join(c for c in s if c.isdigit())
    if not digitos:
        return None
    n = int(digitos)
    unidad = s[-1].upper() if s[-1].isalpha() else "Y"
    nombre = {"Y": "años", "M": "meses", "W": "semanas", "D": "días"}.get(unidad, "años")
    return f"{n} {nombre}"


def leer(archivo):
    """Devuelve ``(PIL.Image en escala de grises, dict de metadatos)`` o ``(None, None)``.

    Aplica la LUT de visualización (ventana) si está presente, invierte las imágenes
    MONOCHROME1 (donde el hueso es oscuro) y normaliza a 8 bits.
    """
    try:
        import pydicom
        from pydicom.pixels import apply_voi_lut
    except Exception:
        try:
            from pydicom.pixel_data_handlers.util import apply_voi_lut  # versiones previas
        except Exception:
            return None, None
    try:
        ds = pydicom.dcmread(archivo, force=True)
        arr = ds.pixel_array
        try:
            arr = apply_voi_lut(arr, ds)
        except Exception:
            pass
        arr = arr.astype(np.float32)
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            arr = arr.max() - arr
        arr = arr - arr.min()
        maximo = arr.max()
        if maximo > 0:
            arr = arr / maximo
        imagen = Image.fromarray((arr * 255).astype(np.uint8)).convert("L")

        sexo = {"M": "masculino", "F": "femenino"}.get(
            str(getattr(ds, "PatientSex", "")).strip().upper())
        meta = {
            "sexo": sexo,
            "edad": _edad(getattr(ds, "PatientAge", "")),
            "proyeccion": str(getattr(ds, "ViewPosition", "")).strip() or None,
            "modalidad": str(getattr(ds, "Modality", "")).strip() or None,
        }
        meta = {k: v for k, v in meta.items() if v}
        return imagen, meta
    except Exception:
        return None, None
