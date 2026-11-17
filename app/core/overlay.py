"""Convierte un mapa de calor en una imagen superpuesta sobre la radiografía.

Sin dependencias de graficación: la paleta se calcula con numpy para no depender de
matplotlib en la aplicación. La paleta imita a
``jet`` (azul para lo irrelevante, rojo para lo influyente), que es la convención
habitual en los mapas de saliencia de imágenes médicas.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _paleta_jet(v: np.ndarray) -> np.ndarray:
    """Mapea valores en [0, 1] a colores RGB estilo jet. Entrada 2D, salida HxWx3."""
    v = np.clip(v, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def _mascara_cuerpo(gris: np.ndarray, lado: int) -> np.ndarray:
    """Máscara del cuerpo: descarta el fondo negro y un marco delgado en el borde.

    Grad-CAM se calcula sobre un mapa de 7x7 e interpolado puede encender esquinas y
    bordes que no son anatomía (artefactos del relleno de la convolución). Restringir
    el calor a las zonas con señal (el tórax es más brillante que el fondo) y
    suprimir un marco perimetral elimina esos falsos focos fuera del cuerpo.
    """
    cuerpo = gris > 0.06
    marco = max(1, int(lado * 0.03))
    cuerpo[:marco, :] = False
    cuerpo[-marco:, :] = False
    cuerpo[:, :marco] = False
    cuerpo[:, -marco:] = False
    return cuerpo


def _roi_torax(lado: int) -> np.ndarray:
    """Ventana suave centrada en los campos pulmonares.

    Los focos falsos de Grad-CAM aparecen casi siempre en los hombros, las clavículas,
    las esquinas superiores y sobre el marcador de lateralidad (todo en la periferia y
    la banda superior de la imagen). Esta ventana atenúa esas zonas y realza el centro
    del tórax, sin anular del todo la periferia para no borrar hallazgos apicales
    genuinos si son intensos.
    """
    ejes = (np.arange(lado) + 0.5) / lado
    yy, xx = np.meshgrid(ejes, ejes, indexing="ij")
    # coseno alzado: 1 en el centro, 0 a partir de cierto radio horizontal/vertical.
    wx = np.cos(np.clip((xx - 0.50) / 0.52, -1, 1) * (np.pi / 2))
    wy = np.cos(np.clip((yy - 0.54) / 0.50, -1, 1) * (np.pi / 2))
    roi = np.clip(wx * wy, 0, 1)
    return 0.12 + 0.88 * roi  # piso 0.12 para no anular la periferia por completo


def superponer(imagen: Image.Image, mapa: np.ndarray,
               alpha: float = 0.5, lado: int = 512,
               foco: float = 0.45, mascara=None) -> Image.Image:
    """Devuelve la radiografía con el mapa de calor encima.

    ``alpha`` regula la opacidad del calor; ``lado`` es el tamaño de salida; ``foco``
    descarta las activaciones débiles (por debajo de esa fracción del máximo) para que
    solo se pinte la región dominante. ``mascara`` es una máscara booleana opcional (por
    ejemplo la de los campos pulmonares) a la que se restringe el calor; si no se pasa,
    se usa la máscara de cuerpo (descartar el fondo y el marco). La radiografía se muestra
    en escala de grises, como en el negatoscopio.
    """
    base = imagen.convert("L").resize((lado, lado), Image.BILINEAR)
    gris = np.asarray(base).astype(np.float32) / 255.0
    base_rgb = np.stack([gris] * 3, axis=-1)

    mapa_img = np.asarray(
        Image.fromarray((np.clip(mapa, 0, 1) * 255).astype(np.uint8))
        .resize((lado, lado), Image.BILINEAR)
    ).astype(np.float32) / 255.0

    # Restringir el calor: a los pulmones si hay máscara de segmentación, o al cuerpo si no.
    if mascara is not None:
        m = np.asarray(
            Image.fromarray((np.asarray(mascara) > 0).astype(np.uint8) * 255)
            .resize((lado, lado), Image.NEAREST)
        ) > 127
    else:
        m = _mascara_cuerpo(gris, lado)
    mapa_img = mapa_img * m
    maximo = mapa_img.max()
    if maximo > 0:
        mapa_img = mapa_img / maximo

    # Enfocar: quedarse con la región fuerte. Todo lo que está por debajo de ``foco``
    # queda en cero; lo que está por encima se reescala a [0, 1]. Así el mapa marca un
    # foco claro en vez de teñir media imagen.
    mapa_img = np.clip((mapa_img - foco) / (1.0 - foco), 0.0, 1.0)

    calor = _paleta_jet(mapa_img)
    peso = mapa_img[..., None] * alpha
    mezcla = base_rgb * (1 - peso) + calor * peso
    return Image.fromarray((np.clip(mezcla, 0, 1) * 255).astype(np.uint8))


def combinar_mapas(mapas: list[np.ndarray]) -> np.ndarray:
    """Promedia varios mapas de calor (ya normalizados) en uno solo.

    Es la atención combinada del ensemble: donde varios modelos coinciden en mirar, el
    promedio se refuerza; donde solo uno enciende una zona, el promedio la atenúa. El
    resultado se vuelve a normalizar a [0, 1].
    """
    if not mapas:
        raise ValueError("No hay mapas para combinar.")
    prom = np.mean(np.stack(mapas, axis=0), axis=0)
    maximo = prom.max()
    return prom / maximo if maximo > 0 else prom


def unir_mapas(mapas: list[np.ndarray]) -> np.ndarray:
    """Une varios mapas de calor tomando el máximo punto a punto.

    A diferencia del promedio, el máximo preserva el foco de cada patología: cuando se
    superponen los mapas de dos o más observaciones, cada región caliente sigue
    visible en lugar de diluirse. El resultado se renormaliza a [0, 1].
    """
    if not mapas:
        raise ValueError("No hay mapas para unir.")
    m = np.maximum.reduce(mapas)
    maximo = m.max()
    return m / maximo if maximo > 0 else m


def escala_grises(imagen: Image.Image, lado: int = 512) -> Image.Image:
    """La radiografía en gris al mismo tamaño, para mostrarla al lado del mapa."""
    return imagen.convert("L").resize((lado, lado), Image.BILINEAR).convert("RGB")
