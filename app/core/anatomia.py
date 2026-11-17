"""Segmentación anatómica del tórax para acotar el mapa de calor.

Usa el modelo de segmentación pre-entrenado de ``torchxrayvision`` (PSPNet sobre
CheXpert-Det), que separa 14 estructuras. La máscara de tórax es la unión de todo el
interior torácico (campos pulmonares, corazón, mediastino, hilios, aorta, diafragma,
columna) y EXCLUYE la cintura escapular (clavículas y escápulas). Así se elimina el
foco espurio sobre hombros y marcadores sin recortar el corazón ni el mediastino, que son
la región relevante para patologías como la cardiomegalia.

Es una dependencia opcional: si ``torchxrayvision`` no está instalado o el modelo falla,
las funciones devuelven ``None`` y el resto del sistema cae a la máscara de cuerpo.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

_modelo = None
_disponible = None

# Estructuras a excluir de la máscara de tórax: la cintura escapular (hombros), que es
# donde el Grad-CAM suele encender focos falsos.
_EXCLUIR = ("Clavicle", "Scapula")


def disponible() -> bool:
    global _disponible
    if _disponible is None:
        try:
            import torchxrayvision  # noqa: F401
            _disponible = True
        except Exception:
            _disponible = False
    return _disponible


def _cargar():
    global _modelo
    if _modelo is None:
        import torchxrayvision as xrv
        _modelo = xrv.baseline_models.chestx_det.PSPNet()
        _modelo.eval()
    return _modelo


def mascara_torax(imagen: Image.Image, lado: int = 224):
    """Máscara booleana ``lado x lado`` del tórax completo, o ``None`` si no se puede.

    Une todas las estructuras torácicas internas y descarta clavículas y escápulas.
    """
    if not disponible():
        return None
    try:
        import torch
        import torchxrayvision as xrv

        modelo = _cargar()
        gris = np.array(imagen.convert("L").resize((512, 512))).astype(np.float32)
        x = xrv.datasets.normalize(gris, 255)
        x = torch.from_numpy(x)[None, None]
        with torch.no_grad():
            pred = modelo(x).sigmoid()[0].cpu().numpy()

        objetivos = list(getattr(modelo, "targets", []))
        idx = [i for i, t in enumerate(objetivos)
               if not any(e in t for e in _EXCLUIR)]
        if not idx:
            return None
        torax = np.zeros((512, 512), dtype=bool)
        for i in idx:
            torax |= pred[i] > 0.5
        if not torax.any():
            return None
        m = np.array(
            Image.fromarray((torax * 255).astype(np.uint8)).resize((lado, lado))
        ) > 127
        return m if m.any() else None
    except Exception:
        return None
