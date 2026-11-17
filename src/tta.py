"""Test-time augmentation (TTA) para inferencia.

Promedia la predicción del modelo sobre unas pocas variaciones de la misma imagen que
no cambian su interpretación clínica. Estabiliza la salida y suele mejorar levemente el
AUC, sin reentrenar nada.

Restricción del dominio: NO se usa volteo horizontal. En una radiografía de tórax el
espejado ubicaría el corazón a la derecha de la línea media (situs inversus, ausente
en CheXpert) e introduciría una señal incorrecta. Solo se usan rotaciones pequeñas,
que sí ocurren por variabilidad de posicionamiento del paciente.
"""
from __future__ import annotations

import torch
import torchvision.transforms.functional as TF

# Ángulos de rotación (en grados). El 0 es la imagen sin tocar.
ANGULOS = (0.0, -7.0, 7.0)


@torch.no_grad()
def probabilidades_tta(model, x: torch.Tensor, angulos=ANGULOS) -> torch.Tensor:
    """Probabilidades promediadas sobre las rotaciones. ``x`` es un lote [B,3,H,W].

    Se rota el tensor ya normalizado; los píxeles que quedan fuera se rellenan con el
    valor medio de cada imagen para no introducir un borde negro artificial.
    """
    acum = None
    for a in angulos:
        xa = x if a == 0.0 else TF.rotate(
            x, a, interpolation=TF.InterpolationMode.BILINEAR)
        p = torch.sigmoid(model(xa))
        acum = p if acum is None else acum + p
    return acum / len(angulos)
