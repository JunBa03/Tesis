"""Grad-CAM++: mapa de calor de las regiones que más pesan en una observación.

Se toma el último mapa de activaciones convolucionales, se lo pondera con los
gradientes de la salida objetivo y el resultado positivo, interpolado al tamaño de la
imagen, marca las regiones influyentes. El mapa orienta la mirada; no es una
segmentación de la lesión.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.data.labels import OBSERVATIONS
from src.inference import preparar_entrada

# Determinismo: el backward de las convoluciones en cuDNN no es determinístico por
# defecto, así que el mismo Grad-CAM podía variar levemente entre una corrida y otra.
# Fijar estos flags hace que un mismo estudio produzca siempre el mismo mapa.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _capa_objetivo(clasificador):
    """Última capa convolucional según la arquitectura.

    Para DenseNet-121 es el último bloque denso; para ResNet-50, el último bloque
    residual. En ambos casos entrega un mapa espacial (7x7) antes del pooling global,
    que es lo que Grad-CAM necesita.

    Se apunta al ``denseblock4`` y no a la ``norm5`` posterior a propósito: DenseNet
    aplica ``relu(inplace=True)`` sobre la salida de ``norm5``, y un gancho de
    retropropagación sobre un tensor modificado in situ produce gradientes inválidos.
    La salida del bloque denso no sufre esa modificación.
    """
    net = clasificador.modelo.features
    if clasificador.backbone == "densenet121":
        return net.features.denseblock4
    if clasificador.backbone == "resnet50":
        return net.layer4
    raise ValueError(f"Grad-CAM no soportado para backbone {clasificador.backbone!r}")


class GradCAM:
    """Genera mapas de calor para un clasificador ya cargado.

    Se construye una vez por modelo y se reutiliza. Registra dos ganchos sobre la
    capa objetivo (uno para guardar las activaciones y otro para los gradientes)
    y los libera con :meth:`cerrar`.
    """

    def __init__(self, clasificador) -> None:
        self.clf = clasificador
        self.capa = _capa_objetivo(clasificador)
        self._activaciones: torch.Tensor | None = None
        self._gradientes: torch.Tensor | None = None
        self._h1 = self.capa.register_forward_hook(self._guardar_activaciones)
        self._h2 = self.capa.register_full_backward_hook(self._guardar_gradientes)

    def _guardar_activaciones(self, _modulo, _entrada, salida) -> None:
        # clone() para que una eventual modificación in situ aguas abajo no altere
        # el valor guardado.
        self._activaciones = salida.detach().clone()

    def _guardar_gradientes(self, _modulo, _grad_entrada, grad_salida) -> None:
        self._gradientes = grad_salida[0].detach()

    def generar(self, imagen, observacion: str) -> np.ndarray:
        """Mapa de calor normalizado [0, 1] para una observación sobre una imagen.

        Devuelve un arreglo 2D del tamaño de la entrada del modelo (224x224), donde
        1 es la zona de máxima influencia y 0 la de mínima.
        """
        if observacion not in OBSERVATIONS:
            raise ValueError(f"Observación desconocida: {observacion!r}")
        idx = OBSERVATIONS.index(observacion)

        x = self.clf.transform(preparar_entrada(imagen)).unsqueeze(0).to(self.clf.device)

        # Grad-CAM necesita gradientes: no puede correr bajo torch.no_grad.
        self.clf.modelo.zero_grad(set_to_none=True)
        logits = self.clf.modelo(x)
        objetivo = logits[0, idx]
        objetivo.backward()

        A = self._activaciones                      # [1, C, h, w] activaciones
        g = self._gradientes                        # [1, C, h, w] gradientes

        # Ponderación estilo Grad-CAM++ (Chattopadhyay et al., 2018): pesa cada posición
        # por su relevancia de segundo orden, lo que produce mapas más nítidos y mejor
        # localizados que el Grad-CAM clásico, sobre todo cuando hay varias regiones.
        g2 = g * g
        g3 = g2 * g
        suma_A = A.sum(dim=(2, 3), keepdim=True)     # sum espacial de activaciones
        denom = 2 * g2 + suma_A * g3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = g2 / denom
        pesos = (alpha * F.relu(g)).sum(dim=(2, 3), keepdim=True)  # [1, C, 1, 1]

        mapa = F.relu((pesos * A).sum(dim=1, keepdim=True))
        # Tamaño canónico fijo (no el de la entrada) para que mapas de modelos con
        # distinta resolución (por ejemplo 224 y 320) se puedan promediar entre sí.
        mapa = F.interpolate(mapa, size=(224, 224), mode="bilinear",
                             align_corners=False)[0, 0]

        mapa = mapa - mapa.min()
        maximo = mapa.max()
        if maximo > 0:
            mapa = mapa / maximo
        return mapa.detach().cpu().numpy()

    def cerrar(self) -> None:
        """Libera los ganchos. Conviene llamarlo al descartar el objeto."""
        self._h1.remove()
        self._h2.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *_exc) -> None:
        self.cerrar()
