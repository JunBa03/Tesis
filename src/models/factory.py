"""Construccion de los backbones convolucionales con cabeza multilabel de 14 salidas."""
from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torchvision import models

from ..data.labels import OBSERVATIONS

logger = logging.getLogger("chexpert")

NUM_CLASSES = len(OBSERVATIONS)


class MultiLabelClassifier(nn.Module):
    """Backbone preentrenado + dropout + capa lineal de 14 salidas.

    La red devuelve LOGITS, no probabilidades: la sigmoide se aplica dentro de la
    funcion de perdida (BCEWithLogits / focal) por estabilidad numerica, y de forma
    explicita en inferencia. Cada salida es independiente de las demas, que es lo
    que corresponde a un problema multilabel donde varios hallazgos coexisten.
    """

    def __init__(self, backbone: str = "densenet121", pretrained: bool = True,
                 dropout: float = 0.2, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = num_classes

        if backbone == "densenet121":
            weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            net = models.densenet121(weights=weights)
            in_features = net.classifier.in_features
            net.classifier = nn.Identity()
            self.features = net
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = models.resnet50(weights=weights)
            in_features = net.fc.in_features
            net.fc = nn.Identity()
            self.features = net
        else:
            raise ValueError(f"Backbone no soportado: {backbone!r}")

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        self.in_features = in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        return self.head(feats)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def build_model(cfg) -> MultiLabelClassifier:
    model = MultiLabelClassifier(
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
    )
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Modelo | %s (pretrained=%s) | %d parametros entrenables | %d salidas",
        cfg["model"]["backbone"], cfg["model"]["pretrained"], n_params, NUM_CLASSES,
    )
    return model
