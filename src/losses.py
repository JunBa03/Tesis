"""Funciones de perdida para clasificacion multilabel con desbalance de clases.

Las tres opciones evaluadas en el diseno experimental:
  - bce:          entropia cruzada binaria estandar (baseline, E1).
  - weighted_bce: BCE con pos_weight por clase = N_neg / N_pos (E4).
  - focal:        focal loss de Lin et al. (2017) (E5).

Todas aceptan una mascara por elemento para soportar U-Ignore en modo 'mask'.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("chexpert")


def _reduce(loss: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Promedia sobre los elementos activos de la mascara."""
    if mask is None:
        return loss.mean()
    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom


class MaskedBCEWithLogits(nn.Module):
    """BCE estandar. Con pos_weight se convierte en la version ponderada por frecuencia."""

    def __init__(self, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        return _reduce(loss, mask)


class MaskedFocalLoss(nn.Module):
    """Focal loss multilabel.

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    El factor (1 - p_t)^gamma reduce la contribucion de los ejemplos faciles -- los
    negativos abundantes que el clasificador ya resuelve con alta confianza -- y
    concentra el entrenamiento en los casos ambiguos. alpha reequilibra el peso
    relativo entre la clase positiva y la negativa.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        modulating = (1.0 - p_t).clamp(min=1e-6) ** self.gamma
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        loss = alpha_t * modulating * bce
        return _reduce(loss, mask)


def compute_pos_weight(labels: np.ndarray, mask: np.ndarray,
                       clip: float = 10.0) -> torch.Tensor:
    """pos_weight_c = N_neg_c / N_pos_c, acotado para evitar gradientes explosivos.

    Sin el clip, las observaciones con prevalencia muy baja producen pesos de dos
    ordenes de magnitud que desestabilizan el entrenamiento del resto de las clases.
    """
    pos = ((labels == 1.0) & (mask == 1.0)).sum(axis=0).astype(np.float64)
    neg = ((labels == 0.0) & (mask == 1.0)).sum(axis=0).astype(np.float64)
    weight = np.where(pos > 0, neg / np.maximum(pos, 1.0), 1.0)
    weight = np.clip(weight, 1.0, clip)
    return torch.tensor(weight, dtype=torch.float32)


def build_loss(cfg, train_labels: Optional[np.ndarray] = None,
               train_mask: Optional[np.ndarray] = None) -> nn.Module:
    name = cfg["loss"]["name"]

    if name == "bce":
        return MaskedBCEWithLogits()

    if name == "weighted_bce":
        if train_labels is None or train_mask is None:
            raise ValueError("weighted_bce requiere las etiquetas de entrenamiento para pos_weight")
        pw = compute_pos_weight(train_labels, train_mask, clip=cfg["loss"]["pos_weight_clip"])
        logger.info("pos_weight por clase: %s", np.round(pw.numpy(), 2).tolist())
        return MaskedBCEWithLogits(pos_weight=pw)

    if name == "focal":
        return MaskedFocalLoss(
            gamma=cfg["loss"]["focal_gamma"], alpha=cfg["loss"]["focal_alpha"]
        )

    raise ValueError(f"Perdida desconocida: {name!r}")
