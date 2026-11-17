"""Utilidades generales: semillas, dispositivo, logging y serializacion."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# torch se importa de forma diferida dentro de cada funcion que lo necesita, para
# que los scripts de preparacion de datos y verificacion puedan correr sin GPU.
if TYPE_CHECKING:  # pragma: no cover
    import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fija las semillas de todas las fuentes de aleatoriedad del pipeline.

    La reproducibilidad es un requisito del diseno experimental: las seis
    configuraciones deben diferir en una unica variable, no en la inicializacion.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_device(prefer: str | None = None) -> "torch.device":
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("chexpert")
    logger.setLevel(level)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", "%H:%M:%S")

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _limpiar_nan(obj: Any) -> Any:
    """Convierte NaN e infinitos en None de forma recursiva.

    Python escribe NaN/Infinity en el JSON, pero no son JSON valido: cualquier
    lector estricto (por ejemplo JSON.parse) los rechaza. Los valores NaN aparecen, entre otros, en las columnas de
    las filas de referencia al alinear tablas con pandas.
    """
    import math

    if isinstance(obj, dict):
        return {k: _limpiar_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_limpiar_nan(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, np.floating) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        # allow_nan=False garantiza que no se escape ningun NaN: si quedara alguno
        # sin limpiar, falla ruidosamente en vez de producir un JSON invalido.
        json.dump(_limpiar_nan(obj), fh, indent=2, ensure_ascii=False,
                  default=_default, allow_nan=False)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"No serializable: {type(obj)}")


def count_parameters(model: "torch.nn.Module") -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Promedio movil ponderado por tamano de batch."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0
