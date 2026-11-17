"""Dataset PyTorch para CheXpert y construccion de los DataLoader."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .io import read_chexpert_csv
from .labels import OBSERVATIONS, build_label_matrix
from .splits import filter_frontal, patient_level_split

logger = logging.getLogger("chexpert")


class CheXpertDataset(Dataset):
    """Devuelve (imagen, etiquetas, mascara) para clasificacion multilabel.

    La mascara vale 0 en las posiciones que no deben contribuir a la perdida
    (politica U-Ignore en modo 'mask'); en las demas politicas es todo unos.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        root: str | Path,
        transform: Optional[Callable] = None,
        uncertainty_policy: str = "u_zeros",
        u_ignore_mode: str = "row",
        blank_as_negative: bool = True,
        path_col: str = "Path",
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.path_col = path_col

        labels, mask, df_out = build_label_matrix(
            df,
            policy=uncertainty_policy,
            u_ignore_mode=u_ignore_mode,
            blank_as_negative=blank_as_negative,
        )
        self.df = df_out.reset_index(drop=True)
        self.labels = labels
        self.mask = mask
        self.paths = self.df[path_col].tolist()

    def __len__(self) -> int:
        return len(self.df)

    def _resolve(self, rel_path: str) -> Path:
        candidate = self.root / rel_path
        if candidate.exists():
            return candidate
        # Tolera que la raiz ya incluya el prefijo 'CheXpert-v1.0-small/'.
        alt = self.root / Path(rel_path).relative_to(Path(rel_path).parts[0])
        if alt.exists():
            return alt
        raise FileNotFoundError(f"No se encontro la imagen: {candidate}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self._resolve(self.paths[idx])
        # Las radiografias son de un canal; se replican a 3 para los backbones ImageNet.
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        labels = torch.from_numpy(self.labels[idx])
        mask = torch.from_numpy(self.mask[idx])
        return image, labels, mask

    @property
    def label_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.labels, columns=OBSERVATIONS)


def build_datasets(cfg) -> dict:
    """Construye los tres conjuntos del protocolo experimental.

    - train:          particion de entrenamiento (split por paciente).
    - val_internal:   particion interna para early stopping y seleccion de modelo.
    - test_official:  valid.csv de CheXpert (234 imagenes, consenso de 3 radiologos).
                      Nunca se usa para tomar decisiones de entrenamiento.
    """
    from .io import resolve_dataset_root
    from .transforms import build_eval_transform, build_train_transform

    root = resolve_dataset_root(cfg["data"]["root"])
    image_size = cfg["data"]["image_size"]

    df_train_full = read_chexpert_csv(root, cfg["data"]["train_csv"])
    df_test = read_chexpert_csv(root, cfg["data"]["valid_csv"])

    if cfg["data"]["frontal_only"]:
        df_train_full = filter_frontal(df_train_full)
        df_test = filter_frontal(df_test)

    df_tr, df_va = patient_level_split(
        df_train_full,
        val_fraction=cfg["data"]["internal_val_fraction"],
        # La particion usa su propia semilla (por defecto la del experimento) para que las
        # corridas con otra semilla de entrenamiento compartan exactamente el mismo split.
        seed=cfg["data"].get("split_seed", cfg["experiment"]["seed"]),
    )

    policy = cfg["labels"]["uncertainty_policy"]
    mode = cfg["labels"].get("u_ignore_mode", "row")
    blank_neg = cfg["labels"].get("blank_as_negative", True)

    train_ds = CheXpertDataset(
        df_tr, root, build_train_transform(image_size), policy, mode, blank_neg
    )
    # La validacion interna se evalua siempre con U-Zeros para que las seis
    # configuraciones se comparen contra el mismo conjunto de referencia.
    val_ds = CheXpertDataset(
        df_va, root, build_eval_transform(image_size), "u_zeros", "row", blank_neg
    )
    # El conjunto oficial ya viene etiquetado por radiologos: no tiene inciertos.
    test_ds = CheXpertDataset(
        df_test, root, build_eval_transform(image_size), "u_zeros", "row", blank_neg
    )

    logger.info(
        "Datasets | train=%d img / %d pac | val_int=%d img / %d pac | test_oficial=%d img",
        len(train_ds), df_tr["patient_id"].nunique(),
        len(val_ds), df_va["patient_id"].nunique(),
        len(test_ds),
    )
    return {"train": train_ds, "val_internal": val_ds, "test_official": test_ds}


def build_dataloaders(cfg, datasets: dict) -> dict:
    """Construye los DataLoader.

    Los loaders de evaluación usan MENOS workers que el de entrenamiento y no los
    dejan vivos entre épocas. La razón es de memoria: en Windows cada worker es un
    proceso nuevo que reimporta torch y ocupa varios cientos de MB. Con
    persistent_workers en los tres loaders, durante la validación conviven los
    workers de entrenamiento y los de validación, y la suma agota la RAM del
    sistema — el síntoma es un error de OpenBLAS al asignar memoria, que despista
    porque parece un problema de álgebra lineal y no de procesos.
    """
    bs = cfg["train"]["batch_size"]
    nw = int(cfg["data"]["num_workers"])
    # La validación corre una vez por época sobre un conjunto chico: no necesita
    # la misma paralelización que el entrenamiento.
    nw_eval = int(cfg["data"].get("eval_num_workers", min(2, nw)))
    pin = cfg["data"]["pin_memory"]
    pf = int(cfg["data"].get("prefetch_factor", 2))

    def extra(n: int) -> dict:
        # prefetch_factor solo es valido con workers > 0
        return {"prefetch_factor": pf} if n > 0 else {}

    logger.info(
        "DataLoaders | batch=%d | workers train=%d eval=%d | prefetch=%d",
        bs, nw, nw_eval, pf,
    )

    return {
        "train": DataLoader(
            datasets["train"], batch_size=bs, shuffle=True, num_workers=nw,
            pin_memory=pin, drop_last=True, persistent_workers=nw > 0, **extra(nw),
        ),
        "val_internal": DataLoader(
            datasets["val_internal"], batch_size=bs, shuffle=False,
            num_workers=nw_eval, pin_memory=pin, persistent_workers=False,
            **extra(nw_eval),
        ),
        "test_official": DataLoader(
            datasets["test_official"], batch_size=bs, shuffle=False,
            num_workers=nw_eval, pin_memory=pin, persistent_workers=False,
            **extra(nw_eval),
        ),
    }
