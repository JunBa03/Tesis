"""Lectura de los CSV de CheXpert. Sin dependencias de torch."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATASET_DIRNAME = "CheXpert-v1.0-small"


def resolve_dataset_root(root: str | Path) -> Path:
    """Normaliza la raiz del dataset.

    La columna `Path` de los CSV de CheXpert guarda rutas con el prefijo
    'CheXpert-v1.0-small/...', asi que la raiz correcta es la carpeta que
    *contiene* a CheXpert-v1.0-small, no la carpeta misma. Es un error facil de
    cometer, y el sintoma es un FileNotFoundError recien al cargar la primera
    imagen, despues de haber construido todo el pipeline. Esta funcion acepta
    cualquiera de las dos formas y devuelve la correcta.

        .../data/raw                        -> .../data/raw
        .../data/raw/CheXpert-v1.0-small    -> .../data/raw
    """
    root = Path(root).expanduser()
    if (root / DATASET_DIRNAME).is_dir():
        return root
    if root.name.lower().startswith("chexpert-v1.0"):
        return root.parent
    return root


def read_chexpert_csv(root: str | Path, rel_csv: str) -> pd.DataFrame:
    """Lee un CSV de CheXpert tolerando que la ruta se pase relativa o absoluta.

    Prueba, en orden: root/rel_csv, rel_csv tal cual, y root/<nombre del archivo>.
    """
    root = resolve_dataset_root(root)
    candidates = [root / rel_csv, Path(rel_csv), root / Path(rel_csv).name]
    for cand in candidates:
        if cand.exists():
            return pd.read_csv(cand)
    raise FileNotFoundError(
        f"No se encontro el CSV. Rutas probadas: {[str(c) for c in candidates]}"
    )
