"""Particionado del conjunto de entrenamiento a nivel de PACIENTE.

Dividir por imagen produce filtracion de informacion: un mismo paciente aporta
varios estudios y varias vistas, de modo que imagenes casi identicas terminarian
repartidas entre entrenamiento y validacion, inflando la metrica. El split se hace
por identificador de paciente extraido de la ruta del archivo.
"""
from __future__ import annotations

import re
from typing import Tuple

import numpy as np
import pandas as pd

PATIENT_RE = re.compile(r"(patient\d+)")


def extract_patient_id(path: str) -> str:
    """Obtiene 'patientXXXXX' de una ruta tipo
    'CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg'."""
    match = PATIENT_RE.search(str(path))
    if match is None:
        raise ValueError(f"No se pudo extraer el patient id de la ruta: {path!r}")
    return match.group(1)


def add_patient_column(df: pd.DataFrame, path_col: str = "Path") -> pd.DataFrame:
    df = df.copy()
    df["patient_id"] = df[path_col].map(extract_patient_id)
    return df


def patient_level_split(
    df: pd.DataFrame,
    val_fraction: float,
    seed: int,
    path_col: str = "Path",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el DataFrame en train/val garantizando que ningun paciente aparezca en ambos.

    Returns:
        (df_train, df_val) con la columna 'patient_id' agregada.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction debe estar en (0, 1), recibido {val_fraction}")

    df = add_patient_column(df, path_col=path_col)
    patients = np.sort(df["patient_id"].unique())

    rng = np.random.default_rng(seed)
    shuffled = patients.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * val_fraction)))
    val_patients = set(shuffled[:n_val].tolist())

    is_val = df["patient_id"].isin(val_patients)
    df_train = df.loc[~is_val].reset_index(drop=True)
    df_val = df.loc[is_val].reset_index(drop=True)

    assert not (set(df_train["patient_id"]) & set(df_val["patient_id"])), \
        "Filtracion detectada: hay pacientes compartidos entre train y val."

    return df_train, df_val


def filter_frontal(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva solo proyecciones frontales (PA/AP).

    El benchmark se apoya en la placa frontal; las laterales estan presentes en
    CheXpert pero son secundarias y mezclan dos distribuciones visuales distintas.
    """
    if "Frontal/Lateral" in df.columns:
        out = df[df["Frontal/Lateral"].str.strip().str.lower() == "frontal"]
    else:
        out = df[df["Path"].str.contains("frontal", case=False, na=False)]
    return out.reset_index(drop=True)


def split_summary(df_train: pd.DataFrame, df_val: pd.DataFrame) -> dict:
    return {
        "train_images": int(len(df_train)),
        "val_images": int(len(df_val)),
        "train_patients": int(df_train["patient_id"].nunique()),
        "val_patients": int(df_val["patient_id"].nunique()),
        "patient_overlap": int(
            len(set(df_train["patient_id"]) & set(df_val["patient_id"]))
        ),
    }
