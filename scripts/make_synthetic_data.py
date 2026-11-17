"""Genera un CheXpert sintetico en miniatura con la MISMA estructura que el real.

No sirve para producir resultados: sirve para verificar que el pipeline completo
corre end-to-end (splits, politicas de incertidumbre, perdidas, entrenamiento,
AUC-ROC) sin necesidad de descargar los 11 GB del dataset.

Uso:
    python scripts/make_synthetic_data.py --out data_synthetic --patients 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image

from src.data.labels import OBSERVATIONS

DATASET_DIR = "CheXpert-v1.0-small"


def _make_image(rng: np.random.Generator, size: int, positive: bool) -> Image.Image:
    """Imagen con una senal debil pero aprendible correlacionada con la etiqueta."""
    base = rng.normal(loc=110, scale=25, size=(size, size))
    if positive:
        yy, xx = np.mgrid[0:size, 0:size]
        cx, cy = size * 0.5, size * 0.6
        blob = 70 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (size * 0.18) ** 2))
        base = base + blob
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="L")


def _sample_labels(rng: np.random.Generator, uncertain_rate: float) -> tuple[list, bool]:
    """Devuelve una fila de 14 valores en {1.0, 0.0, -1.0, nan} y si es 'positiva'."""
    row: list = []
    positive = bool(rng.random() < 0.45)
    for obs in OBSERVATIONS:
        u = rng.random()
        if u < uncertain_rate:
            row.append(-1.0)
        elif u < uncertain_rate + 0.15:
            row.append(np.nan)  # no mencionado
        else:
            # Prevalencias heterogeneas, como en el dataset real.
            p = 0.35 if obs in ("Lung Opacity", "Support Devices", "Pleural Effusion") else 0.12
            if obs in ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"):
                p = 0.55 if positive else 0.08
            row.append(1.0 if rng.random() < p else 0.0)
    return row, positive


def build(out_root: Path, n_patients: int, size: int, seed: int,
          uncertain_rate: float) -> None:
    rng = np.random.default_rng(seed)
    ds_root = out_root / DATASET_DIR

    def write_split(split: str, patients: range, offset: int) -> pd.DataFrame:
        records = []
        for pid in patients:
            pname = f"patient{pid + offset:05d}"
            for study in range(1, rng.integers(1, 4) + 1):
                for view, kind in ((1, "frontal"), (2, "lateral")):
                    if kind == "lateral" and rng.random() < 0.6:
                        continue
                    rel = f"{DATASET_DIR}/{split}/{pname}/study{study}/view{view}_{kind}.jpg"
                    fpath = out_root / rel
                    fpath.parent.mkdir(parents=True, exist_ok=True)
                    labels, positive = _sample_labels(rng, uncertain_rate)
                    _make_image(rng, size, positive).save(fpath, quality=90)
                    records.append(
                        {
                            "Path": rel,
                            "Sex": rng.choice(["Male", "Female"]),
                            "Age": int(rng.integers(20, 90)),
                            "Frontal/Lateral": "Frontal" if kind == "frontal" else "Lateral",
                            "AP/PA": rng.choice(["AP", "PA"]) if kind == "frontal" else "",
                            **dict(zip(OBSERVATIONS, labels)),
                        }
                    )
        return pd.DataFrame(records)

    ds_root.mkdir(parents=True, exist_ok=True)

    df_train = write_split("train", range(n_patients), offset=1)
    df_train.to_csv(ds_root / "train.csv", index=False)

    # El conjunto oficial no tiene inciertos: sus etiquetas son consenso de radiologos.
    n_valid = max(4, n_patients // 8)
    df_valid = write_split("valid", range(n_valid), offset=90000)
    obs_cols = df_valid[OBSERVATIONS].fillna(0.0).replace(-1.0, 0.0)
    df_valid[OBSERVATIONS] = obs_cols
    df_valid.to_csv(ds_root / "valid.csv", index=False)

    print(f"Dataset sintetico en {out_root}")
    print(f"  train.csv: {len(df_train)} filas / {n_patients} pacientes")
    print(f"  valid.csv: {len(df_valid)} filas / {n_valid} pacientes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera un CheXpert sintetico de prueba")
    parser.add_argument("--out", default="data_synthetic")
    parser.add_argument("--patients", type=int, default=40)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--uncertain-rate", type=float, default=0.18)
    args = parser.parse_args()
    build(Path(args.out), args.patients, args.size, args.seed, args.uncertain_rate)


if __name__ == "__main__":
    main()
