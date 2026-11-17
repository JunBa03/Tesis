"""Genera y audita el split por paciente. Ejecutar una vez antes de entrenar.

Escribe results/splits/ con los CSV de cada particion y un reporte de prevalencia
por observacion, que es el insumo para justificar la ponderacion de clases.

Uso:
    python scripts/prepare_splits.py --config configs/e1_baseline.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import apply_overrides, load_config
from src.data.io import read_chexpert_csv, resolve_dataset_root
from src.data.labels import build_label_matrix, prevalence_report
from src.data.splits import filter_frontal, patient_level_split, split_summary
from src.utils import save_json, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara el split por paciente de CheXpert")
    parser.add_argument("--config", default="configs/e1_baseline.yaml")
    parser.add_argument("--set", nargs="*", default=[], help="overrides: data.root=/ruta")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    out_dir = Path(cfg["output"]["results_dir"]) / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "prepare_splits.log")

    root = resolve_dataset_root(cfg["data"]["root"])
    logger.info("Raiz del dataset: %s", root)
    df_train_full = read_chexpert_csv(root, cfg["data"]["train_csv"])
    df_test = read_chexpert_csv(root, cfg["data"]["valid_csv"])
    logger.info("CSV leidos | train=%d filas | valid oficial=%d filas",
                len(df_train_full), len(df_test))

    if cfg["data"]["frontal_only"]:
        df_train_full = filter_frontal(df_train_full)
        df_test = filter_frontal(df_test)
        logger.info("Solo frontales | train=%d | valid=%d", len(df_train_full), len(df_test))

    df_tr, df_va = patient_level_split(
        df_train_full,
        val_fraction=cfg["data"]["internal_val_fraction"],
        # La particion usa su propia semilla (por defecto la del experimento) para que las
        # corridas con otra semilla de entrenamiento compartan exactamente el mismo split.
        seed=cfg["data"].get("split_seed", cfg["experiment"]["seed"]),
    )
    summary = split_summary(df_tr, df_va)
    logger.info("Split por paciente: %s", summary)
    assert summary["patient_overlap"] == 0, "Hay filtracion entre particiones"

    df_tr.to_csv(out_dir / "train_split.csv", index=False)
    df_va.to_csv(out_dir / "val_internal_split.csv", index=False)
    df_test.to_csv(out_dir / "test_official.csv", index=False)

    # Prevalencia bajo cada politica de incertidumbre.
    for policy in ("u_zeros", "u_ones", "u_ignore"):
        labels, mask, df_p = build_label_matrix(
            df_tr, policy=policy, u_ignore_mode=cfg["labels"].get("u_ignore_mode", "row")
        )
        rep = prevalence_report(labels, mask)
        rep.to_csv(out_dir / f"prevalence_{policy}.csv", index=False)
        logger.info("%s | filas efectivas=%d", policy, len(df_p))

    save_json(summary, out_dir / "split_summary.json")
    logger.info("Listo. Archivos en %s", out_dir)


if __name__ == "__main__":
    main()
