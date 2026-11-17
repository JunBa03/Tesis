"""Evalua un checkpoint ya entrenado sobre el conjunto de validacion oficial de CheXpert.

Uso:
    python scripts/evaluate.py --config configs/e1_baseline.yaml \
        --checkpoint checkpoints/E1_baseline_densenet121_uzeros_bce.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import apply_overrides, load_config
from src.data.dataset import build_dataloaders, build_datasets
from src.data.labels import OBSERVATIONS
from src.engine import evaluate, load_checkpoint
from src.metrics import threshold_metrics
from src.models import build_model
from src.utils import get_device, save_json, set_seed, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua un checkpoint sobre el conjunto oficial")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--split", default="test_official",
                        choices=["test_official", "val_internal"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-predictions", action="store_true")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    name = cfg["experiment"]["name"]
    results_dir = Path(cfg["output"]["results_dir"]) / name
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(results_dir / "evaluate.log")

    set_seed(cfg["experiment"]["seed"])
    device = get_device(args.device)

    datasets = build_datasets(cfg)
    loaders = build_dataloaders(cfg, datasets)

    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(Path(args.checkpoint), model, device=device)
    logger.info("Checkpoint cargado (epoca %s, AUC val %.4f)",
                ckpt.get("epoch"), ckpt.get("best_metric", float("nan")))

    res, y_true, y_score, mask = evaluate(
        model, loaders[args.split], device, desc=args.split
    )
    res["threshold_metrics"] = threshold_metrics(y_true, y_score, 0.5, mask)

    logger.info("AUC(5 patologias) = %.4f", res["auc_competition_mean"] or float("nan"))
    for obs, value in res["auc_per_observation"].items():
        logger.info("  %-28s %s", obs, f"{value:.4f}" if value is not None else "n/d")

    save_json(res, results_dir / f"eval_{args.split}.json")

    if args.save_predictions:
        preds = pd.DataFrame(y_score, columns=[f"pred_{o}" for o in OBSERVATIONS])
        truth = pd.DataFrame(y_true, columns=[f"true_{o}" for o in OBSERVATIONS])
        out = pd.concat([datasets[args.split].df[["Path"]], preds, truth], axis=1)
        out.to_csv(results_dir / f"predictions_{args.split}.csv", index=False)
        logger.info("Predicciones guardadas.")


if __name__ == "__main__":
    main()
