"""Entrena una configuracion experimental y la evalua sobre el conjunto oficial.

Uso:
    python scripts/train.py --config configs/e1_baseline.yaml
    python scripts/train.py --config configs/e5_focal.yaml --set train.epochs=12 data.root=/datos
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Hay que fijar esto ANTES de importar numpy/torch: de lo contrario cada proceso
# (el principal y cada worker del DataLoader) arranca su propio pool de hilos de
# OpenBLAS dimensionado según los 12 hilos del CPU. Con varios procesos vivos a la
# vez, esos pools compiten por RAM y terminan en
# "OpenBLAS error: Memory allocation still failed". El trabajo pesado va en la GPU;
# el CPU solo decodifica imágenes, que es tarea de un hilo por worker.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.config import apply_overrides, load_config
from src.data.dataset import build_dataloaders, build_datasets
from src.engine import evaluate, fit
from src.losses import build_loss
from src.metrics import best_threshold_per_observation, threshold_metrics
from src.models import build_model
from src.utils import get_device, save_json, set_seed, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena una configuracion E1-E6")
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--device", default=None, help="cuda | cpu (autodeteccion por defecto)")
    parser.add_argument("--skip-test", action="store_true",
                        help="no evaluar sobre el conjunto oficial al finalizar")
    parser.add_argument("--resume", action="store_true",
                        help="retomar desde el ultimo estado guardado si existe "
                             "(<checkpoint>_last.pt). Util en corridas de varios dias.")
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.set)
    name = cfg["experiment"]["name"]

    results_dir = Path(cfg["output"]["results_dir"]) / name
    results_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(results_dir / "train.log")

    logger.info("=" * 72)
    logger.info("Experimento: %s", name)
    logger.info("Notas: %s", cfg["experiment"].get("notes", ""))
    logger.info("Incertidumbre=%s | perdida=%s | backbone=%s",
                cfg["labels"]["uncertainty_policy"], cfg["loss"]["name"],
                cfg["model"]["backbone"])
    logger.info("=" * 72)

    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    set_seed(cfg["experiment"]["seed"],
             deterministic=cfg["train"].get("deterministic", True))
    device = get_device(args.device)
    logger.info("Dispositivo: %s", device)

    datasets = build_datasets(cfg)
    loaders = build_dataloaders(cfg, datasets)

    model = build_model(cfg).to(device)
    if cfg["train"].get("channels_last", False) and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
        logger.info("Formato de memoria: channels_last")
    criterion = build_loss(
        cfg, datasets["train"].labels, datasets["train"].mask
    ).to(device)

    ckpt_path = Path(cfg["output"]["checkpoint_dir"]) / f"{name}.pt"
    state = fit(cfg, model, loaders, criterion, device, ckpt_path,
                resume=args.resume, progress_path=results_dir / "progress.json")

    logger.info("Mejor epoca: %d | AUC(5) validacion interna: %.4f",
                state.best_epoch, state.best_metric)

    # Umbrales calibrados sobre la validacion INTERNA, nunca sobre el conjunto oficial.
    val_res, val_true, val_score, _ = evaluate(
        model, loaders["val_internal"], device, criterion, desc="val final"
    )
    thresholds = best_threshold_per_observation(val_true, val_score)

    output = {
        "experiment": name,
        "config": dict(cfg),
        "best_epoch": state.best_epoch,
        "best_val_auc_competition_mean": state.best_metric,
        "history": state.history,
        "val_internal": val_res,
        "thresholds_from_val": thresholds,
        "checkpoint": str(ckpt_path),
    }

    if not args.skip_test:
        test_res, test_true, test_score, test_mask = evaluate(
            model, loaders["test_official"], device, desc="test oficial"
        )
        logger.info("Conjunto oficial | AUC(5)=%.4f | AUC(14)=%s",
                    test_res["auc_competition_mean"] or float("nan"),
                    test_res["auc_all_mean"])
        for task, value in test_res["auc_competition"].items():
            logger.info("  %-18s AUC = %s", task,
                        f"{value:.4f}" if value is not None else "n/d")
        output["test_official"] = test_res
        output["test_official_threshold_metrics"] = threshold_metrics(
            test_true, test_score, threshold=0.5, mask=test_mask
        )

    save_json(output, results_dir / "results.json")
    logger.info("Resultados guardados en %s", results_dir / "results.json")


if __name__ == "__main__":
    main()
