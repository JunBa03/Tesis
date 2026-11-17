"""Exporta las probabilidades por imagen de cada configuración, para el análisis estadístico.

Los ``results.json`` de cada experimento guardan solo el AUC final. Para poder calcular
intervalos de confianza (bootstrap) y comparaciones pareadas entre configuraciones hace
falta la probabilidad que asignó cada modelo a cada imagen. Este script la guarda una sola
vez; el análisis (``scripts/analisis_estadistico.py``) no necesita GPU ni los modelos.

Guarda en ``results/predicciones/``:
    test.npz   conjunto oficial (radiografías frontales con consenso de radiólogos)
    val.npz    partición de validación interna (opcional, ``--sin-val`` la omite)

Cada archivo contiene ``y_true``, ``mask`` y ``paths`` (orden de las imágenes) y, por
configuración ``Ek``, ``score_Ek`` (pasada simple) y ``tta_Ek`` (con TTA, solo en test).

    python scripts/exportar_predicciones.py
    python scripts/exportar_predicciones.py --sin-val --device cpu
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from src.config import load_config
from src.data.dataset import build_dataloaders, build_datasets
from src.engine import load_checkpoint, predict
from src.models import build_model
from src.tta import probabilidades_tta
from src.utils import get_device, set_seed, setup_logging

RAIZ = Path(__file__).resolve().parents[1]


@torch.no_grad()
def predecir_tta(model, loader, device) -> np.ndarray:
    model.eval()
    trozos = [probabilidades_tta(model, lote[0].to(device)).float().cpu().numpy()
              for lote in loader]
    return np.concatenate(trozos, axis=0)


def configuraciones() -> dict:
    """``{"E1": (cfg, ruta_checkpoint), ...}`` para las configuraciones entrenadas."""
    modelos = {}
    for yaml in sorted(glob.glob(str(RAIZ / "configs" / "e*.yaml"))):
        cfg = load_config(yaml)
        nombre = cfg["experiment"]["name"]
        ckpt = RAIZ / "checkpoints" / f"{nombre}.pt"
        if ckpt.exists():
            modelos[nombre.split("_", 1)[0].upper()] = (cfg, ckpt)
    return modelos


def main() -> None:
    ap = argparse.ArgumentParser(description="Exporta predicciones por imagen")
    ap.add_argument("--device", default=None)
    ap.add_argument("--sin-val", action="store_true", help="no exportar la validación interna")
    args = ap.parse_args()

    salida = RAIZ / "results" / "predicciones"
    salida.mkdir(parents=True, exist_ok=True)
    log = setup_logging(salida / "exportar.log")
    device = get_device(args.device)
    set_seed(0)

    modelos = configuraciones()
    if not modelos:
        sys.exit("No hay checkpoints en checkpoints/. Entrená primero.")
    log.info("Configuraciones: %s | dispositivo: %s", ", ".join(modelos), device)

    test, val = {}, {}
    for eid, (cfg, ckpt) in modelos.items():
        datasets = build_datasets(cfg)
        loaders = build_dataloaders(cfg, datasets)
        model = build_model(cfg).to(device)
        load_checkpoint(ckpt, model, device=device)

        yt, st, mt, _ = predict(model, loaders["test_official"], device, desc=f"{eid} test")
        test[f"score_{eid}"] = st
        test[f"tta_{eid}"] = predecir_tta(model, loaders["test_official"], device)
        # Las etiquetas del conjunto oficial son de consenso y no dependen de la política
        # de incertidumbre: se guardan una sola vez (se verifica que coincidan).
        if "y_true" not in test:
            test["y_true"], test["mask"] = yt, mt
            test["paths"] = np.array(datasets["test_official"].paths)
        elif not np.array_equal(test["y_true"], yt):
            log.warning("%s: etiquetas del test distintas a las de la primera configuración", eid)

        if not args.sin_val:
            yv, sv, mv, _ = predict(model, loaders["val_internal"], device, desc=f"{eid} val")
            val[f"score_{eid}"] = sv
            # En validación interna las etiquetas SÍ dependen de la política (U-Zeros, U-Ones,
            # U-Ignore): se guardan las de cada configuración.
            val[f"y_{eid}"], val[f"mask_{eid}"] = yv, mv
            if "paths" not in val:
                val["paths"] = np.array(datasets["val_internal"].paths)
        log.info("%s listo | test=%d imágenes", eid, len(yt))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    np.savez_compressed(salida / "test.npz", **test)
    log.info("Guardado %s", salida / "test.npz")
    if val:
        np.savez_compressed(salida / "val.npz", **val)
        log.info("Guardado %s", salida / "val.npz")


if __name__ == "__main__":
    main()
