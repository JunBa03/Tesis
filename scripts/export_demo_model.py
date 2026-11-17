"""Exporta un checkpoint liviano, solo para inferencia.

Un checkpoint de entrenamiento pesa unos 85 MB porque guarda el estado del
optimizador Adam (dos tensores extra por cada parámetro) más el historial y la
configuración completa. Nada de eso hace falta para predecir, y GitHub rechaza
archivos de más de 100 MB.

Este script deja solo los pesos, en media precisión:

    DenseNet-121   85 MB  ->  ~14 MB
    ResNet-50     ~300 MB ->  ~47 MB

La media precisión no cambia las predicciones de forma perceptible: la red se
entrenó con AMP, así que ya operaba en fp16 durante la mayor parte del cómputo.

Además embebe los umbrales calibrados, para que la demostración funcione sin
necesitar la carpeta results/ ni el dataset.

    python scripts/export_demo_model.py                  # todos los disponibles
    python scripts/export_demo_model.py --solo E1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.inference import buscar_checkpoints, cargar_umbrales


def exportar(origen: Path, destino_dir: Path, media_precision: bool = True) -> Path:
    ckpt = torch.load(origen, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})

    pesos = ckpt["model_state"]
    if media_precision:
        pesos = {
            k: (v.half() if v.is_floating_point() else v)
            for k, v in pesos.items()
        }

    experimento = cfg.get("experiment", {}).get("name", origen.stem)

    payload = {
        "model_state": pesos,
        "media_precision": media_precision,
        # Solo lo que la inferencia necesita: nada de datos, rutas ni historial.
        "config": {
            "experiment": {"name": experimento},
            "model": {
                "backbone": cfg.get("model", {}).get("backbone", "densenet121"),
                "dropout": cfg.get("model", {}).get("dropout", 0.2),
            },
            "labels": {
                "uncertainty_policy": cfg.get("labels", {}).get("uncertainty_policy", "?")
            },
            "loss": {"name": cfg.get("loss", {}).get("name", "?")},
            "data": {"image_size": cfg.get("data", {}).get("image_size", 224)},
        },
        "epoch": ckpt.get("epoch"),
        "best_metric": ckpt.get("best_metric"),
        "umbrales": cargar_umbrales(experimento),
    }

    destino_dir.mkdir(parents=True, exist_ok=True)
    destino = destino_dir / f"{experimento}.pt"
    torch.save(payload, destino)
    return destino


def main() -> None:
    parser = argparse.ArgumentParser(description="Exporta modelos livianos para la demo")
    parser.add_argument("--checkpoints", default="checkpoints")
    parser.add_argument("--destino", default="models")
    parser.add_argument("--solo", default=None,
                        help="prefijo, por ejemplo E1, para exportar uno solo")
    parser.add_argument("--precision-completa", action="store_true",
                        help="no convertir a fp16 (archivos del doble de tamaño)")
    args = parser.parse_args()

    disponibles = buscar_checkpoints(args.checkpoints)
    if args.solo:
        disponibles = [p for p in disponibles if p.stem.upper().startswith(args.solo.upper())]

    if not disponibles:
        print(f"No hay checkpoints en {args.checkpoints}/")
        print("Hay que entrenar al menos una configuración: 2_entrenar_todo.bat")
        sys.exit(1)

    destino_dir = Path(args.destino)
    print(f"Exportando {len(disponibles)} modelo(s) a {destino_dir}/\n")
    total_antes = total_despues = 0

    for origen in disponibles:
        antes = origen.stat().st_size / 1e6
        destino = exportar(origen, destino_dir,
                           media_precision=not args.precision_completa)
        despues = destino.stat().st_size / 1e6
        total_antes += antes
        total_despues += despues
        print(f"  {destino.name:<45} {antes:6.1f} MB -> {despues:5.1f} MB")
        if despues > 95:
            print(f"     AVISO: {despues:.0f} MB está cerca del límite de 100 MB de GitHub.")

    print(f"\n  Total: {total_antes:.0f} MB -> {total_despues:.0f} MB "
          f"({100*(1-total_despues/total_antes):.0f} % menos)")
    print(f"\n  Estos archivos SÍ van al repositorio. Los de checkpoints/ no.")


if __name__ == "__main__":
    main()
