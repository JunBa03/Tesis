"""Mide cuánto tarda el sistema en analizar una radiografía, etapa por etapa.

Reproduce el camino que sigue la pantalla Análisis: ensamblado de todas las
configuraciones (con y sin TTA), mapa de calor Grad-CAM++ del modelo E7 y segmentación del
tórax. Respalda con una medición el requerimiento no funcional de tiempo de respuesta.

Por defecto corre en CPU, el caso de un equipo sin placa de video. Cada etapa se
repite sobre ``--n`` radiografías del conjunto oficial (después de dos pasadas de
calentamiento) y se informa la mediana y el percentil 95.

    python scripts/medir_latencia.py                 (CPU, 20 imágenes)
    python scripts/medir_latencia.py --device cuda
    python scripts/medir_latencia.py --carpeta mis_rx (radiografías propias)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from PIL import Image

from app.core import ablation, anatomia, registry
from app.core.gradcam import GradCAM
from src.config import load_config
from src.data.io import read_chexpert_csv, resolve_dataset_root
from src.inference import Clasificador

RAIZ = Path(__file__).resolve().parents[1]


def imagenes_de_prueba(carpeta: str | None, n: int) -> list[Path]:
    if carpeta:
        exts = {".jpg", ".jpeg", ".png"}
        rutas = sorted(p for p in Path(carpeta).iterdir() if p.suffix.lower() in exts)
    else:
        cfg = load_config(RAIZ / "configs" / "e1_baseline.yaml")
        root = resolve_dataset_root(cfg["data"]["root"])
        df = read_chexpert_csv(root, cfg["data"]["valid_csv"])
        df = df[df["Frontal/Lateral"].str.strip().str.lower() == "frontal"]
        rutas = [Path(root) / p for p in df["Path"]]
    if not rutas:
        sys.exit("No se encontraron imágenes para medir.")
    return rutas[:n]


def cronometrar(funcion, imagenes, sincronizar) -> dict:
    """Mediana y p95 (en segundos) de ``funcion(imagen)`` sobre las imágenes."""
    for img in imagenes[:2]:      # calentamiento: no se cuenta
        funcion(img)
    tiempos = []
    for img in imagenes:
        t0 = time.perf_counter()
        funcion(img)
        sincronizar()
        tiempos.append(time.perf_counter() - t0)
    return {"mediana_s": round(statistics.median(tiempos), 3),
            "p95_s": round(float(np.percentile(tiempos, 95)), 3),
            "n": len(tiempos)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Latencia del análisis de una radiografía")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--carpeta", default=None)
    args = ap.parse_args()

    device = args.device
    sincronizar = (torch.cuda.synchronize if device.startswith("cuda")
                   else (lambda: None))
    rutas = imagenes_de_prueba(args.carpeta, args.n)
    imagenes = [Image.open(r).convert("RGB") for r in rutas]

    modelos = registry.descubrir(RAIZ)
    if not modelos:
        sys.exit("No hay modelos en models/ ni en checkpoints/.")
    clasif = {eid: Clasificador(ruta, device=device) for eid, ruta in modelos.items()}
    id_mapa = "E7" if "E7" in clasif else list(clasif)[-1]
    gcam = GradCAM(Clasificador(modelos[id_mapa], device=device))

    etapas = {}
    primero = next(iter(clasif))
    etapas[f"1 modelo ({primero}, sin TTA)"] = cronometrar(
        lambda im: clasif[primero].predecir(im), imagenes, sincronizar)
    etapas[f"1 modelo ({id_mapa}, sin TTA)"] = cronometrar(
        lambda im: clasif[id_mapa].predecir(im), imagenes, sincronizar)
    etapas[f"Ensamblado de {len(clasif)} (sin TTA)"] = cronometrar(
        lambda im: ablation.predecir_ensemble(clasif, im, tta=False), imagenes, sincronizar)
    etapas[f"Ensamblado de {len(clasif)} + TTA"] = cronometrar(
        lambda im: ablation.predecir_ensemble(clasif, im, tta=True), imagenes, sincronizar)
    etapas[f"Grad-CAM++ ({id_mapa}, 1 patología)"] = cronometrar(
        lambda im: gcam.generar(im, "Pleural Effusion"), imagenes, sincronizar)
    if anatomia.disponible():
        etapas["Segmentación del tórax"] = cronometrar(
            lambda im: anatomia.mascara_torax(im), imagenes, sincronizar)

    def completo(im):
        ablation.predecir_ensemble(clasif, im, tta=True)
        gcam.generar(im, "Pleural Effusion")
        if anatomia.disponible():
            anatomia.mascara_torax(im)
    etapas["Análisis completo (como la demo)"] = cronometrar(completo, imagenes, sincronizar)

    equipo = {
        "dispositivo": device,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "cpu": platform.processor() or platform.machine(),
        "hilos_torch": torch.get_num_threads(),
        "nucleos": os.cpu_count(),
        "sistema": platform.platform(),
        "modelos": list(clasif),
    }
    resultado = {"equipo": equipo, "etapas": etapas}

    print("\n" + "=" * 66)
    print(f"  Latencia por radiografía · {device} · {len(imagenes)} imágenes")
    print("=" * 66)
    for nombre, r in etapas.items():
        print(f"  {nombre:<40} mediana {r['mediana_s']:6.3f} s   p95 {r['p95_s']:6.3f} s")
    print("=" * 66)

    salida = RAIZ / "results" / f"latencia_{device.split(':')[0]}.json"
    salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Guardado en {salida}")


if __name__ == "__main__":
    main()
