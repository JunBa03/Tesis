"""Prueba de robustez frente a radiografías con formatos de otros equipos.

El sistema está entrenado con radiografías de CheXpert (JPEG de 8 bits, escala de grises,
contraste homogéneo). Las imágenes que llegan de otros equipos difieren en profundidad de
bits, orientación, resolución y contraste. Esta prueba toma radiografías del conjunto
oficial, genera variantes que simulan esas diferencias y compara la predicción del
ensamblado sobre cada variante con la de la imagen original.

Variantes:
    png16       PNG de 16 bits (como un DICOM exportado)
    exif_rot    imagen girada 90° con la etiqueta EXIF que indica corregirla
    baja_res    reducida a 160 px de lado (escaneo o captura de baja calidad)
    bajo_contr  contraste comprimido al 40 % del rango (placa sub o sobreexpuesta)
    negativo    escala invertida (algunos visores exportan así) — caso límite esperado

Criterios de aceptación (por variante):
    - el sistema procesa la imagen sin error;
    - el cambio medio de probabilidad en las 5 patologías del benchmark es < 0,05;
    - el hallazgo principal (mayor probabilidad) coincide con el de la original.

    python scripts/prueba_robustez.py            (10 imágenes)
    python scripts/prueba_robustez.py --n 20
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageOps

from app.core import ablation, registry
from src.config import load_config
from src.data.io import read_chexpert_csv, resolve_dataset_root
from src.data.labels import COMPETITION_TASKS
from src.inference import Clasificador

RAIZ = Path(__file__).resolve().parents[1]
UMBRAL_CAMBIO = 0.05


def _por_archivo(img: Image.Image, **kw) -> Image.Image:
    """Guarda y relee la imagen, como pasaría al subir un archivo."""
    buf = io.BytesIO()
    img.save(buf, **kw)
    buf.seek(0)
    return Image.open(buf)


def variantes(original: Image.Image) -> dict:
    gris = original.convert("L")
    arr = np.asarray(gris).astype(np.float32)

    png16 = _por_archivo(Image.fromarray((arr * 257).astype(np.uint16)), format="PNG")

    girada = gris.rotate(90, expand=True)          # contenido girado 90° antihorario
    exif = Image.Exif()
    exif[0x0112] = 6                                 # orientación: rotar 90° horario al mostrar
    exif_rot = _por_archivo(girada, format="JPEG", quality=95, exif=exif)

    lado = 160
    baja = gris.resize((lado, int(lado * gris.height / gris.width)), Image.BILINEAR)

    m = arr.mean()
    bajo = Image.fromarray(np.clip(m + (arr - m) * 0.4, 0, 255).astype(np.uint8))

    return {"png16": png16, "exif_rot": exif_rot, "baja_res": baja,
            "bajo_contr": bajo, "negativo": ImageOps.invert(gris)}


def benchmark(pred) -> np.ndarray:
    return np.array([pred.probabilidades[t] for t in COMPETITION_TASKS])


def main() -> None:
    ap = argparse.ArgumentParser(description="Robustez frente a formatos externos")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(RAIZ / "configs" / "e1_baseline.yaml")
    root = resolve_dataset_root(cfg["data"]["root"])
    df = read_chexpert_csv(root, cfg["data"]["valid_csv"])
    df = df[df["Frontal/Lateral"].str.strip().str.lower() == "frontal"]
    rutas = [Path(root) / p for p in df["Path"].iloc[: args.n]]

    clasif = {eid: Clasificador(r, device=args.device)
              for eid, r in registry.descubrir(RAIZ).items()}

    por_variante: dict[str, dict] = {}
    for ruta in rutas:
        original = Image.open(ruta)
        base = benchmark(ablation.predecir_ensemble(clasif, original))
        top_base = int(np.argmax(base))
        for nombre, img in variantes(original).items():
            r = por_variante.setdefault(nombre, {"errores": 0, "cambios": [], "top_igual": 0})
            try:
                p = benchmark(ablation.predecir_ensemble(clasif, img))
            except Exception as e:  # una excepción es un fallo de la prueba
                r["errores"] += 1
                r.setdefault("detalle_errores", []).append(f"{ruta.name}: {e}")
                continue
            r["cambios"].append(float(np.abs(p - base).mean()))
            r["top_igual"] += int(int(np.argmax(p)) == top_base)

    resultado = {}
    print("\n" + "=" * 78)
    print(f"  Robustez del ensamblado · {len(rutas)} radiografías del conjunto oficial")
    print("=" * 78)
    print(f"  {'Variante':<12}{'errores':>8}{'cambio medio':>15}{'cambio máx':>13}"
          f"{'mismo hallazgo':>17}   resultado")
    for nombre, r in por_variante.items():
        c = np.array(r["cambios"]) if r["cambios"] else np.array([np.nan])
        ok = r["errores"] == 0 and float(np.nanmean(c)) < UMBRAL_CAMBIO \
            and r["top_igual"] == len(rutas)
        resultado[nombre] = {
            "imagenes": len(rutas), "errores": r["errores"],
            "cambio_medio": round(float(np.nanmean(c)), 4),
            "cambio_max": round(float(np.nanmax(c)), 4),
            "mismo_hallazgo": r["top_igual"], "aprobada": bool(ok),
            **({"detalle_errores": r["detalle_errores"]} if "detalle_errores" in r else {}),
        }
        print(f"  {nombre:<12}{r['errores']:>8}{np.nanmean(c):>15.3f}{np.nanmax(c):>13.3f}"
              f"{r['top_igual']:>12}/{len(rutas):<4}   {'APROBADA' if ok else 'NO APROBADA'}")
    print("=" * 78)

    salida = RAIZ / "results" / "pruebas"
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "robustez.json").write_text(
        json.dumps({"umbral_cambio": UMBRAL_CAMBIO, "variantes": resultado},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Guardado en {salida / 'robustez.json'}")


if __name__ == "__main__":
    main()
