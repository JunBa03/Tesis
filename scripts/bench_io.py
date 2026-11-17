"""Mide dónde se pierde el tiempo al cargar imágenes.

La calibración dio 29 imágenes por segundo cuando la GPU puede con unas 400: el
entrenamiento está esperando datos. Este script separa las tres etapas de la carga
para saber cuál es la lenta, en vez de adivinar.

  1. lectura de bytes    sin decodificar: mide el disco
  2. decodificación JPEG  PIL abre y convierte a RGB: mide el CPU
  3. transformaciones     recorte, rotación, jitter, normalización

Uso:
    python scripts/bench_io.py
    python scripts/bench_io.py --n 300
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from src.config import load_config
from src.data.io import read_chexpert_csv, resolve_dataset_root
from src.data.transforms import build_train_transform


def cronometrar(fn, rutas) -> tuple[float, float]:
    """Devuelve (segundos totales, imágenes por segundo)."""
    t0 = time.perf_counter()
    for p in rutas:
        fn(p)
    dt = time.perf_counter() - t0
    return dt, len(rutas) / dt if dt > 0 else float("inf")


def diagnostico_ruta(root: Path) -> None:
    """Pistas sobre dónde vive el dataset."""
    ruta = str(root.resolve())
    print(f"  ruta         {ruta}")

    sospechas = []
    if "OneDrive" in ruta:
        sospechas.append(
            "La ruta contiene 'OneDrive'. Si los archivos están en modo\n"
            "               'Archivos a petición', cada lectura puede bajar el archivo\n"
            "               de la nube. Es la causa más cara posible."
        )
    if "\\Desktop\\" in ruta or "/Desktop/" in ruta:
        sospechas.append(
            "Está en el Escritorio. En Windows 11 el Escritorio suele estar\n"
            "               sincronizado con OneDrive aunque la ruta no lo diga.\n"
            "               Verificá si la carpeta tiene un ícono de nube."
        )

    try:
        import ctypes

        letra = os.path.splitdrive(ruta)[0]
        if letra:
            tipo = ctypes.windll.kernel32.GetDriveTypeW(f"{letra}\\")
            nombres = {2: "extraíble", 3: "disco fijo", 4: "de red", 5: "CD-ROM", 6: "RAM"}
            print(f"  unidad       {letra}  ({nombres.get(tipo, 'desconocida')})")
            if tipo == 4:
                sospechas.append("Es una unidad de RED. Va a ser lentísimo.")
    except Exception:
        pass

    for s in sospechas:
        print(f"  ATENCIÓN:    {s}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico de velocidad de carga")
    parser.add_argument("--config", default="configs/e1_baseline.yaml")
    parser.add_argument("--n", type=int, default=200, help="imágenes a muestrear")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = resolve_dataset_root(cfg["data"]["root"])

    print()
    print("=" * 68)
    print("  DIAGNÓSTICO DE CARGA DE IMÁGENES")
    print("=" * 68)
    diagnostico_ruta(Path(root))

    df = read_chexpert_csv(root, cfg["data"]["train_csv"])
    if cfg["data"]["frontal_only"]:
        df = df[df["Frontal/Lateral"].str.strip().str.lower() == "frontal"]

    rng = random.Random(args.seed)
    muestras = rng.sample(df["Path"].tolist(), min(args.n, len(df)))
    rutas = [Path(root) / p for p in muestras]

    faltan = [p for p in rutas[:20] if not p.exists()]
    if faltan:
        print(f"\n  ERROR: no se encontró {faltan[0]}")
        sys.exit(1)

    tam = sum(p.stat().st_size for p in rutas[:50]) / 50
    print(f"  archivos     {len(rutas)} muestreados al azar, {tam/1024:.0f} KB promedio")
    print()

    # --- 1. lectura de bytes, en frío (archivos nunca tocados en esta corrida)
    def leer(p): return p.read_bytes()
    dt_io, ips_io = cronometrar(leer, rutas)
    mb_s = (tam * len(rutas) / 1e6) / dt_io
    print(f"  1. lectura de bytes      {ips_io:8.1f} img/s   "
          f"({1000*dt_io/len(rutas):5.1f} ms/img, {mb_s:.1f} MB/s)")

    # --- 2. la misma lectura otra vez: ahora el SO ya cacheó
    dt_io2, ips_io2 = cronometrar(leer, rutas)
    print(f"     misma lectura cacheada {ips_io2:8.1f} img/s   "
          f"({1000*dt_io2/len(rutas):5.1f} ms/img)")

    # --- 3. decodificación
    def decodificar(p): return Image.open(p).convert("RGB")
    dt_dec, ips_dec = cronometrar(decodificar, rutas)
    print(f"  2. + decodificar JPEG    {ips_dec:8.1f} img/s   "
          f"({1000*dt_dec/len(rutas):5.1f} ms/img)")

    # --- 4. pipeline completo
    tf = build_train_transform(cfg["data"]["image_size"])
    def completo(p): return tf(Image.open(p).convert("RGB"))
    dt_tf, ips_tf = cronometrar(completo, rutas)
    print(f"  3. + transformaciones    {ips_tf:8.1f} img/s   "
          f"({1000*dt_tf/len(rutas):5.1f} ms/img)")

    nw = cfg["data"]["num_workers"]
    print()
    print(f"  Con {nw} workers en paralelo: ~{ips_tf*nw:.0f} img/s teóricos")
    print(f"  Una época de 181.291 imágenes: ~{181291/(ips_tf*nw)/60:.0f} min")
    print()

    # --- veredicto
    # La pregunta correcta no es cuál de las tres etapas es la más lenta, sino si
    # el conjunto alcanza para alimentar a la GPU. Una 3060 Ti entrenando
    # DenseNet-121 a 224x224 con AMP consume del orden de 300-400 img/s.
    NECESITA = 350
    capacidad = ips_tf * nw

    print("=" * 68)
    if ips_io < 150:
        print("  EL CUELLO DE BOTELLA ES EL DISCO.")
        print()
        print("  Leer los bytes ya es lento por sí solo, antes de decodificar nada.")
        print("  Por orden de impacto:")
        print("   1. Si la carpeta está en OneDrive: sacala de ahí, o marcala como")
        print("      'Conservar siempre en este dispositivo'.")
        print("   2. Excluir la carpeta del análisis en tiempo real de Windows Defender.")
        print("      Son 191.027 aperturas de archivo por época.")
        print("   3. Si el disco es mecánico, mover el dataset a un SSD.")
    elif capacidad < NECESITA:
        print(f"  LA CARGA NO ALCANZA: {capacidad:.0f} img/s contra ~{NECESITA} que pide la GPU.")
        print()
        if ips_tf < ips_dec / 2:
            print("  Las transformaciones son lo más caro. ColorJitter es la peor del")
            print("  pipeline: se puede bajar su intensidad sin alterar el diseño")
            print("  experimental, porque no cambia qué variable se está midiendo.")
        else:
            print("  La decodificación JPEG domina. Subir data.num_workers hasta 6.")
    else:
        print(f"  LA CARGA NO ES EL PROBLEMA: {capacidad:.0f} img/s disponibles contra")
        print(f"  ~{NECESITA} que pide la GPU. Sobra {capacidad/NECESITA:.1f}x de margen.")
        print()
        print("  Si el entrenamiento igual va lento, el cuello está en el paso de GPU.")
        print("  Medilo aislado con:   python scripts\\bench_gpu.py")
    print("=" * 68)
    print()


if __name__ == "__main__":
    main()
