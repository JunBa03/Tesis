"""Mide el paso de GPU aislado, sin ninguna carga de datos.

bench_io.py mostró que la carga de imágenes puede entregar ~1.100 img/s con 6
workers, muy por encima de lo que la GPU necesita. Si el entrenamiento real anda a
29 img/s, el problema está en el paso de GPU, no en el DataLoader.

Este script entrena con tensores aleatorios ya en la placa (cero disco, cero CPU) y
compara las combinaciones de formato de memoria y precisión mixta. En particular
mide si `channels_last` ayuda o perjudica: en DenseNet-121, que concatena en cada
bloque, el formato NHWC puede forzar conversiones constantes y degradar el
rendimiento en vez de mejorarlo.

Uso:
    python scripts/bench_gpu.py
    python scripts/bench_gpu.py --batch 64 --iters 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from src.models import MultiLabelClassifier


def medir(backbone: str, batch: int, size: int, iters: int, warmup: int,
          channels_last: bool, amp: bool) -> float:
    """Devuelve imágenes por segundo de un paso completo de entrenamiento."""
    torch.cuda.empty_cache()
    modelo = MultiLabelClassifier(backbone=backbone, pretrained=False).cuda().train()
    if channels_last:
        modelo = modelo.to(memory_format=torch.channels_last)

    opt = torch.optim.Adam(modelo.parameters(), lr=1e-4)
    crit = nn.BCEWithLogitsLoss()
    try:
        scaler = torch.amp.GradScaler("cuda") if amp else None
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler() if amp else None

    fmt = torch.channels_last if channels_last else torch.contiguous_format
    x = torch.randn(batch, 3, size, size, device="cuda").to(memory_format=fmt)
    y = torch.randint(0, 2, (batch, 14), device="cuda").float()

    def paso():
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=amp):
            loss = crit(modelo(x), y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

    for _ in range(warmup):
        paso()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        paso()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    del modelo, opt, x, y
    torch.cuda.empty_cache()
    return batch * iters / dt


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark del paso de GPU")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No hay GPU disponible.")
        sys.exit(1)

    torch.backends.cudnn.benchmark = True

    print()
    print("=" * 68)
    print("  PASO DE GPU AISLADO  (tensores sintéticos, sin DataLoader)")
    print("=" * 68)
    print(f"  {torch.cuda.get_device_name(0)}")
    print(f"  batch={args.batch}  resolución={args.size}  iteraciones={args.iters}")
    print()
    print(f"  {'backbone':<14}{'formato':<16}{'AMP':<7}{'img/s':>9}   min/época")
    print("  " + "-" * 62)

    resultados = {}
    for backbone in ("densenet121", "resnet50"):
        for cl in (False, True):
            for amp in (True,):
                try:
                    ips = medir(backbone, args.batch, args.size,
                                args.iters, args.warmup, cl, amp)
                except torch.cuda.OutOfMemoryError:
                    print(f"  {backbone:<14}{'channels_last' if cl else 'contiguous':<16}"
                          f"{'sí' if amp else 'no':<7}{'OOM':>9}")
                    continue
                resultados[(backbone, cl, amp)] = ips
                minutos = 181291 / ips / 60
                print(f"  {backbone:<14}{'channels_last' if cl else 'contiguous':<16}"
                      f"{'sí' if amp else 'no':<7}{ips:9.0f}   {minutos:6.1f}")

    # sin AMP, solo para el baseline de DenseNet
    try:
        ips = medir("densenet121", args.batch, args.size, args.iters, args.warmup,
                    False, False)
        resultados[("densenet121", False, False)] = ips
        print(f"  {'densenet121':<14}{'contiguous':<16}{'no':<7}{ips:9.0f}"
              f"   {181291/ips/60:6.1f}")
    except torch.cuda.OutOfMemoryError:
        pass

    print()
    print("=" * 68)

    d_cont = resultados.get(("densenet121", False, True))
    d_chan = resultados.get(("densenet121", True, True))
    if d_cont and d_chan:
        if d_chan < d_cont * 0.9:
            print(f"  channels_last PERJUDICA a DenseNet-121: {d_chan:.0f} contra "
                  f"{d_cont:.0f} img/s")
            print("  Poner  channels_last: false  en configs/base.yaml")
        elif d_chan > d_cont * 1.1:
            print(f"  channels_last ayuda: {d_chan:.0f} contra {d_cont:.0f} img/s. Dejarlo.")
        else:
            print(f"  channels_last es indiferente ({d_chan:.0f} vs {d_cont:.0f} img/s).")

    mejor = max(resultados.values()) if resultados else 0
    if mejor:
        print()
        print(f"  Mejor caso medido: {mejor:.0f} img/s "
              f"→ {181291/mejor/60:.0f} min por época de puro cómputo.")
        print(f"  El entrenamiento real dio 29 img/s.")
        if mejor > 200:
            print()
            print("  La GPU rinde bien en aislamiento, así que la pérdida está en la")
            print("  interacción con el DataLoader. Sospechas, en orden:")
            print("   1. channels_last (ver arriba)")
            print("   2. num_workers demasiado bajo para 5,5 ms de transformación por")
            print("      imagen: 6 workers dan ~1.100 img/s teóricos pero en Windows el")
            print("      arranque por spawn y la serialización se comen buena parte.")
            print("   3. pin_memory con poca RAM libre.")
        else:
            print()
            print("  La GPU tampoco rinde en aislamiento. Revisar temperatura y límite")
            print("  de potencia con nvidia-smi mientras corre.")
    print("=" * 68)
    print()


if __name__ == "__main__":
    main()
