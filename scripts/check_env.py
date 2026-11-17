"""Comprueba que el entorno pueda entrenar de verdad. Sale con codigo 1 si no.

El error que este script existe para atrapar: `pip install torch` desde PyPI en
Windows instala la build **solo-CPU**. Todo funciona, nada falla, y el
entrenamiento corre unas cien veces mas lento sin avisar. La unica senal es una
linea que dice `Dispositivo: cpu` en medio del log.

Uso:
    python scripts/check_env.py
"""
from __future__ import annotations

import sys

CUDA_INDEX = "https://download.pytorch.org/whl/cu124"
COMANDO = (
    "pip uninstall -y torch torchvision\n"
    f"    pip install torch torchvision --index-url {CUDA_INDEX}"
)


def fallar(titulo: str, detalle: str) -> None:
    print()
    print("=" * 68)
    print(f"  {titulo}")
    print("=" * 68)
    print(detalle)
    print()
    sys.exit(1)


def main() -> None:
    try:
        import torch
    except ImportError:
        fallar("PyTorch no está instalado",
               f"  Instalalo con:\n\n    {COMANDO}\n")

    print(f"  torch        {torch.__version__}")
    try:
        import torchvision
        print(f"  torchvision  {torchvision.__version__}")
    except ImportError:
        fallar("torchvision no está instalado", f"    {COMANDO}")

    compilado_con_cuda = torch.version.cuda
    print(f"  compilado con CUDA: {compilado_con_cuda or 'NO (build solo-CPU)'}")

    if compilado_con_cuda is None:
        fallar(
            "PyTorch está instalado SIN soporte CUDA",
            "  Esta es la build solo-CPU. El entrenamiento correría en el procesador:\n"
            "  una época de 181.291 imágenes pasaría de minutos a días.\n\n"
            "  Causa: `pip install torch` desde PyPI baja la build CPU en Windows.\n"
            "  Hay que instalarlo desde el índice de CUDA:\n\n"
            f"    {COMANDO}\n\n"
            "  Si tu driver NVIDIA es viejo, verificá la versión que te corresponde en\n"
            "  https://pytorch.org/get-started/locally/ y cambiá cu124 por la que indique.",
        )

    if not torch.cuda.is_available():
        fallar(
            "PyTorch tiene CUDA pero no encuentra la placa",
            "  La build es correcta, pero `torch.cuda.is_available()` da False.\n"
            "  Suele ser el driver NVIDIA desactualizado.\n\n"
            "  Verificá que `nvidia-smi` funcione en una consola.\n"
            "  Si no existe el comando, hay que instalar o actualizar el driver.",
        )

    # RAM del sistema: los workers del DataLoader son procesos independientes en
    # Windows y su consumo se suma. Es la memoria que se agota primero.
    try:
        import shutil  # noqa: F401
        import ctypes

        class _MS(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _MS()
        st.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
        ram_total = st.ullTotalPhys / 1e9
        ram_libre = st.ullAvailPhys / 1e9
        print(f"  RAM          {ram_total:.1f} GB totales, {ram_libre:.1f} GB libres")
        if ram_total < 15:
            print("               AVISO: con menos de 16 GB, bajar data.num_workers a 4")
    except Exception:
        pass

    props = torch.cuda.get_device_properties(0)
    vram = props.total_memory / 1e9
    print(f"  GPU          {props.name}")
    print(f"  VRAM         {vram:.1f} GB")
    print(f"  capacidad    sm_{props.major}{props.minor}")

    # Prueba real: una convolución sobre la GPU con precisión mixta.
    try:
        x = torch.randn(8, 3, 224, 224, device="cuda")
        conv = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
        with torch.autocast("cuda"):
            y = conv(x)
        torch.cuda.synchronize()
        assert y.shape == (8, 16, 224, 224)
        print("  prueba AMP   OK")
    except Exception as exc:  # pragma: no cover
        fallar("La GPU no pudo ejecutar una convolución de prueba", f"  {exc}")

    if vram < 7.0:
        print()
        print(f"  AVISO: {vram:.1f} GB de VRAM. Con batch_size=64 puede faltar memoria.")
        print("         Si aparece CUDA out of memory, bajar a 32 en configs/base.yaml.")

    print()
    print("  Entorno listo para entrenar en GPU.")


if __name__ == "__main__":
    main()
