"""Prepara una versión portátil de la aplicación para un equipo sin el dataset.

Sin el dataset (11 GB) ni los checkpoints de entrenamiento (85 MB cada uno), la
aplicación necesita:

  models/           modelos livianos, solo pesos en fp16
  demo_samples/     un puñado de radiografías del conjunto oficial, con sus
                    etiquetas de consenso, para no depender de tener archivos a mano

    python scripts/preparar_demo.py
    python scripts/preparar_demo.py --n 10 --solo E1

AVISO SOBRE LICENCIA: las imágenes de CheXpert están sujetas al acuerdo de uso de
Stanford, que no permite redistribuirlas. Si vas a subir demo_samples/ a GitHub,
**el repositorio tiene que ser privado**. En un repositorio público hay que dejar
la carpeta afuera y llevar las imágenes por separado.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import load_config
from src.data.io import read_chexpert_csv, resolve_dataset_root
from src.data.labels import COMPETITION_TASKS, OBSERVATIONS


def _tomar(grupos, n: int) -> pd.DataFrame:
    """Arma la muestra tomando de cada grupo su cuota, rellenando si falta."""
    partes, restante = [], n
    for grupo, cuota in grupos:
        if len(grupo) and cuota and restante > 0:
            tomar = min(cuota, len(grupo), restante)
            partes.append(grupo.sample(tomar, random_state=7))
            restante -= tomar
    return partes, restante


def elegir_muestras(df: pd.DataFrame, n: int, dificiles: bool = False) -> pd.DataFrame:
    """Selecciona estudios para la demostración.

    Por defecto arma un conjunto variado (casos con varios hallazgos, con uno solo
    y sin nada), porque una demo donde todo da positivo no muestra nada.

    Con ``dificiles=True`` sesga la selección hacia los casos donde el modelo tiene
    más chances de equivocarse: positivos en las patologías de menor AUC
    (atelectasia y consolidación) y estudios con muchos hallazgos simultáneos. Permite
    mostrar también los casos en los que el sistema falla.
    """
    positivos = df[COMPETITION_TASKS].sum(axis=1)
    if dificiles:
        dificil = df[(df["Atelectasis"] == 1.0) | (df["Consolidation"] == 1.0)]
        muchos = df[positivos >= 3]
        sin_hallazgos = df[positivos == 0]
        grupos = ((dificil, n // 2), (muchos, n // 4),
                  (sin_hallazgos, n - n // 2 - n // 4))
    else:
        grupos = ((df[positivos >= 2], n // 3), (df[positivos == 1], n // 3),
                  (df[positivos == 0], n // 3))

    partes, restante = _tomar(grupos, n)
    if restante > 0:
        ya = pd.concat(partes).index if partes else []
        resto = df.drop(index=ya, errors="ignore")
        if len(resto):
            partes.append(resto.sample(min(restante, len(resto)), random_state=7))
    return pd.concat(partes).drop_duplicates(subset="Path").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepara la demo portátil")
    parser.add_argument("--config", default="configs/e1_baseline.yaml")
    parser.add_argument("--n", type=int, default=12,
                        help="imágenes de ejemplo (la cola de priorización luce mejor "
                             "con 12-15)")
    parser.add_argument("--dificiles", action="store_true",
                        help="sesgar la selección hacia casos ambiguos, donde el modelo "
                             "tiene más chances de equivocarse")
    parser.add_argument("--destino", default="demo_samples")
    parser.add_argument("--solo", default=None, help="exportar solo un modelo, ej. E1")
    parser.add_argument("--sin-modelos", action="store_true",
                        help="solo preparar las imágenes")
    args = parser.parse_args()

    raiz = Path(__file__).resolve().parents[1]

    # --- 1. modelos livianos
    if not args.sin_modelos:
        print("=" * 68)
        print("  MODELOS")
        print("=" * 68)
        import subprocess

        cmd = [sys.executable, str(raiz / "scripts" / "export_demo_model.py")]
        if args.solo:
            cmd += ["--solo", args.solo]
        r = subprocess.run(cmd, cwd=raiz)
        if r.returncode != 0:
            print("\n  No se pudieron exportar los modelos. ¿Terminó al menos E1?")
        print()

    # --- 2. imágenes de ejemplo
    print("=" * 68)
    print("  IMÁGENES DE EJEMPLO")
    print("=" * 68)

    cfg = load_config(raiz / args.config)
    try:
        root = resolve_dataset_root(cfg["data"]["root"])
        df = read_chexpert_csv(root, cfg["data"]["valid_csv"])
    except FileNotFoundError:
        print("  No se encontró el conjunto de validación. Se omite este paso.")
        return

    df = df[df["Frontal/Lateral"].str.strip().str.lower() == "frontal"].copy()
    for col in OBSERVATIONS:
        df[col] = df[col].fillna(0.0)

    muestras = elegir_muestras(df, args.n, dificiles=args.dificiles)
    if args.dificiles:
        print("  (modo --dificiles: sesgado a casos ambiguos)")

    destino = raiz / args.destino
    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)

    manifiesto = []
    for i, fila in muestras.iterrows():
        origen = Path(root) / fila["Path"]
        if not origen.exists():
            continue
        nombre = f"estudio_{i+1:02d}{origen.suffix}"
        shutil.copy2(origen, destino / nombre)
        verdad = {o: float(fila[o]) for o in OBSERVATIONS}
        positivos = [o for o in COMPETITION_TASKS if verdad[o] == 1.0]
        manifiesto.append({
            "archivo": nombre,
            "origen": fila["Path"],
            "etiquetas": verdad,
            "hallazgos_benchmark": positivos,
        })
        etiqueta = ", ".join(positivos) if positivos else "sin hallazgos del benchmark"
        print(f"  {nombre:<20} {etiqueta}")

    (destino / "manifiesto.json").write_text(
        json.dumps(manifiesto, indent=2, ensure_ascii=False), encoding="utf-8")

    peso = sum(f.stat().st_size for f in destino.iterdir()) / 1e6
    print(f"\n  {len(manifiesto)} imágenes en {destino}/  ({peso:.1f} MB)")
    print()
    print("=" * 68)
    print("  Las imágenes de CheXpert están sujetas al acuerdo de uso de Stanford,")
    print("  que no permite redistribuirlas. Si subís demo_samples/ a GitHub, el")
    print("  repositorio tiene que ser PRIVADO.")
    print("=" * 68)


if __name__ == "__main__":
    main()
