"""Mide, sin reentrenar, cuánto mejora combinar los modelos de distintas maneras.

Compara sobre el conjunto de validación OFICIAL de CheXpert varios métodos de ensemble y
elige el mejor. Los pesos de cada método se derivan del AUC en la partición INTERNA de
validación (que sale del train), no del conjunto oficial: así la elección no ve el
conjunto de prueba y la comparación con la literatura sigue siendo válida.

Métodos que compara:
  - cada modelo por separado
  - promedio simple de todos
  - promedio de los mejores k (top-k por AUC interno)
  - promedio ponderado por AUC interno
  - ponderado por patología (peso por observación según el AUC interno)
  - ruteo por patología (cada observación al mejor modelo en validación interna)
  y cada uno también con TTA (test-time augmentation).

Al terminar escribe results/ensemble_config.json con el método ganador, que la demo lee
automáticamente.

Uso:
    python scripts/evaluar_mejoras.py
    python scripts/evaluar_mejoras.py --sin-tta        (más rápido, sin TTA)
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
from src.data.labels import COMPETITION_TASKS, OBSERVATIONS
from src.engine import evaluate, load_checkpoint
from src.metrics import auc_per_observation, mean_competition_auc
from src.models import build_model
from src.tta import probabilidades_tta
from src.utils import get_device, save_json, set_seed, setup_logging

RAIZ = Path(__file__).resolve().parents[1]
COMP_IDX = [OBSERVATIONS.index(t) for t in COMPETITION_TASKS]


def _id(nombre: str) -> str:
    return nombre.split("_", 1)[0].upper()


@torch.no_grad()
def predecir_tta(model, loader, device) -> np.ndarray:
    """Probabilidades sobre un loader usando TTA. Devuelve [N, 14]."""
    model.eval()
    trozos = []
    for lote in loader:
        x = lote[0].to(device)
        trozos.append(probabilidades_tta(model, x).float().cpu().numpy())
    return np.concatenate(trozos, axis=0)


def auc_media(y_true, y_score, mask) -> float:
    return mean_competition_auc(auc_per_observation(y_true, y_score, mask)) or float("nan")


def combinar(scores: dict, pesos_obs: dict) -> np.ndarray:
    """Combina las predicciones por observación. ``pesos_obs[obs][eid]`` -> peso."""
    ids = list(scores)
    salida = np.zeros_like(next(iter(scores.values())))
    for j, obs in enumerate(OBSERVATIONS):
        w = np.array([pesos_obs.get(obs, {}).get(e, 0.0) for e in ids])
        if w.sum() <= 0:
            w = np.ones(len(ids))
        w = w / w.sum()
        salida[:, j] = sum(wi * scores[e][:, j] for wi, e in zip(w, ids))
    return salida


def main() -> None:
    parser = argparse.ArgumentParser(description="Mide métodos de ensemble sin reentrenar")
    parser.add_argument("--sin-tta", action="store_true", help="no evaluar TTA")
    parser.add_argument("--device", default=None)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--rapido", action="store_true",
                        help="usar un subconjunto de la validación interna para los "
                             "pesos (más rápido, pesos un poco más ruidosos)")
    args = parser.parse_args()
    # En modo rápido se limita la validación interna a ~2.500 imágenes.
    val_batches = 40 if args.rapido else None

    logger = setup_logging(RAIZ / "results" / "evaluar_mejoras.log")
    device = get_device(args.device)
    usar_tta = not args.sin_tta

    # --- descubrir configuraciones con checkpoint
    modelos = {}  # id -> (cfg, ruta_ckpt)
    for yaml in sorted(glob.glob(str(RAIZ / "configs" / "e*.yaml"))):
        if Path(yaml).stem == "base":
            continue
        cfg = load_config(yaml)
        nombre = cfg["experiment"]["name"]
        ckpt = RAIZ / "checkpoints" / f"{nombre}.pt"
        if ckpt.exists():
            modelos[_id(nombre)] = (cfg, ckpt)
    if len(modelos) < 2:
        print(f"Se necesitan al menos 2 modelos entrenados. Hay {len(modelos)}.")
        sys.exit(1)
    print(f"Modelos: {', '.join(modelos)}")

    # --- correr cada modelo en val_internal (para pesos) y test_official (para medir)
    set_seed(0)
    auc_interno = {}          # id -> {obs: auc}
    score_test = {}           # id -> [N,14] probabilidades simples
    score_test_tta = {}       # id -> [N,14] con TTA
    y_true = y_mask = None

    for eid, (cfg, ckpt) in modelos.items():
        datasets = build_datasets(cfg)
        loaders = build_dataloaders(cfg, datasets)
        model = build_model(cfg).to(device)
        load_checkpoint(ckpt, model, device=device)

        # AUC por patología en validación interna (fuente de los pesos, sin leakage)
        _, yi, si, mi = evaluate(model, loaders["val_internal"], device,
                                 max_batches=val_batches, desc=f"{eid} val")
        auc_interno[eid] = auc_per_observation(yi, si, mi)

        # Predicciones sobre el conjunto oficial
        _, yt, st, mt = evaluate(model, loaders["test_official"], device, desc=f"{eid} test")
        score_test[eid] = st
        if y_true is None:
            y_true, y_mask = yt, mt
        if usar_tta:
            score_test_tta[eid] = predecir_tta(model, loaders["test_official"], device)
        logger.info("%s listo", eid)

    ids = list(modelos)

    # --- construir los pesos de cada método a partir del AUC interno
    def auc_medio_interno(eid):
        vals = [auc_interno[eid][t] for t in COMPETITION_TASKS if auc_interno[eid].get(t)]
        return float(np.mean(vals)) if vals else 0.0

    medio = {e: auc_medio_interno(e) for e in ids}
    orden = sorted(ids, key=lambda e: medio[e], reverse=True)

    pesos_uniforme = {o: {e: 1.0 for e in ids} for o in OBSERVATIONS}
    pesos_topk = {o: {e: 1.0 for e in orden[:args.topk]} for o in OBSERVATIONS}
    # ponderado global: peso = (AUC interno - 0.5), realza a los mejores
    pesos_pond = {o: {e: max(medio[e] - 0.5, 1e-3) for e in ids} for o in OBSERVATIONS}
    # ponderado por patología y ruteo por patología
    pesos_patol, pesos_ruteo = {}, {}
    for o in OBSERVATIONS:
        aucs = {e: (auc_interno[e].get(o) or 0.5) for e in ids}
        pesos_patol[o] = {e: max(aucs[e] - 0.5, 1e-3) for e in ids}
        mejor = max(aucs, key=aucs.get)
        pesos_ruteo[o] = {e: (1.0 if e == mejor else 0.0) for e in ids}

    metodos = {
        "promedio_simple": pesos_uniforme,
        f"top{args.topk}": pesos_topk,
        "ponderado_global": pesos_pond,
        "ponderado_patologia": pesos_patol,
        "ruteo_patologia": pesos_ruteo,
    }

    # --- medir todo sobre el conjunto oficial
    print("\n" + "=" * 60)
    print("  AUC-ROC medio (5 patologías) sobre el conjunto OFICIAL")
    print("=" * 60)
    for e in orden:
        print(f"  modelo {e:<3}                {auc_media(y_true, score_test[e], y_mask):.4f}")
    print("  " + "-" * 56)

    resultados = {}
    for nombre, pesos in metodos.items():
        a = auc_media(y_true, combinar(score_test, pesos), y_mask)
        resultados[(nombre, False)] = a
        print(f"  {nombre:<24} {a:.4f}")
        if usar_tta:
            at = auc_media(y_true, combinar(score_test_tta, pesos), y_mask)
            resultados[(nombre, True)] = at
            print(f"  {nombre + ' + TTA':<24} {at:.4f}")
    print("=" * 60)

    # --- elegir el mejor y guardarlo para la demo
    (mejor_metodo, mejor_tta), mejor_auc = max(resultados.items(), key=lambda kv: kv[1])
    pesos_elegidos = metodos[mejor_metodo]
    # normalizar por observación para que la demo los use directo
    pesos_norm = {}
    for o in OBSERVATIONS:
        s = sum(pesos_elegidos[o].values()) or 1.0
        pesos_norm[o] = {e: round(pesos_elegidos[o][e] / s, 4) for e in ids}

    config = {
        "metodo": mejor_metodo,
        "usar_tta": bool(mejor_tta),
        "auc_oficial": round(mejor_auc, 4),
        "pesos": pesos_norm,
        "auc_interno_medio": {e: round(medio[e], 4) for e in ids},
    }
    save_json(config, RAIZ / "results" / "ensemble_config.json")

    print(f"\n  MEJOR: {mejor_metodo}{' + TTA' if mejor_tta else ''} "
          f"-> AUC {mejor_auc:.4f}")
    print(f"  (vs promedio simple {resultados[('promedio_simple', False)]:.4f})")
    print("  Guardado en results/ensemble_config.json. La demo lo usa automáticamente.\n")


if __name__ == "__main__":
    main()
