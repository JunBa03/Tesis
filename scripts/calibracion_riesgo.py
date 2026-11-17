"""Calibra el ensemble y evalúa la derivación selectiva con la curva riesgo-cobertura.

Corre las configuraciones sobre la validación interna (para ajustar la temperatura, sin
mirar el conjunto de prueba) y sobre el conjunto oficial (para medir). Produce:

  - la temperatura de calibración y el ECE antes/después, guardados en
    results/calibracion.json;
  - un diagrama de fiabilidad (results/figuras/calibracion_reliability.png);
  - la curva riesgo-cobertura, comparando una confianza por margen contra una que además
    usa el desacuerdo entre modelos (results/figuras/riesgo_cobertura.png).

Uso:
    python scripts/calibracion_riesgo.py
    python scripts/calibracion_riesgo.py --rapido --sin-tta
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import calibracion as cal
from src.config import load_config
from src.data.dataset import build_dataloaders, build_datasets
from src.data.labels import COMPETITION_TASKS, OBSERVATIONS
from src.engine import evaluate, load_checkpoint
from src.metrics import auc_per_observation, mean_competition_auc
from src.models import build_model
from src.tta import probabilidades_tta
from src.utils import get_device, save_json, set_seed, setup_logging

RAIZ = Path(__file__).resolve().parents[1]
FIG = RAIZ / "results" / "figuras"
COMP_IDX = [OBSERVATIONS.index(t) for t in COMPETITION_TASKS]
AZUL, AMBAR, GRIS = "#2563eb", "#d97706", "#9aa4b2"


def _id(nombre):
    return nombre.split("_", 1)[0].upper()


def predecir_tta(model, loader, device):
    import torch
    model.eval()
    trozos = []
    with torch.no_grad():
        for lote in loader:
            trozos.append(probabilidades_tta(model, lote[0].to(device)).cpu().numpy())
    return np.concatenate(trozos, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sin-tta", action="store_true")
    ap.add_argument("--rapido", action="store_true",
                    help="ajustar la temperatura sobre un subconjunto de validación")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logger = setup_logging(RAIZ / "results" / "calibracion.log")
    device = get_device(args.device)
    usar_tta = not args.sin_tta
    val_batches = 40 if args.rapido else None

    modelos = {}
    for y in sorted(glob.glob(str(RAIZ / "configs" / "e*.yaml"))):
        if Path(y).stem == "base":
            continue
        cfg = load_config(y)
        ck = RAIZ / "checkpoints" / f"{cfg['experiment']['name']}.pt"
        if ck.exists():
            modelos[_id(cfg["experiment"]["name"])] = (cfg, ck)
    if not modelos:
        print("No hay modelos entrenados en checkpoints/.")
        sys.exit(1)
    print(f"Modelos: {', '.join(modelos)}")

    set_seed(0)
    val_scores, test_scores = [], []
    yv = mv = yt = mt = None
    for eid, (cfg, ck) in modelos.items():
        ds = build_datasets(cfg)
        ld = build_dataloaders(cfg, ds)
        model = build_model(cfg).to(device)
        load_checkpoint(ck, model, device=device)
        _, yv_, sv, mv_ = evaluate(model, ld["val_internal"], device,
                                   max_batches=val_batches, desc=f"{eid} val")
        _, yt_, st, mt_ = evaluate(model, ld["test_official"], device, desc=f"{eid} test")
        if usar_tta:
            st = predecir_tta(model, ld["test_official"], device)
        val_scores.append(sv)
        test_scores.append(st)
        if yv is None:
            yv, mv, yt, mt = yv_, mv_, yt_, mt_
        logger.info("%s listo", eid)

    # Ensemble por promedio.
    ens_val = np.mean(val_scores, axis=0)
    ens_test = np.mean(test_scores, axis=0)
    disp_test = np.std(test_scores, axis=0)   # desacuerdo entre modelos, por entrada

    auc = mean_competition_auc(auc_per_observation(yt, ens_test, mt))
    print(f"\nAUC-ROC medio del ensemble (oficial): {auc:.4f}")

    # --- calibración por temperatura, ajustada en validación interna
    T = cal.ajustar_temperatura(ens_val, yv, mv)
    ens_test_cal = cal.aplicar_temperatura(ens_test, T)
    ece_antes = cal.ece(ens_test, yt, mt)
    ece_desp = cal.ece(ens_test_cal, yt, mt)
    print(f"Temperatura T = {T:.2f}")
    print(f"ECE oficial:  sin calibrar {ece_antes:.4f}  ->  calibrado {ece_desp:.4f}")

    # --- diagrama de fiabilidad (antes/después) sobre las patologías del benchmark
    def puntos_fiabilidad(prob):
        p = prob[:, COMP_IDX].ravel()
        y = yt[:, COMP_IDX].ravel()
        b = np.linspace(0, 1, 11)
        xs, ys = [], []
        for i in range(10):
            m = (p >= b[i]) & (p < b[i + 1] if i < 9 else p <= b[i + 1])
            if m.any():
                xs.append(p[m].mean()); ys.append(y[m].mean())
        return xs, ys

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], "--", color=GRIS, lw=1, label="Perfecta")
    x0, y0 = puntos_fiabilidad(ens_test)
    x1, y1 = puntos_fiabilidad(ens_test_cal)
    ax.plot(x0, y0, "o-", color=GRIS, label=f"Sin calibrar (ECE {ece_antes:.3f})")
    ax.plot(x1, y1, "o-", color=AZUL, label=f"Calibrado (ECE {ece_desp:.3f})")
    ax.set_xlabel("Confianza predicha"); ax.set_ylabel("Frecuencia real")
    ax.set_title("Diagrama de fiabilidad"); ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG / "calibracion_reliability.png", dpi=200)
    plt.close(fig)

    # --- curva riesgo-cobertura sobre decisiones del benchmark
    p = ens_test_cal[:, COMP_IDX].ravel()
    y = yt[:, COMP_IDX].ravel()
    d = disp_test[:, COMP_IDX].ravel()
    # umbral 0,5 sobre la probabilidad calibrada (decisión binaria por patología)
    correcto = ((p >= 0.5).astype(float) == y).astype(float)
    margen = np.abs(p - 0.5)
    conf_ens = margen * (1 - np.clip(d / 0.3, 0, 1))   # penaliza el desacuerdo

    cob_m, rie_m, aurc_m = cal.curva_riesgo_cobertura(margen, correcto)
    cob_e, rie_e, aurc_e = cal.curva_riesgo_cobertura(conf_ens, correcto)

    r100 = cal.riesgo_a_cobertura(cob_e, rie_e, 1.0)
    r80 = cal.riesgo_a_cobertura(cob_e, rie_e, 0.8)
    r50 = cal.riesgo_a_cobertura(cob_e, rie_e, 0.5)

    # Guardar los números PRIMERO: son lo más importante y así un problema al graficar
    # no obliga a repetir toda la inferencia.
    save_json({
        "temperatura": round(T, 4),
        "ece_sin_calibrar": round(ece_antes, 4),
        "ece_calibrado": round(ece_desp, 4),
        "auc_oficial": round(auc, 4),
        "aurc_margen": round(aurc_m, 4),
        "aurc_ensemble": round(aurc_e, 4),
        "riesgo_cobertura_100": round(r100, 4),
        "riesgo_cobertura_80": round(r80, 4),
        "riesgo_cobertura_50": round(r50, 4),
    }, RAIZ / "results" / "calibracion.json")

    print("\nRiesgo (error) según cuánto responde el sistema:")
    print(f"  responde el 100%:  {r100:.3f}")
    print(f"  responde el 80%:   {r80:.3f}")
    print(f"  responde el 50%:   {r50:.3f}")

    fig, ax = plt.subplots(figsize=(6, 4.4))
    ax.plot(cob_m, rie_m, color=GRIS, lw=2, label=f"Confianza por margen (AURC {aurc_m:.3f})")
    ax.plot(cob_e, rie_e, color=AMBAR, lw=2,
            label=f"Margen + desacuerdo del ensemble (AURC {aurc_e:.3f})")
    ax.set_xlabel("Cobertura (fracción de decisiones respondidas)")
    ax.set_ylabel("Riesgo (tasa de error)")
    ax.set_title("Curva riesgo-cobertura")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(FIG / "riesgo_cobertura.png", dpi=200)
    plt.close(fig)

    print("\nGuardado: results/calibracion.json y las dos figuras en results/figuras/.")


if __name__ == "__main__":
    main()
