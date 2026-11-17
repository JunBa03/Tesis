"""Genera las figuras de resultados a partir de las métricas guardadas.

Cada figura se produce solo si estan los datos que necesita, asi que el script se
puede correr en cualquier momento del proyecto: hoy produce la de composicion del
dataset, y despues del 11/8 produce las de resultados sin cambiar nada.

Figuras:
  fig1_composicion.png   composicion del dataset: positivos vs inciertos por observacion
  fig2_auc_config.png    AUC-ROC por patologia y configuracion (E1-E6 + referencia)
  fig3_deltas.png        efecto aislado de cada decision respecto al baseline
  fig4_curvas.png        curvas de entrenamiento por configuracion

Uso:
    python scripts/make_figures.py
    python scripts/make_figures.py --only 1        # solo la de composicion
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.io import read_chexpert_csv, resolve_dataset_root
from src.data.labels import (COMPETITION_TASKS, OBSERVATIONS, REFERENCE_AUC,
                             SHORT_NAMES)
from src.config import load_config

# Paleta sobria, legible en proyector. El baseline siempre en gris para que las
# configuraciones que cambian algo resalten contra el.
C_BASE = "#6b7280"
C_UNC = "#d97706"
C_LOSS = "#2563eb"
C_ARCH = "#059669"
C_REF = "#9ca3af"
C_POS = "#1f2937"

# Un tono por eje experimental, dos intensidades dentro de cada eje: el color dice
# QUE se modifico y la intensidad CUAL de las dos variantes. Asi la figura se lee
# sin recurrir a la leyenda.
CONFIG_COLOR = {
    "E1": C_BASE,       # baseline, gris
    "E2": "#b45309",    # incertidumbre — U-Ones
    "E3": "#fbbf24",    # incertidumbre — U-Ignore
    "E4": "#1d4ed8",    # pérdida — BCE ponderada
    "E5": "#93c5fd",    # pérdida — Focal
    "E6": C_ARCH,       # arquitectura — ResNet-50
}

CONFIG_LABEL = {
    "E1": "E1 · baseline",
    "E2": "E2 · U-Ones",
    "E3": "E3 · U-Ignore",
    "E4": "E4 · BCE pond.",
    "E5": "E5 · Focal",
    "E6": "E6 · ResNet-50",
}

plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 160,
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
})


# --------------------------------------------------------------------------- fig 1

def fig_composicion(cfg, out_dir: Path) -> bool:
    """Positivos vs inciertos por observacion. Justifica por que el eje de
    incertidumbre es la variable mas interesante del diseno."""
    try:
        root = resolve_dataset_root(cfg["data"]["root"])
        df = read_chexpert_csv(root, cfg["data"]["train_csv"])
    except FileNotFoundError:
        print("  fig1: no se encontro train.csv, se omite")
        return False

    if cfg["data"]["frontal_only"]:
        df = df[df["Frontal/Lateral"].str.strip().str.lower() == "frontal"]

    rows = []
    for obs in OBSERVATIONS:
        col = df[obs]
        rows.append({
            "obs": obs,
            "pos": int((col == 1).sum()),
            "inc": int((col == -1).sum()),
            "bench": obs in COMPETITION_TASKS,
        })
    t = pd.DataFrame(rows).sort_values("inc", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(t))
    ax.barh(y - 0.2, t["pos"], height=0.38, color=C_POS, label="Positivos")
    ax.barh(y + 0.2, t["inc"], height=0.38, color=C_UNC, label="Inciertos (−1)")

    ax.set_yticks(y)
    ax.set_yticklabels(t["obs"], fontsize=9)
    for tick, bench in zip(ax.get_yticklabels(), t["bench"]):
        if bench:
            tick.set_fontweight("bold")

    ax.set_xlabel("Imágenes frontales de entrenamiento")
    ax.set_title("Composición de CheXpert: positivos frente a etiquetas inciertas\n"
                 "En negrita, las cinco observaciones del benchmark oficial",
                 fontsize=11, loc="left")
    ax.legend(loc="lower right", frameon=False)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", "."))
    )

    top = t.iloc[-1]
    ax.annotate(
        f"{top['obs']}: {top['inc']:,}".replace(",", ".") + " inciertos\n"
        f"contra {top['pos']:,}".replace(",", ".") + " positivos",
        xy=(top["inc"], len(t) - 1 + 0.2), xytext=(top["inc"] * 1.08, len(t) - 3.2),
        fontsize=9, color=C_UNC,
        arrowprops=dict(arrowstyle="->", color=C_UNC, lw=1),
    )

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_composicion.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig1_composicion.png")
    return True


# ----------------------------------------------------------------------- resultados

def _load_results(results_dir: Path) -> list[dict]:
    out = []
    for path in sorted(results_dir.glob("*/results.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("test_official"):
            data["code"] = data["experiment"].split("_")[0].upper()
            out.append(data)
    return out


def fig_auc_config(results: list[dict], out_dir: Path) -> bool:
    """AUC-ROC por patologia y configuracion. Es la diapositiva 7."""
    if not results:
        print("  fig2: todavia no hay resultados, se omite")
        return False

    codes = [r["code"] for r in results]
    x = np.arange(len(COMPETITION_TASKS))
    width = 0.8 / (len(results) + 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, r in enumerate(results):
        vals = [r["test_official"]["auc_competition"].get(t) or np.nan
                for t in COMPETITION_TASKS]
        ax.bar(x + i * width, vals, width,
               label=CONFIG_LABEL.get(r["code"], r["code"]),
               color=CONFIG_COLOR.get(r["code"], C_BASE),
               edgecolor="white", linewidth=0.4)

    ref = REFERENCE_AUC["DenseNet-121 + U-Zeros (Irvin et al., 2019)"]
    ref_vals = [ref[t] for t in COMPETITION_TASKS]
    ax.plot(x + (len(results) - 1) * width / 2, ref_vals, "D--", color=C_REF,
            markersize=5, lw=1.2, label="Irvin et al. (2019)")

    ax.set_xticks(x + (len(results) - 1) * width / 2)
    ax.set_xticklabels([SHORT_NAMES[t] for t in COMPETITION_TASKS])
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color="#9ca3af", lw=0.8, ls=":")
    ax.set_title("AUC-ROC por patología — conjunto de validación oficial (202 radiografías frontales)",
                 fontsize=11, loc="left")
    ax.legend(ncol=4, frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_auc_config.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig2_auc_config.png")
    return True


def fig_deltas(results: list[dict], out_dir: Path) -> bool:
    """Efecto aislado de cada decision respecto al baseline. Es la diapositiva 8,
    la que responde la pregunta del trabajo."""
    base = next((r for r in results if r["code"] == "E1"), None)
    others = [r for r in results if r["code"] != "E1"]
    if base is None or not others:
        print("  fig3: hace falta E1 y al menos otra configuracion, se omite")
        return False

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(COMPETITION_TASKS))
    width = 0.8 / len(others)

    for i, r in enumerate(others):
        deltas = []
        for t in COMPETITION_TASKS:
            a = r["test_official"]["auc_competition"].get(t)
            b = base["test_official"]["auc_competition"].get(t)
            deltas.append((a - b) if (a is not None and b is not None) else np.nan)
        ax.bar(x + i * width, deltas, width,
               label=CONFIG_LABEL.get(r["code"], r["code"]),
               color=CONFIG_COLOR.get(r["code"], C_BASE),
               edgecolor="white", linewidth=0.4)

    ax.axhline(0, color=C_POS, lw=1.2)
    ax.set_xticks(x + (len(others) - 1) * width / 2)
    ax.set_xticklabels([SHORT_NAMES[t] for t in COMPETITION_TASKS])
    ax.set_ylabel("Δ AUC-ROC respecto al baseline E1")
    ax.set_title("Efecto aislado de cada decisión metodológica\n"
                 "Naranja: estrategia de incertidumbre · Azul: función de pérdida · "
                 "Verde: arquitectura", fontsize=11, loc="left")
    ax.legend(ncol=5, frameon=False, fontsize=8.5, loc="upper center",
              bbox_to_anchor=(0.5, -0.08))

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_deltas.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig3_deltas.png")
    return True


def fig_curvas(results: list[dict], out_dir: Path) -> bool:
    """Curvas de entrenamiento. Sirve para mostrar que el early stopping actuo y
    que ninguna configuracion quedo sobreajustada."""
    with_history = [r for r in results if r.get("history")]
    if not with_history:
        print("  fig4: no hay historias de entrenamiento, se omite")
        return False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for r in with_history:
        h = pd.DataFrame(r["history"])
        color = CONFIG_COLOR.get(r["code"], C_BASE)
        ax1.plot(h["epoch"], h["train_loss"], "-", color=color,
                 label=CONFIG_LABEL.get(r["code"], r["code"]))
        ax1.plot(h["epoch"], h["val_loss"], "--", color=color, alpha=0.6)
        ax2.plot(h["epoch"], h["val_auc_competition_mean"], "-o",
                 color=color, markersize=3,
                 label=CONFIG_LABEL.get(r["code"], r["code"]))
        if r.get("best_epoch", -1) > 0:
            ax2.axvline(r["best_epoch"], color=color, lw=0.6, ls=":", alpha=0.5)

    ax1.set_xlabel("Época"); ax1.set_ylabel("Pérdida")
    ax1.set_title("Pérdida (continua: train · punteada: validación)",
                  fontsize=10, loc="left")
    ax2.set_xlabel("Época"); ax2.set_ylabel("AUC-ROC medio (5 patologías)")
    ax2.set_title("AUC en validación interna", fontsize=10, loc="left")
    ax2.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "fig4_curvas.png", bbox_inches="tight")
    plt.close(fig)
    print("  fig4_curvas.png")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera las figuras de resultados")
    parser.add_argument("--config", default="configs/e1_baseline.yaml")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="results/figuras")
    parser.add_argument("--only", type=int, choices=[1, 2, 3, 4], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = _load_results(Path(args.results_dir))

    print(f"Configuraciones con resultados: {[r['code'] for r in results] or 'ninguna'}")
    print(f"Escribiendo en {out_dir}/")

    made = 0
    if args.only in (None, 1):
        made += fig_composicion(cfg, out_dir)
    if args.only in (None, 2):
        made += fig_auc_config(results, out_dir)
    if args.only in (None, 3):
        made += fig_deltas(results, out_dir)
    if args.only in (None, 4):
        made += fig_curvas(results, out_dir)

    print(f"\n{made} figura(s) generada(s).")


if __name__ == "__main__":
    main()
