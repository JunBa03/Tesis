"""Consolida los resultados de E1-E6 en las tablas comparativas del informe.

Genera:
  results/comparison/comparison_table.csv   - AUC por patologia y configuracion
  results/comparison/comparison_table.md    - misma tabla en Markdown/LaTeX-friendly
  results/comparison/ablation_deltas.csv    - delta de cada configuracion vs E1
  results/comparison/summary.json

Uso:
    python scripts/compare_results.py --results-dir results
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.labels import COMPETITION_TASKS, REFERENCE_AUC, SHORT_NAMES
from src.utils import load_json, save_json, setup_logging

# Que variable modifica cada configuracion respecto al baseline E1.
ABLATION_AXIS = {
    "E1": ("baseline", "-"),
    "E2": ("incertidumbre", "U-Ones"),
    "E3": ("incertidumbre", "U-Ignore"),
    "E4": ("perdida", "BCE ponderada"),
    "E5": ("perdida", "Focal loss"),
    "E6": ("arquitectura", "ResNet-50"),
}


def collect(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(results_dir.glob("*/results.json")):
        data = load_json(path)
        test = data.get("test_official")
        if test is None:
            continue
        code = data["experiment"].split("_")[0].upper()
        axis, value = ABLATION_AXIS.get(code, ("-", "-"))
        row = {
            "code": code,
            "experiment": data["experiment"],
            "variable_modificada": axis,
            "valor": value,
            "backbone": data["config"]["model"]["backbone"],
            "incertidumbre": data["config"]["labels"]["uncertainty_policy"],
            "perdida": data["config"]["loss"]["name"],
            "best_epoch": data.get("best_epoch"),
            "val_auc_mean": data.get("best_val_auc_competition_mean"),
        }
        for task in COMPETITION_TASKS:
            row[SHORT_NAMES[task]] = test["auc_competition"].get(task)
        row["Media"] = test["auc_competition_mean"]
        row["Media(14)"] = test["auc_all_mean"]
        rows.append(row)

    if not rows:
        raise SystemExit(
            f"No se encontraron results.json en {results_dir}. "
            "Ejecutar scripts/train.py para al menos una configuracion."
        )
    return pd.DataFrame(rows).sort_values("code").reset_index(drop=True)


def reference_frame() -> pd.DataFrame:
    rows = []
    for label, values in REFERENCE_AUC.items():
        row = {"code": "REF", "experiment": label, "variable_modificada": "referencia",
               "valor": "-", "backbone": "-", "incertidumbre": "-", "perdida": "-",
               "best_epoch": None, "val_auc_mean": None}
        for task in COMPETITION_TASKS:
            row[SHORT_NAMES[task]] = values[task]
        row["Media"] = values["mean"]
        row["Media(14)"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def deltas_vs_baseline(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["code"] == "E1"]
    if base.empty:
        raise SystemExit("Falta E1: sin baseline no hay ablacion que reportar.")
    base = base.iloc[0]
    cols = [SHORT_NAMES[t] for t in COMPETITION_TASKS] + ["Media"]
    out = df[df["code"] != "E1"].copy()
    for col in cols:
        out[f"d_{col}"] = out[col].astype(float) - float(base[col])
    keep = ["code", "experiment", "variable_modificada", "valor"] + [f"d_{c}" for c in cols]
    return out[keep].reset_index(drop=True)


def to_markdown(df: pd.DataFrame) -> str:
    cols = ["code", "experiment", "variable_modificada", "valor"] + \
           [SHORT_NAMES[t] for t in COMPETITION_TASKS] + ["Media"]
    view = df[cols].copy()
    for col in cols[4:]:
        view[col] = view[col].map(lambda v: f"{v:.3f}" if pd.notna(v) else "n/d")
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = "\n".join("| " + " | ".join(str(v) for v in row) + " |"
                     for row in view.itertuples(index=False))
    return "\n".join([header, sep, body])


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolida los resultados E1-E6")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--no-reference", action="store_true",
                        help="omitir los valores de referencia de Irvin et al.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = results_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(out_dir / "compare.log")

    df = collect(results_dir)
    if args.no_reference:
        full = df
    else:
        ref = reference_frame().reindex(columns=df.columns)
        full = pd.concat([df, ref.dropna(axis=1, how="all")], ignore_index=True)
        full = full.reindex(columns=df.columns)

    full.to_csv(out_dir / "comparison_table.csv", index=False)
    (out_dir / "comparison_table.md").write_text(to_markdown(full), encoding="utf-8")

    if (df["code"] == "E1").any() and len(df) > 1:
        deltas = deltas_vs_baseline(df)
        deltas.to_csv(out_dir / "ablation_deltas.csv", index=False)
        logger.info("\n%s", deltas.to_string(index=False))

    best = df.loc[df["Media"].idxmax()] if df["Media"].notna().any() else None
    save_json(
        {
            "n_experiments": int(len(df)),
            "best_configuration": None if best is None else str(best["experiment"]),
            "best_mean_auc": None if best is None else float(best["Media"]),
            "table": full.to_dict(orient="records"),
        },
        out_dir / "summary.json",
    )

    logger.info("\n%s", to_markdown(full))
    logger.info("Tablas escritas en %s", out_dir)


if __name__ == "__main__":
    main()
