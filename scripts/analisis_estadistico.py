"""Intervalos de confianza y comparaciones pareadas entre configuraciones.

Responde dos preguntas que el AUC puntual no contesta:

1. ¿Cuánta incertidumbre tiene cada AUC medido sobre ~200 imágenes?  Bootstrap pareado:
   se remuestrean las imágenes del conjunto oficial con reposición (las mismas para todas
   las configuraciones) y se recalcula el AUC en cada remuestreo.
2. ¿La diferencia entre una configuración y el baseline supera ese ruido?  Se reporta la
   diferencia con su intervalo del 95 % y un p-valor bootstrap bilateral, con corrección de
   Holm por comparaciones múltiples.

Además contrasta las decisiones de la configuración aplicada E7 con la validación interna
(sin mirar el conjunto oficial) y, si existen corridas con otras semillas
(``results/<experimento>_seedNN``), agrega la variabilidad por semilla.

No necesita GPU ni los modelos: lee ``results/predicciones/{test,val}.npz`` que genera
``scripts/exportar_predicciones.py``.

    python scripts/analisis_estadistico.py
    python scripts/analisis_estadistico.py --b 5000
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

RAIZ = Path(__file__).resolve().parents[1]

# labels.py se carga directo por ruta para no arrastrar torch (src/data/__init__ lo importa).
_spec = importlib.util.spec_from_file_location("labels", RAIZ / "src" / "data" / "labels.py")
_labels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_labels)
OBSERVATIONS = _labels.OBSERVATIONS
COMPETITION_TASKS = _labels.COMPETITION_TASKS
COMP_IDX = [OBSERVATIONS.index(t) for t in COMPETITION_TASKS]
CORTO = {"Atelectasis": "Ate.", "Cardiomegaly": "Card.", "Consolidation": "Con.",
         "Edema": "Ede.", "Pleural Effusion": "Der."}


# ----------------------------------------------------------------------------- métricas
def auc_binaria(y: np.ndarray, s: np.ndarray) -> float:
    """AUC-ROC por Mann-Whitney (equivalente a sklearn, más rápido en el bootstrap)."""
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def aucs_benchmark(y, s, m, idx=None) -> np.ndarray:
    """AUC de las 5 patologías del benchmark (en ``idx`` si se da un remuestreo)."""
    if idx is not None:
        y, s, m = y[idx], s[idx], m[idx]
    out = []
    for j in COMP_IDX:
        v = m[:, j] > 0
        out.append(auc_binaria(y[v, j], s[v, j]))
    return np.array(out)


def ic95(valores) -> tuple[float, float]:
    v = np.asarray(valores, dtype=float)
    v = v[~np.isnan(v)]
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def holm(pvals: dict) -> dict:
    """Corrección de Holm-Bonferroni."""
    orden = sorted(pvals, key=pvals.get)
    k, ajustados, previo = len(orden), {}, 0.0
    for i, clave in enumerate(orden):
        previo = max(previo, min(1.0, (k - i) * pvals[clave]))
        ajustados[clave] = previo
    return ajustados


# ----------------------------------------------------------------------------- análisis
def analizar_test(d: dict, b: int, semilla: int) -> dict:
    y, m = d["y_true"], d["mask"]
    ids = sorted(k[len("score_"):] for k in d if k.startswith("score_"))
    ablacion = [e for e in ids if e != "E7"]

    variantes = {e: d[f"score_{e}"] for e in ids}
    variantes["Ensamblado E1-E6 (promedio)"] = np.mean([d[f"score_{e}"] for e in ablacion], 0)
    variantes["Ensamblado E1-E7 (promedio)"] = np.mean([d[f"score_{e}"] for e in ids], 0)
    variantes["Ensamblado E1-E7 + TTA"] = np.mean([d[f"tta_{e}"] for e in ids], 0)
    if "E7" in ids:
        variantes["E7 + TTA"] = d["tta_E7"]

    rng = np.random.default_rng(semilla)
    n = len(y)
    muestras = [rng.integers(0, n, n) for _ in range(b)]

    puntual = {k: aucs_benchmark(y, s, m) for k, s in variantes.items()}
    boot = {k: np.array([aucs_benchmark(y, s, m, idx) for idx in muestras])
            for k, s in variantes.items()}

    resumen = {}
    for k in variantes:
        medias = np.nanmean(boot[k], axis=1)
        resumen[k] = {
            "auc_media": round(float(np.nanmean(puntual[k])), 4),
            "ic95": [round(x, 4) for x in ic95(medias)],
            "por_patologia": {CORTO[t]: {"auc": round(float(puntual[k][i]), 4),
                                         "ic95": [round(x, 4) for x in ic95(boot[k][:, i])]}
                              for i, t in enumerate(COMPETITION_TASKS)},
        }

    # Diferencias pareadas contra el baseline E1.
    deltas, pvals = {}, {}
    base_boot = np.nanmean(boot["E1"], axis=1)
    for e in [x for x in ablacion if x != "E1"]:
        dif = np.nanmean(boot[e], axis=1) - base_boot
        p = 2 * min((dif <= 0).mean(), (dif >= 0).mean())
        pvals[e] = float(min(1.0, p))
        dif_pat = boot[e] - boot["E1"]
        deltas[e] = {
            "delta": round(float(np.nanmean(puntual[e]) - np.nanmean(puntual["E1"])), 4),
            "ic95": [round(x, 4) for x in ic95(dif)],
            "p_bootstrap": round(pvals[e], 4),
            "por_patologia": {
                CORTO[t]: {"delta": round(float(puntual[e][i] - puntual["E1"][i]), 4),
                           "ic95": [round(x, 4) for x in ic95(dif_pat[:, i])]}
                for i, t in enumerate(COMPETITION_TASKS)},
        }
    for e, p_adj in holm(pvals).items():
        deltas[e]["p_holm"] = round(p_adj, 4)
        deltas[e]["significativo_5pct"] = bool(p_adj < 0.05)

    # ¿Cuánto suma el TTA y la inclusión de E7 sobre el ensamblado de la ablación?
    extra = {}
    for a, b_ in [("Ensamblado E1-E6 (promedio)", "Ensamblado E1-E7 (promedio)"),
                  ("Ensamblado E1-E7 (promedio)", "Ensamblado E1-E7 + TTA"),
                  ("E1", "Ensamblado E1-E6 (promedio)")]:
        dif = np.nanmean(boot[b_], axis=1) - np.nanmean(boot[a], axis=1)
        extra[f"{b_} vs {a}"] = {
            "delta": round(float(np.nanmean(puntual[b_]) - np.nanmean(puntual[a])), 4),
            "ic95": [round(x, 4) for x in ic95(dif)]}

    return {"n_imagenes": int(n), "b": b, "variantes": resumen,
            "deltas_vs_E1": deltas, "componentes_del_sistema": extra}


def etiquetas_crudas(paths) -> np.ndarray | None:
    """Etiquetas originales del CSV de entrenamiento (con -1 = incierta) para ``paths``."""
    import pandas as pd
    for raiz in (RAIZ.parent / "data" / "raw", RAIZ / "data" / "raw"):
        csv = raiz / "CheXpert-v1.0-small" / "train.csv"
        if csv.exists():
            df = pd.read_csv(csv).set_index("Path")
            return df.loc[list(paths), OBSERVATIONS].to_numpy(dtype=float)
    return None


def analizar_val(d: dict) -> dict:
    """AUC de cada configuración en la validación interna, con dos criterios de etiqueta.

    La validación interna se etiqueta con U-Zeros (la incierta cuenta como negativa) para
    todas las configuraciones. Eso penaliza a los modelos entrenados con U-Ones, que
    aprendieron a dar probabilidad alta en esos casos. Por eso se reporta además el AUC
    usando solo las etiquetas ciertas (se excluyen las -1 y las vacías), que no favorece a
    ninguna política.
    """
    ids = sorted(k[len("score_"):] for k in d if k.startswith("score_"))
    y, m = d["y_E1"], d["mask_E1"]
    crudas = etiquetas_crudas(d["paths"])
    out = {}
    for e in ids:
        a_uz = aucs_benchmark(y, d[f"score_{e}"], m)
        fila = {"auc_media_uzeros": round(float(np.nanmean(a_uz)), 4)}
        if crudas is not None:
            ciertas = ((crudas == 0) | (crudas == 1)).astype(float)
            a_c = aucs_benchmark(np.nan_to_num(crudas), d[f"score_{e}"], ciertas)
            fila["auc_media"] = round(float(np.nanmean(a_c)), 4)
            fila.update({CORTO[t]: round(float(a_c[i]), 4)
                         for i, t in enumerate(COMPETITION_TASKS)})
        else:
            fila["auc_media"] = fila["auc_media_uzeros"]
        out[e] = fila
    return {"n_imagenes": int(len(y)),
            "criterio": "auc_media = solo etiquetas ciertas; auc_media_uzeros = como en el entrenamiento",
            "aucs": out}


def analizar_semillas() -> dict:
    """Agrupa las corridas ``<experimento>`` y ``<experimento>_seedNN`` de results/."""
    grupos = {}
    for rj in sorted((RAIZ / "results").glob("E*/results.json")):
        nombre = rj.parent.name
        base = nombre.split("_seed")[0]
        datos = json.loads(rj.read_text(encoding="utf-8"))
        test = (datos.get("test_official") or {}).get("auc_competition_mean")
        val = datos.get("best_val_auc_competition_mean")
        semilla = datos.get("config", {}).get("experiment", {}).get("seed")
        grupos.setdefault(base, []).append({"corrida": nombre, "semilla": semilla,
                                            "test": test, "val": val})
    out = {}
    for base, corridas in grupos.items():
        if len(corridas) < 2:
            continue
        t = np.array([c["test"] for c in corridas if c["test"] is not None])
        out[base] = {"corridas": corridas, "n": len(t),
                     "test_media": round(float(t.mean()), 4),
                     "test_desvio": round(float(t.std(ddof=1)), 4),
                     "test_rango": [round(float(t.min()), 4), round(float(t.max()), 4)]}
    return out


def tabla_markdown(r: dict) -> str:
    t = r["test"]
    lineas = [f"# Análisis estadístico — conjunto oficial ({t['n_imagenes']} imágenes, "
              f"{t['b']} remuestreos bootstrap)", "",
              "| Variante | AUC medio | IC 95 % |", "|---|---|---|"]
    for k, v in t["variantes"].items():
        lineas.append(f"| {k} | {v['auc_media']:.4f} | [{v['ic95'][0]:.3f}; {v['ic95'][1]:.3f}] |")
    lineas += ["", "## Diferencia contra el baseline E1 (pareada)", "",
               "| Config. | Δ AUC | IC 95 % | p | p (Holm) |", "|---|---|---|---|---|"]
    for e, v in t["deltas_vs_E1"].items():
        lineas.append(f"| {e} | {v['delta']:+.4f} | [{v['ic95'][0]:+.3f}; {v['ic95'][1]:+.3f}] "
                      f"| {v['p_bootstrap']:.3f} | {v['p_holm']:.3f} |")
    lineas += ["", "## Componentes del sistema", ""]
    for k, v in t["componentes_del_sistema"].items():
        lineas.append(f"- {k}: Δ {v['delta']:+.4f} [{v['ic95'][0]:+.3f}; {v['ic95'][1]:+.3f}]")
    if r.get("val"):
        lineas += ["", f"## Validación interna ({r['val']['n_imagenes']} imágenes)", "",
                   "| Config. | AUC (solo etiquetas ciertas) | AUC (U-Zeros, como al entrenar) |",
                   "|---|---|---|"]
        for e, v in sorted(r["val"]["aucs"].items(), key=lambda kv: -kv[1]["auc_media"]):
            lineas.append(f"| {e} | {v['auc_media']:.4f} | {v['auc_media_uzeros']:.4f} |")
    if r.get("semillas"):
        lineas += ["", "## Variabilidad por semilla", "",
                   "| Configuración | Corridas | Media | Desvío | Rango |", "|---|---|---|---|---|"]
        for base, v in r["semillas"].items():
            lineas.append(f"| {base} | {v['n']} | {v['test_media']:.4f} | {v['test_desvio']:.4f} "
                          f"| [{v['test_rango'][0]:.4f}; {v['test_rango'][1]:.4f}] |")
    return "\n".join(lineas) + "\n"


def figura(r: dict, ruta: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    v = r["test"]["variantes"]
    nombres = list(v)[::-1]
    medias = [v[k]["auc_media"] for k in nombres]
    lo = [v[k]["auc_media"] - v[k]["ic95"][0] for k in nombres]
    hi = [v[k]["ic95"][1] - v[k]["auc_media"] for k in nombres]
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(nombres) + 1.2))
    colores = ["#1D4ED8" if "Ensamblado" in k or "E7" in k else "#64748B" for k in nombres]
    for i, (x, a, b_, c) in enumerate(zip(medias, lo, hi, colores)):
        ax.errorbar([x], [i], xerr=[[a], [b_]], fmt="o", color=c, ecolor=c,
                    elinewidth=2, capsize=3, zorder=3)
    ax.axvline(v["E1"]["auc_media"], color="#94A3B8", ls="--", lw=1)
    ax.set_yticks(range(len(nombres)), nombres)
    ax.set_xlabel("AUC-ROC medio (5 patologías) · IC 95 % bootstrap")
    ax.grid(axis="x", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="IC bootstrap y comparaciones pareadas")
    ap.add_argument("--b", type=int, default=2000, help="cantidad de remuestreos")
    ap.add_argument("--semilla", type=int, default=2026)
    args = ap.parse_args()

    carpeta = RAIZ / "results" / "predicciones"
    test = dict(np.load(carpeta / "test.npz", allow_pickle=True))
    r = {"test": analizar_test(test, args.b, args.semilla)}
    if (carpeta / "val.npz").exists():
        r["val"] = analizar_val(dict(np.load(carpeta / "val.npz", allow_pickle=True)))
    r["semillas"] = analizar_semillas()

    salida = RAIZ / "results" / "estadistica"
    salida.mkdir(parents=True, exist_ok=True)
    (salida / "resumen.json").write_text(json.dumps(r, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    (salida / "tabla.md").write_text(tabla_markdown(r), encoding="utf-8")
    figura(r, salida / "ic95_configuraciones.png")
    print(tabla_markdown(r))
    print(f"Guardado en {salida}")


if __name__ == "__main__":
    main()
