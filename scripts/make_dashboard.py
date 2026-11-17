"""Panel de experimentos: un HTML autocontenido para seguir y comparar E1-E6.

No es una aplicacion clinica ni forma parte del alcance del trabajo: es una
herramienta de analisis. Sirve para ver en que estado esta cada configuracion,
si alguna se esta sobreajustando, y como se comparan los AUC por patologia sin
tener que abrir seis archivos JSON a mano.

Lee, por cada configuracion:
  results/<experimento>/progress.json   mientras entrena (escrito por epoca)
  results/<experimento>/results.json    cuando termino

Los datos quedan embebidos en el HTML, asi que el archivo funciona sin servidor,
sin internet y se puede mandar por mail o abrir desde un pendrive.

Uso:
    python scripts/make_dashboard.py
    python scripts/make_dashboard.py --open        # y abrirlo en el navegador
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.io import read_chexpert_csv, resolve_dataset_root
from src.data.labels import (COMPETITION_TASKS, OBSERVATIONS, REFERENCE_AUC,
                             SHORT_NAMES)

ORDEN = ["E1", "E2", "E3", "E4", "E5", "E6"]

META = {
    "E1": {"eje": "base", "variable": "— (baseline)", "valor": "DenseNet-121 · U-Zeros · BCE"},
    "E2": {"eje": "inc", "variable": "Incertidumbre", "valor": "U-Ones"},
    "E3": {"eje": "inc", "variable": "Incertidumbre", "valor": "U-Ignore"},
    "E4": {"eje": "loss", "variable": "Función de pérdida", "valor": "BCE ponderada"},
    "E5": {"eje": "loss", "variable": "Función de pérdida", "valor": "Focal loss"},
    "E6": {"eje": "arch", "variable": "Arquitectura", "valor": "ResNet-50"},
}


def recolectar(results_dir: Path) -> list[dict]:
    """Une progress.json y results.json de cada configuracion en un solo registro."""
    porcodigo: dict[str, dict] = {}

    for sub in sorted(results_dir.glob("*/")):
        code = sub.name.split("_")[0].upper()
        if code not in META:
            continue
        reg = porcodigo.setdefault(code, {"code": code, **META[code],
                                          "status": "pendiente", "history": [],
                                          "auc": None, "auc_mean": None,
                                          "epoch": 0, "total_epochs": None,
                                          "best_epoch": None, "experiment": sub.name})

        prog = sub / "progress.json"
        if prog.exists():
            try:
                d = json.loads(prog.read_text(encoding="utf-8"))
                reg.update({
                    "status": d.get("status", "corriendo"),
                    "history": d.get("history", []),
                    "epoch": d.get("epoch", 0),
                    "total_epochs": d.get("total_epochs"),
                    "best_epoch": d.get("best_epoch"),
                    "val_auc": d.get("best_val_auc_competition_mean"),
                    "updated_at": d.get("updated_at"),
                })
            except json.JSONDecodeError:
                pass

        res = sub / "results.json"
        if res.exists():
            try:
                d = json.loads(res.read_text(encoding="utf-8"))
                reg["history"] = d.get("history", reg["history"])
                reg["best_epoch"] = d.get("best_epoch", reg["best_epoch"])
                reg["val_auc"] = d.get("best_val_auc_competition_mean", reg.get("val_auc"))
                test = d.get("test_official")
                if test:
                    reg["status"] = "completa"
                    reg["auc"] = {t: test["auc_competition"].get(t) for t in COMPETITION_TASKS}
                    reg["auc_mean"] = test.get("auc_competition_mean")
                    reg["auc_all"] = test.get("auc_all_mean")
                    reg["auc_por_obs"] = test.get("auc_per_observation")
            except json.JSONDecodeError:
                pass

    for code in ORDEN:
        porcodigo.setdefault(code, {"code": code, **META[code], "status": "pendiente",
                                    "history": [], "auc": None, "auc_mean": None,
                                    "epoch": 0, "total_epochs": None,
                                    "best_epoch": None, "experiment": code})
    return [porcodigo[c] for c in ORDEN]


def composicion(cfg) -> list[dict] | None:
    """Positivos / negativos / inciertos por observacion, sobre el dataset real."""
    try:
        root = resolve_dataset_root(cfg["data"]["root"])
        df = read_chexpert_csv(root, cfg["data"]["train_csv"])
    except FileNotFoundError:
        return None
    if cfg["data"]["frontal_only"]:
        df = df[df["Frontal/Lateral"].str.strip().str.lower() == "frontal"]
    n = len(df)
    out = []
    for obs in OBSERVATIONS:
        col = df[obs]
        pos, inc = int((col == 1).sum()), int((col == -1).sum())
        out.append({"obs": obs, "pos": pos, "inc": inc, "n": n,
                    "pct_inc": round(100 * inc / n, 2),
                    "prev": round(pos / n, 4),
                    "bench": obs in COMPETITION_TASKS})
    return out


# ------------------------------------------------------------------------ HTML

HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Panel de experimentos — RXTórax</title>
<style>
  :root {
    --ink:#1f2937; --ink-soft:#374151; --muted:#6b7280; --line:#e5e7eb;
    --paper:#ffffff; --bg:#f7f8fa; --amber:#d97706; --blue:#1d4ed8;
    --green:#059669; --base:#6b7280; --amber-soft:#fef3e2;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         font-size:14px; line-height:1.5; }
  .wrap { max-width:1180px; margin:0 auto; padding:28px 24px 60px; }
  header { border-bottom:1px solid var(--line); padding-bottom:18px; margin-bottom:24px; }
  h1 { font-size:21px; margin:0 0 4px; font-weight:700; letter-spacing:-.01em; }
  .sub { color:var(--muted); font-size:13px; }
  h2 { font-size:15px; margin:32px 0 12px; font-weight:700; }
  h2 .hint { font-weight:400; color:var(--muted); font-size:12.5px; margin-left:8px; }
  .card { background:var(--paper); border:1px solid var(--line); border-radius:10px; padding:18px; }
  .grid6 { display:grid; grid-template-columns:repeat(auto-fit,minmax(178px,1fr)); gap:12px; }
  .exp { background:var(--paper); border:1px solid var(--line); border-radius:10px;
         padding:14px; position:relative; }
  .exp.activo { border-color:var(--amber); }
  .chip { display:inline-flex; align-items:center; justify-content:center;
          width:30px; height:30px; border-radius:50%; color:#fff;
          font-weight:700; font-size:12px; }
  .exp .var { font-size:12.5px; font-weight:600; margin-top:9px; }
  .exp .val { font-size:12px; color:var(--muted); }
  .estado { position:absolute; top:14px; right:14px; font-size:10.5px; font-weight:600;
            padding:3px 8px; border-radius:20px; letter-spacing:.02em; }
  .st-pendiente { background:#f3f4f6; color:var(--muted); }
  .st-corriendo { background:var(--amber-soft); color:var(--amber); }
  .st-completa  { background:#e8f5f0; color:var(--green); }
  .barra { height:4px; background:var(--line); border-radius:3px; margin-top:11px; overflow:hidden; }
  .barra > div { height:100%; background:var(--amber); border-radius:3px; }
  .metric { margin-top:10px; font-size:19px; font-weight:700; font-variant-numeric:tabular-nums; }
  .metric small { font-size:11px; font-weight:400; color:var(--muted); display:block;
                  margin-top:-2px; letter-spacing:.01em; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { padding:8px 10px; text-align:right; font-variant-numeric:tabular-nums; }
  th:first-child, td:first-child { text-align:left; }
  thead th { font-size:11px; text-transform:uppercase; letter-spacing:.04em;
             color:var(--muted); font-weight:600; border-bottom:1px solid var(--line); }
  tbody tr:nth-child(even) { background:#fafbfc; }
  tbody tr.ref td { color:var(--muted); font-style:italic; }
  td.best { font-weight:700; }
  .pos { color:var(--green); } .neg { color:#b91c1c; }
  .vacio { color:#c9ced6; }
  .leyenda { display:flex; gap:18px; flex-wrap:wrap; font-size:12.5px; color:var(--muted);
             margin:10px 0 0; }
  .leyenda span { display:flex; align-items:center; gap:6px; }
  .dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
  .nota { background:var(--amber-soft); border:1px solid #f6d9a8; border-radius:8px;
          padding:12px 14px; font-size:12.5px; color:#8a5108; margin-top:14px; }
  .dos { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  @media (max-width:900px){ .dos { grid-template-columns:1fr; } }
  svg text { font-family:inherit; }
  footer { margin-top:44px; padding-top:16px; border-top:1px solid var(--line);
           color:var(--muted); font-size:12px; }
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Panel de experimentos — ablación controlada sobre CheXpert</h1>
    <div class="sub">Proyecto Final de Ingeniería · Juan Bautista Felizia · generado el __FECHA__</div>
  </header>

  <h2>Estado de las seis configuraciones</h2>
  <div class="grid6" id="tarjetas"></div>
  <div class="leyenda">
    <span><i class="dot" style="background:var(--base)"></i> baseline</span>
    <span><i class="dot" style="background:var(--amber)"></i> estrategia de incertidumbre</span>
    <span><i class="dot" style="background:var(--blue)"></i> función de pérdida</span>
    <span><i class="dot" style="background:var(--green)"></i> arquitectura</span>
  </div>

  <h2>Curvas de entrenamiento<span class="hint">línea continua: entrenamiento · punteada: validación interna</span></h2>
  <div class="dos">
    <div class="card"><div id="c-loss"></div></div>
    <div class="card"><div id="c-auc"></div></div>
  </div>

  <h2>AUC-ROC por patología<span class="hint">conjunto de validación oficial · 202 radiografías frontales, consenso de tres radiólogos</span></h2>
  <div class="card"><div id="tabla-auc"></div></div>

  <h2>Efecto aislado respecto del baseline<span class="hint">diferencia de AUC-ROC contra E1</span></h2>
  <div class="card"><div id="tabla-delta"></div></div>

  <h2>Composición del conjunto de entrenamiento<span class="hint">imágenes frontales</span></h2>
  <div class="card"><div id="tabla-comp"></div></div>

  <footer>
    Herramienta de análisis del estudio experimental. No es una aplicación clínica:
    el alcance del trabajo está acotado a clasificación a nivel de imagen sobre CheXpert,
    sin validación clínica ni integración con sistemas hospitalarios.
  </footer>
</div>

<script>
const DATA = __DATA__;
const COMP = __COMP__;
const TASKS = __TASKS__;
const SHORT = __SHORT__;
const REF = __REF__;
const COLOR = { base:"#6b7280", inc:"#d97706", loss:"#1d4ed8", arch:"#059669" };

const fmt = (v, d=3) => (v===null||v===undefined||Number.isNaN(v)) ? "—"
                        : v.toFixed(d).replace(".", ",");
const el = (h) => { const d=document.createElement("div"); d.innerHTML=h.trim(); return d.firstChild; };

/* ------------------------------------------------------------ tarjetas */
const cont = document.getElementById("tarjetas");
DATA.forEach(e => {
  const pct = e.total_epochs ? Math.round(100*e.epoch/e.total_epochs) : 0;
  const metrica = e.auc_mean ?? e.val_auc;
  const etiqueta = e.auc_mean ? "AUC oficial (5 patologías)" : (e.val_auc ? "AUC validación interna" : "sin datos");
  cont.appendChild(el(`
    <div class="exp ${e.status==='corriendo'?'activo':''}">
      <span class="estado st-${e.status}">${e.status}</span>
      <span class="chip" style="background:${COLOR[e.eje]}">${e.code}</span>
      <div class="var">${e.variable}</div>
      <div class="val">${e.valor}</div>
      ${e.total_epochs ? `<div class="barra"><div style="width:${pct}%"></div></div>
        <div class="val" style="margin-top:5px">época ${e.epoch} de ${e.total_epochs}</div>` : ``}
      <div class="metric">${metrica ? fmt(metrica) : '<span class="vacio">—</span>'}
        <small>${etiqueta}</small></div>
    </div>`));
});

/* ------------------------------------------------------- graficos SVG */
function linea(destino, series, yLabel, yDom) {
  const W=520, H=250, mL=52, mR=14, mT=14, mB=34;
  const todos = series.flatMap(s => s.puntos);
  if (!todos.length) {
    document.getElementById(destino).innerHTML =
      `<div style="height:210px;display:flex;align-items:center;justify-content:center;
       color:var(--muted);font-size:13px;text-align:center">Sin datos todavía.<br>
       Aparecen acá en cuanto termine la primera época.</div>`;
    return;
  }
  const xs = todos.map(p=>p[0]), ys = todos.map(p=>p[1]);
  const x0=Math.min(...xs), x1=Math.max(...xs, x0+1);
  let y0 = yDom ? yDom[0] : Math.min(...ys), y1 = yDom ? yDom[1] : Math.max(...ys);
  if (y1-y0 < 1e-9) { y1 = y0+1; }
  const pad=(y1-y0)*0.1; if(!yDom){ y0-=pad; y1+=pad; }
  const px = v => mL + (v-x0)/(x1-x0)*(W-mL-mR);
  const py = v => mT + (1-(v-y0)/(y1-y0))*(H-mT-mB);

  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img">`;
  for (let i=0;i<=4;i++){
    const v=y0+(y1-y0)*i/4, y=py(v);
    s += `<line x1="${mL}" y1="${y}" x2="${W-mR}" y2="${y}" stroke="#eef0f3"/>`;
    s += `<text x="${mL-8}" y="${y+4}" text-anchor="end" font-size="10" fill="#6b7280">${fmt(v,2)}</text>`;
  }
  for (let e=Math.ceil(x0); e<=x1; e++){
    s += `<text x="${px(e)}" y="${H-12}" text-anchor="middle" font-size="10" fill="#6b7280">${e}</text>`;
  }
  s += `<text x="${(W+mL)/2}" y="${H-1}" text-anchor="middle" font-size="10.5" fill="#6b7280">época</text>`;
  s += `<text x="12" y="${(H-mB)/2}" text-anchor="middle" font-size="10.5" fill="#6b7280"
        transform="rotate(-90 12 ${(H-mB)/2})">${yLabel}</text>`;

  series.forEach(se => {
    if (!se.puntos.length) return;
    const d = se.puntos.map((p,i)=>`${i?"L":"M"}${px(p[0])},${py(p[1])}`).join(" ");
    s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="1.9"
           ${se.dash?'stroke-dasharray="4 3"':''} opacity="${se.dash?0.65:1}"/>`;
    if (!se.dash) se.puntos.forEach(p =>
      s += `<circle cx="${px(p[0])}" cy="${py(p[1])}" r="2.6" fill="${se.color}"/>`);
  });
  s += `</svg>`;
  document.getElementById(destino).innerHTML = s;
}

const conHist = DATA.filter(e => e.history && e.history.length);
linea("c-loss", conHist.flatMap(e => ([
  { color: COLOR[e.eje], puntos: e.history.map(h=>[h.epoch, h.train_loss]) },
  { color: COLOR[e.eje], dash:true, puntos: e.history.map(h=>[h.epoch, h.val_loss]) },
])), "pérdida");
linea("c-auc", conHist.map(e => (
  { color: COLOR[e.eje], puntos: e.history.map(h=>[h.epoch, h.val_auc_competition_mean]) }
)), "AUC-ROC medio", [0.5, 1.0]);

/* --------------------------------------------------------- tabla AUC */
function tablaAuc() {
  let h = `<table><thead><tr><th>Configuración</th>`;
  TASKS.forEach(t => h += `<th>${SHORT[t]}</th>`);
  h += `<th>Media</th></tr></thead><tbody>`;
  DATA.forEach(e => {
    h += `<tr><td><span class="chip" style="background:${COLOR[e.eje]};width:22px;height:22px;
          font-size:10px;vertical-align:middle">${e.code}</span>
          <span style="margin-left:8px">${e.valor}</span></td>`;
    TASKS.forEach(t => {
      const v = e.auc ? e.auc[t] : null;
      h += `<td class="${v===null?'vacio':''}">${fmt(v)}</td>`;
    });
    h += `<td class="best">${fmt(e.auc_mean)}</td></tr>`;
  });
  Object.entries(REF).forEach(([nombre, vals]) => {
    h += `<tr class="ref"><td>${nombre}</td>`;
    TASKS.forEach(t => h += `<td>${fmt(vals[t])}</td>`);
    h += `<td>${fmt(vals["mean"])}</td></tr>`;
  });
  h += `</tbody></table>`;
  document.getElementById("tabla-auc").innerHTML = h;
}
tablaAuc();

/* ------------------------------------------------------- tabla deltas */
function tablaDelta() {
  const base = DATA.find(e => e.code==="E1");
  const destino = document.getElementById("tabla-delta");
  if (!base || !base.auc) {
    destino.innerHTML = `<div style="color:var(--muted);padding:20px 0;font-size:13px">
      Se completa cuando E1 tenga resultados: sin baseline no hay nada contra qué comparar.</div>`;
    return;
  }
  let h = `<table><thead><tr><th>Configuración</th><th>Eje</th>`;
  TASKS.forEach(t => h += `<th>${SHORT[t]}</th>`);
  h += `<th>Media</th></tr></thead><tbody>`;
  DATA.filter(e => e.code!=="E1").forEach(e => {
    h += `<tr><td><span class="chip" style="background:${COLOR[e.eje]};width:22px;height:22px;
          font-size:10px;vertical-align:middle">${e.code}</span>
          <span style="margin-left:8px">${e.valor}</span></td>
          <td style="text-align:left;color:var(--muted)">${e.variable}</td>`;
    let dm = null;
    TASKS.forEach(t => {
      const a = e.auc ? e.auc[t] : null, b = base.auc[t];
      const d = (a!==null && a!==undefined && b!==null) ? a-b : null;
      h += `<td class="${d===null?'vacio':(d>=0?'pos':'neg')}">${d===null?'—':(d>=0?'+':'')+fmt(d)}</td>`;
    });
    if (e.auc_mean!==null && base.auc_mean!==null) dm = e.auc_mean - base.auc_mean;
    h += `<td class="best ${dm===null?'vacio':(dm>=0?'pos':'neg')}">${dm===null?'—':(dm>=0?'+':'')+fmt(dm)}</td></tr>`;
  });
  h += `</tbody></table>
    <div class="nota">El signo puede cambiar entre patologías dentro de una misma
    configuración. Eso no es ruido: es exactamente lo que un AUC promedio global oculta,
    y es el resultado que este diseño permite ver.</div>`;
  destino.innerHTML = h;
}
tablaDelta();

/* ---------------------------------------------------- tabla composicion */
function tablaComp() {
  const destino = document.getElementById("tabla-comp");
  if (!COMP) {
    destino.innerHTML = `<div style="color:var(--muted);padding:20px 0;font-size:13px">
      No se encontró train.csv. Revisá data.root en configs/base.yaml.</div>`;
    return;
  }
  const orden = [...COMP].sort((a,b)=>b.pct_inc-a.pct_inc);
  const maxN = Math.max(...orden.map(o=>Math.max(o.pos,o.inc)));
  let h = `<table><thead><tr><th>Observación</th><th>Positivos</th><th>Inciertos</th>
           <th>% inciertos</th><th>Prevalencia</th><th style="width:210px">Positivos vs inciertos</th>
           </tr></thead><tbody>`;
  orden.forEach(o => {
    const wp = 100*o.pos/maxN, wi = 100*o.inc/maxN;
    h += `<tr><td style="${o.bench?'font-weight:700':''}">${o.obs}${o.bench?
          ' <span style="font-weight:400;color:var(--muted);font-size:11px">benchmark</span>':''}</td>
      <td>${o.pos.toLocaleString("es-AR")}</td>
      <td>${o.inc.toLocaleString("es-AR")}</td>
      <td>${o.pct_inc.toFixed(1).replace(".",",")} %</td>
      <td>${o.prev.toFixed(3).replace(".",",")}</td>
      <td><svg width="200" height="18" role="img">
        <rect x="0" y="2"  width="${wp*2}" height="6" fill="#1f2937" rx="1"/>
        <rect x="0" y="10" width="${wi*2}" height="6" fill="#d97706" rx="1"/>
      </svg></td></tr>`;
  });
  h += `</tbody></table>
    <div class="leyenda" style="margin-top:12px">
      <span><i class="dot" style="background:#1f2937"></i> positivos</span>
      <span><i class="dot" style="background:#d97706"></i> inciertos (−1)</span>
    </div>`;
  destino.innerHTML = h;
}
tablaComp();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el panel de experimentos")
    parser.add_argument("--config", default="configs/e1_baseline.yaml")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="results/panel.html")
    parser.add_argument("--open", action="store_true", help="abrirlo en el navegador")
    args = parser.parse_args()

    cfg = load_config(args.config)
    datos = recolectar(Path(args.results_dir))
    comp = composicion(cfg)

    ref = {nombre: {**{t: v[t] for t in COMPETITION_TASKS}, "mean": v["mean"]}
           for nombre, v in REFERENCE_AUC.items()}

    html = (HTML
            .replace("__FECHA__", datetime.now().strftime("%d/%m/%Y %H:%M"))
            .replace("__DATA__", json.dumps(datos, ensure_ascii=False))
            .replace("__COMP__", json.dumps(comp, ensure_ascii=False))
            .replace("__TASKS__", json.dumps(COMPETITION_TASKS, ensure_ascii=False))
            .replace("__SHORT__", json.dumps(SHORT_NAMES, ensure_ascii=False))
            .replace("__REF__", json.dumps(ref, ensure_ascii=False)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")

    estados = {}
    for d in datos:
        estados[d["status"]] = estados.get(d["status"], 0) + 1
    print(f"Panel generado: {out.resolve()}")
    print("  " + " · ".join(f"{v} {k}" for k, v in estados.items()))
    if comp is None:
        print("  Aviso: no se encontró train.csv, la sección de composición queda vacía.")

    if args.open:
        webbrowser.open(out.resolve().as_uri())


if __name__ == "__main__":
    main()
