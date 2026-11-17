"""Predice sobre una imagen y escribe un informe HTML. Sin servidor ni interfaz.

Produce el mismo análisis que la aplicación en un archivo HTML, sin necesidad de
levantar Streamlit.

    python scripts/predict_image.py --imagen ruta\\a\\radiografia.jpg
    python scripts/predict_image.py --imagen foto.jpg --checkpoint checkpoints/E2_uones.pt
"""
from __future__ import annotations

import argparse
import base64
import io
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from src.data.labels import COMPETITION_TASKS
from src.inference import Clasificador, buscar_modelos, cargar_umbrales

PLANTILLA = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Análisis de radiografía — RXTórax</title>
<style>
  body {{ margin:0; background:#f7f8fa; color:#1f2937; font-size:14px;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:980px; margin:0 auto; padding:28px 24px 50px; }}
  h1 {{ font-size:20px; margin:0 0 4px; }}
  .meta {{ color:#6b7280; font-size:12.5px; }}
  .aviso {{ background:#fef3e2; border:1px solid #f6d9a8; border-radius:8px;
           padding:11px 14px; font-size:13px; color:#8a5108; margin:18px 0; }}
  .cols {{ display:grid; grid-template-columns:1fr 1.15fr; gap:22px; align-items:start; }}
  @media (max-width:820px) {{ .cols {{ grid-template-columns:1fr; }} }}
  .card {{ background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:16px; }}
  img {{ width:100%; border-radius:6px; display:block; }}
  h2 {{ font-size:14px; margin:0 0 10px; }}
  .obs {{ display:flex; justify-content:space-between; align-items:center;
         padding:7px 0; border-bottom:1px solid #f3f4f6; }}
  .obs:last-child {{ border-bottom:none; }}
  .bench {{ font-weight:700; }}
  .valor {{ font-variant-numeric:tabular-nums; font-weight:600; width:60px;
           text-align:right; }}
  .barra {{ height:7px; background:#f3f4f6; border-radius:4px; margin-top:3px; }}
  .barra > div {{ height:100%; border-radius:4px; }}
  table.info {{ font-size:12.5px; color:#6b7280; }}
  table.info td {{ padding:2px 12px 2px 0; }}
</style></head><body><div class="wrap">
  <h1>Análisis automatizado de radiografías de tórax</h1>
  <div class="meta">Proyecto Final de Ingeniería · Juan Bautista Felizia · {fecha}</div>
  <div class="aviso"><b>Herramienta de investigación, no de uso clínico.</b>
    Devuelve una probabilidad por observación y no emite diagnósticos.
    Entrenada y evaluada exclusivamente sobre CheXpert, sin validación clínica.</div>
  <div class="cols">
    <div class="card">
      <h2>Radiografía</h2>
      <img src="data:image/png;base64,{imagen}" alt="radiografía">
      <div class="meta" style="margin-top:8px">{archivo}</div>
    </div>
    <div>
      <div class="card" style="margin-bottom:14px">
        <h2>Patologías del benchmark</h2>{benchmark}
      </div>
      <div class="card" style="margin-bottom:14px">
        <h2>Resto de las observaciones</h2>{resto}
      </div>
      <div class="card">
        <h2>Modelo</h2>
        <table class="info">
          <tr><td>Configuración</td><td>{experimento}</td></tr>
          <tr><td>Arquitectura</td><td>{backbone}</td></tr>
          <tr><td>Etiquetas inciertas</td><td>{incertidumbre}</td></tr>
          <tr><td>Función de pérdida</td><td>{perdida}</td></tr>
          <tr><td>Época seleccionada</td><td>{epoca}</td></tr>
          <tr><td>AUC-ROC en validación interna</td><td>{auc}</td></tr>
        </table>
      </div>
    </div>
  </div>
</div></body></html>
"""


def fila(obs: str, prob: float, umbral: float) -> str:
    destacado = obs in COMPETITION_TASKS
    color = "#d97706" if prob >= umbral else ("#9ca3af" if not destacado else "#cbd5e1")
    return (
        f'<div class="obs"><div style="flex:1">'
        f'<span class="{"bench" if destacado else ""}">{obs}</span>'
        f'<div class="barra"><div style="width:{prob*100:.1f}%;background:{color}"></div></div>'
        f'</div><div class="valor">{prob:.3f}</div></div>'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Informe HTML de una radiografía")
    parser.add_argument("--imagen", required=True)
    parser.add_argument("--checkpoint", default=None,
                        help="por defecto, el primero de checkpoints/")
    parser.add_argument("--salida", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no-abrir", action="store_true")
    args = parser.parse_args()

    ruta_img = Path(args.imagen)
    if not ruta_img.exists():
        print(f"No existe la imagen: {ruta_img}")
        sys.exit(1)

    ckpt = args.checkpoint
    if ckpt is None:
        disponibles = buscar_modelos(Path(__file__).resolve().parents[1])
        if not disponibles:
            print("No hay modelos en models/ ni en checkpoints/.")
            sys.exit(1)
        ckpt = disponibles[0]
        print(f"Usando {Path(ckpt).name}")

    clf = Clasificador(ckpt, device=args.device,
                       umbrales=cargar_umbrales(Path(ckpt).stem))
    imagen = Image.open(ruta_img)
    pred = clf.predecir(imagen)

    # La imagen se embebe en base64 para que el HTML sea un solo archivo portable.
    vista = imagen.convert("RGB")
    vista.thumbnail((640, 640))
    buf = io.BytesIO()
    vista.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    bench = "".join(fila(o, p, pred.umbral(o))
                    for o, p in pred.ordenadas(solo_benchmark=True))
    resto = "".join(fila(o, p, pred.umbral(o)) for o, p in pred.ordenadas()
                    if o not in COMPETITION_TASKS)

    html = PLANTILLA.format(
        fecha=datetime.now().strftime("%d/%m/%Y %H:%M"),
        imagen=b64, archivo=ruta_img.name,
        benchmark=bench, resto=resto,
        experimento=pred.experimento, backbone=pred.backbone,
        incertidumbre=pred.incertidumbre, perdida=pred.perdida,
        epoca=pred.epoca or "—",
        auc=f"{pred.auc_validacion:.4f}" if pred.auc_validacion else "—",
    )

    salida = Path(args.salida or f"results/informe_{ruta_img.stem}.html")
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(html, encoding="utf-8")
    print(f"Informe: {salida.resolve()}")

    print("\n  Patologías del benchmark:")
    for obs, p in pred.ordenadas(solo_benchmark=True):
        marca = "*" if p >= pred.umbral(obs) else " "
        print(f"   {marca} {obs:<20} {p:.3f}")

    if not args.no_abrir:
        webbrowser.open(salida.resolve().as_uri())


if __name__ == "__main__":
    main()
