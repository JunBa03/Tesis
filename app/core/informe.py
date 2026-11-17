"""Genera un informe HTML autocontenido de un análisis, para descargar o imprimir.

Todo va embebido en un único archivo (las imágenes en base64) así que el informe se
abre con doble clic en cualquier máquina, sin depender de la aplicación. Está pensado
como respaldo y como material para adjuntar, con el aviso de uso de investigación bien
visible.
"""
from __future__ import annotations

import base64
import io
from datetime import datetime

from PIL import Image

from src.data.labels import COMPETITION_TASKS

from . import registry


def _b64(imagen: Image.Image, lado: int = 620) -> str:
    vista = imagen.convert("RGB")
    vista.thumbnail((lado, lado))
    buf = io.BytesIO()
    vista.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _fila(obs, prob, umbral, nivel, verdad):
    color = "#b45309" if prob >= umbral else "#94a3b8"
    conf = {"alta": "alta", "media": "media", "baja": "baja"}[nivel]
    gt = ""
    if verdad is not None and verdad.get(obs) in (0.0, 1.0):
        gt = f' · radiólogos: {"positivo" if verdad[obs] == 1.0 else "negativo"}'
    destacado = "font-weight:700" if obs in COMPETITION_TASKS else ""
    return (
        f'<tr><td style="{destacado}">{registry.es(obs)}'
        f'<span style="color:#94a3b8;font-size:12px">{gt}</span></td>'
        f'<td style="text-align:right;font-variant-numeric:tabular-nums;color:{color};'
        f'font-weight:600">{prob:.3f}</td>'
        f'<td style="text-align:center;color:#64748b;font-size:12px">{conf}</td></tr>'
    )


def _reporte_html(rep: dict) -> str:
    """Informe estructurado con estilos claros, para el archivo descargable."""
    if not rep:
        return ""
    items = "".join(f"<li>{h}</li>" for h in rep["hallazgos"])
    sec = lambda t, c: (f'<div class="rsec"><div class="rrot">{t}</div>'
                        f'<div class="rval">{c}</div></div>')
    return ('<div class="reporte">'
            + sec("Técnica", rep["tecnica"])
            + f'<div class="rsec"><div class="rrot">Hallazgos</div>'
              f'<ul class="rval">{items}</ul></div>'
            + sec("Impresión", rep["impresion"])
            + sec("Recomendación", rep["recomendacion"])
            + '</div>')


def construir_html(original: Image.Image, overlay: Image.Image, ensemble,
                   senaladas, resumen, considerar, metodo="promedio + TTA",
                   verdad=None, recomendacion="", estado="", reporte=None) -> str:
    """Devuelve el HTML del informe como string."""
    filas = "".join(
        _fila(o, ensemble.probabilidades[o], ensemble.umbral(o),
              ensemble.nivel_confianza(o), verdad)
        for o in considerar
    )

    bloque_verdad = ""
    if verdad is not None:
        reales = sorted(registry.es(o) for o in considerar if verdad.get(o) == 1.0)
        marcadas = {o for o, _ in senaladas}
        ok = sorted(registry.es(o) for o in considerar
                    if verdad.get(o) == 1.0 and o in marcadas)
        perdidas = sorted(registry.es(o) for o in considerar
                          if verdad.get(o) == 1.0 and o not in marcadas)
        real_txt = ", ".join(reales) if reales else "sin hallazgos"
        bloque_verdad = (
            f'<div class="ref"><b>Condición real (consenso de radiólogos):</b> {real_txt}.'
            f'<br>Detectadas por el sistema: {", ".join(ok) or "—"} · '
            f'No detectadas: {", ".join(perdidas) or "—"}</div>')

    colores = {"hallazgo": ("#fffbeb", "#fde68a", "#92400e"),
               "normal": ("#f0fdf4", "#bbf7d0", "#166534"),
               "derivar": ("#eef2ff", "#c7d2fe", "#3730a3")}
    bg, bd, fg = colores.get(estado, ("#f1f5f9", "#e2e8f0", "#334155"))
    bloque_rec = ""
    if recomendacion:
        bloque_rec = (
            f'<div style="background:{bg};border:1px solid {bd};color:{fg};'
            f'border-radius:10px;padding:12px 16px;margin:14px 0;font-size:15px">'
            f'<b>Recomendación del sistema:</b> {recomendacion}</div>')

    bloque_cuerpo = _reporte_html(reporte) if reporte else (
        f'<div class="resumen">{resumen}</div>')

    return PLANTILLA.format(
        fecha=datetime.now().strftime("%d/%m/%Y %H:%M"),
        metodo=metodo,
        original=_b64(original),
        overlay=_b64(overlay),
        cuerpo=bloque_cuerpo,
        filas=filas,
        bloque_verdad=bloque_verdad,
        bloque_rec=bloque_rec,
        n_modelos=len(ensemble.individuales),
    )


PLANTILLA = """<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>Informe de análisis — RXTórax</title>
<style>
  body {{ font-family:-apple-system,Segoe UI,Roboto,sans-serif; color:#1e293b;
         background:#f8fafc; margin:0; padding:28px; font-size:14px; }}
  .hoja {{ max-width:900px; margin:0 auto; background:#fff; border:1px solid #e2e8f0;
          border-radius:12px; padding:26px 30px; }}
  h1 {{ font-size:19px; margin:0 0 3px; }}
  .meta {{ color:#64748b; font-size:12.5px; }}
  .aviso {{ background:#fff7ed; border:1px solid #fed7aa; color:#9a3412; font-size:12.5px;
           border-radius:8px; padding:9px 13px; margin:16px 0; }}
  .cols {{ display:flex; gap:18px; margin:18px 0; }}
  .cols img {{ width:100%; border-radius:8px; border:1px solid #e2e8f0; }}
  .cap {{ color:#64748b; font-size:12px; text-align:center; margin-top:4px; }}
  .resumen {{ background:#f1f5f9; border-left:4px solid #2563eb; border-radius:6px;
             padding:12px 15px; margin:14px 0; line-height:1.55; }}
  .reporte {{ border:1px solid #e2e8f0; border-radius:10px; margin:14px 0;
             padding:2px 16px 8px; }}
  .rsec {{ padding:10px 0; border-bottom:1px solid #eef2f6; }}
  .rsec:last-child {{ border-bottom:none; }}
  .rrot {{ font-size:11px; text-transform:uppercase; letter-spacing:0.05em;
          font-weight:700; color:#2563eb; margin-bottom:3px; }}
  .rval, .rval li {{ color:#1e293b; font-size:13.5px; line-height:1.5; }}
  .rval {{ margin:0; padding-left:0; }}
  ul.rval {{ padding-left:18px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
  th {{ text-align:left; font-size:12px; color:#64748b; border-bottom:1px solid #e2e8f0;
       padding:6px 4px; }}
  td {{ padding:7px 4px; border-bottom:1px solid #f1f5f9; }}
  .ref {{ background:#f0fdf4; border:1px solid #bbf7d0; color:#166534; border-radius:8px;
         padding:11px 14px; margin-top:14px; font-size:13px; line-height:1.5; }}
  .pie {{ color:#94a3b8; font-size:11.5px; margin-top:18px; }}
</style></head><body><div class="hoja">
  <h1>Análisis automatizado de radiografía de tórax</h1>
  <div class="meta">Proyecto Final de Ingeniería · Juan Bautista Felizia · {fecha}</div>
  <div class="aviso"><b>Herramienta de investigación, no de uso clínico.</b> Devuelve
    probabilidades por observación y una orientación; no emite diagnósticos. Entrenada y
    evaluada exclusivamente sobre CheXpert.</div>
  {bloque_rec}
  <div class="cols">
    <div style="flex:1"><img src="data:image/png;base64,{original}"><div class="cap">Radiografía</div></div>
    <div style="flex:1"><img src="data:image/png;base64,{overlay}"><div class="cap">Mapa de calor (Grad-CAM++)</div></div>
  </div>
  {cuerpo}
  <table>
    <tr><th>Observación</th><th style="text-align:right">Probabilidad</th><th style="text-align:center">Concordancia</th></tr>
    {filas}
  </table>
  {bloque_verdad}
  <div class="pie">Respuesta combinada de {n_modelos} configuraciones ({metodo}). La
    concordancia mide el acuerdo entre modelos: alta = respuesta firme, baja = conviene
    cautela. Las cinco patologías del benchmark de CheXpert están destacadas.</div>
</div></body></html>
"""
