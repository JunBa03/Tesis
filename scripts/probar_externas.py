#!/usr/bin/env python
"""Modo de prueba rápido con radiografías externas al dataset.

Recorre una carpeta con radiografías (JPG, PNG o DICOM), corre el ensemble sobre cada
una y genera un informe HTML autocontenido con la predicción, la recomendación de triage
y (salvo que se pida lo contrario) el mapa de calor. Está pensado para ver, de un
vistazo, cómo responde el sistema a imágenes que NO vienen de CheXpert.

    python scripts/probar_externas.py --carpeta ruta\\a\\radiografias
    python scripts/probar_externas.py --carpeta imgs --salida informe.html --tta
    python scripts/probar_externas.py --carpeta imgs --sin-mapa      (más rápido)

No necesita internet. Requiere los modelos en ``models/`` (o ``checkpoints/``) igual que
la demo. El informe se abre con doble clic en cualquier navegador.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from PIL import Image

from src.inference import Clasificador, cargar_umbrales, preparar_entrada
from src.data.labels import OBSERVATIONS
from app.core import registry, ablation, derivacion, overlay, dicom
from app.core.gradcam import GradCAM

EXTS = {".jpg", ".jpeg", ".png", ".dcm", ".dicom"}
CONSIDERAR = [o for o in OBSERVATIONS if o not in ("No Finding", "Support Devices")]

ESTADO = {
    "hallazgo": ("Revisar", "#2a1e0c", "#7c5312", "#fcd9a8"),
    "normal":   ("Sin hallazgos", "#0e2417", "#1c6b3f", "#a7f3d0"),
    "derivar":  ("Derivar al radiólogo", "#14172e", "#3b3f8f", "#c7d2fe"),
}
CONC_COLOR = {"alta": "#22c55e", "media": "#f59e0b", "baja": "#ef4444"}


def cargar_imagen(ruta: Path):
    """Devuelve (PIL.Image, meta|None) o (None, None) si no se pudo leer."""
    if ruta.suffix.lower() in (".dcm", ".dicom"):
        if not dicom.disponible():
            raise RuntimeError("DICOM sin soporte: instalá pydicom")
        return dicom.leer(str(ruta))
    return Image.open(ruta), None


def b64(img: Image.Image, lado: int = 360) -> str:
    im = img.convert("RGB").copy()
    im.thumbnail((lado, lado))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def render_fila(nombre, pred, senaladas, rec, thumb, heat, meta) -> str:
    etiqueta, bg, borde, tinta = ESTADO.get(rec.estado, ESTADO["derivar"])
    badge = (f'<span style="display:inline-block;padding:4px 12px;border-radius:20px;'
             f'background:{bg};border:1px solid {borde};color:{tinta};font-size:12px;'
             f'font-weight:700">{etiqueta}</span>')

    if senaladas:
        items = []
        for o, p in senaladas[:5]:
            conc = pred.nivel_confianza(o)
            c = CONC_COLOR.get(conc, "#9aa4b2")
            items.append(
                f'<div style="display:flex;justify-content:space-between;gap:10px;'
                f'padding:5px 0;border-bottom:1px solid #23272f">'
                f'<span style="color:#c7ccd6;font-size:13px">{html.escape(registry.es(o))}</span>'
                f'<span style="color:#f3f4f6;font-size:13px;font-weight:600">{p:.2f} '
                f'<span style="color:{c};font-size:11px">· concord. {conc}</span></span></div>')
        hallazgos = "".join(items)
    else:
        hallazgos = ('<div style="color:#9aa4b2;font-size:13px;padding:6px 0">'
                     'Ninguna patología supera el umbral.</div>')

    meta_txt = ""
    if meta:
        partes = [meta.get("proyeccion"), meta.get("sexo"), meta.get("edad")]
        partes = [p for p in partes if p]
        if partes:
            meta_txt = ('<div style="color:#6b7280;font-size:11px;margin-top:4px">DICOM · '
                        + " · ".join(html.escape(str(p)) for p in partes) + "</div>")

    imgs = f'<img src="{thumb}" style="width:100%;border-radius:8px" alt="radiografía normalizada">'
    if heat:
        imgs = (f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">'
                f'<img src="{thumb}" style="width:100%;border-radius:8px" alt="normalizada">'
                f'<img src="{heat}" style="width:100%;border-radius:8px" alt="mapa de calor"></div>')

    return (
        f'<div style="border:1px solid #262b34;border-radius:12px;background:#181c23;'
        f'padding:14px 16px;margin-bottom:12px;display:grid;'
        f'grid-template-columns:300px 1fr;gap:18px;align-items:start">'
        f'<div>{imgs}{meta_txt}</div>'
        f'<div><div style="display:flex;justify-content:space-between;align-items:center;'
        f'margin-bottom:8px"><b style="color:#f3f4f6;font-size:14px">{html.escape(nombre)}</b>'
        f'{badge}</div>{hallazgos}</div></div>')


def documento(filas, conteo, total, usar_tta) -> str:
    tta_txt = " · TTA activado" if usar_tta else ""
    resumen = (f'{total} imágenes · {conteo.get("hallazgo",0)} con hallazgo · '
               f'{conteo.get("normal",0)} sin hallazgos · {conteo.get("derivar",0)} derivadas'
               f'{tta_txt}')
    return (
        '<!doctype html><html lang="es"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Prueba con radiografías externas — RXTórax</title></head>'
        '<body style="background:#0e1117;color:#c7ccd6;font-family:system-ui,Segoe UI,'
        'Roboto,Arial,sans-serif;margin:0;padding:28px 22px">'
        '<div style="max-width:1000px;margin:0 auto">'
        '<div style="display:flex;align-items:center;gap:12px;margin-bottom:6px">'
        '<div style="width:40px;height:40px;border-radius:10px;background:#2563eb;color:#fff;'
        'display:flex;align-items:center;justify-content:center;font-weight:800">RX</div>'
        '<div><div style="font-size:18px;font-weight:700;color:#f3f4f6">RXTórax · prueba con '
        'radiografías externas</div>'
        f'<div style="font-size:12.5px;color:#9aa4b2">{html.escape(resumen)}</div></div></div>'
        '<div style="background:#2a1e10;border:1px solid #6d4a15;color:#fcd9a8;font-size:12px;'
        'border-radius:8px;padding:9px 13px;margin:14px 0 20px">Herramienta de investigación '
        'sobre CheXpert. Las imágenes se normalizan automáticamente (orientación, gris, '
        'contraste, profundidad de bits). Válido para radiografías de tórax frontales; el '
        'desempeño sobre equipos ajenos a CheXpert puede ser menor por cambio de distribución.'
        '</div>'
        + "".join(filas) +
        '</div></body></html>')


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Prueba el sistema con una carpeta de radiografías externas.")
    ap.add_argument("--carpeta", required=True, help="Carpeta con radiografías (JPG/PNG/DICOM).")
    ap.add_argument("--salida", default="resultado_externas.html", help="Archivo HTML de salida.")
    ap.add_argument("--tta", action="store_true", help="Activa test-time augmentation.")
    ap.add_argument("--sin-mapa", action="store_true", dest="sin_mapa",
                    help="No genera mapa de calor (más rápido).")
    args = ap.parse_args()

    carpeta = Path(args.carpeta)
    if not carpeta.is_dir():
        print(f"No es una carpeta válida: {carpeta}")
        sys.exit(1)
    imagenes = sorted(p for p in carpeta.iterdir() if p.suffix.lower() in EXTS)
    if not imagenes:
        print(f"No hay imágenes (JPG/PNG/DICOM) en {carpeta}")
        sys.exit(1)

    modelos = registry.descubrir(RAIZ)
    if not modelos:
        print("No hay modelos entrenados en 'models/' ni 'checkpoints/'. No se puede probar.")
        sys.exit(1)

    print(f"Cargando {len(modelos)} configuraciones…")
    clasificadores = {
        eid: Clasificador(str(ruta),
                          umbrales=cargar_umbrales(Path(ruta).stem, RAIZ / "results"))
        for eid, ruta in modelos.items()}

    cfg = {}
    f_cfg = RAIZ / "results" / "ensemble_config.json"
    if f_cfg.exists():
        try:
            cfg = json.loads(f_cfg.read_text(encoding="utf-8"))
        except Exception:
            cfg = {}
    pesos = cfg.get("pesos")
    usar_tta = args.tta or bool(cfg.get("usar_tta", False))

    gcam = None
    if not args.sin_mapa:
        id_mapa = ("E7" if "E7" in clasificadores else
                   max(clasificadores, key=lambda e: registry.info(e).auc or 0))
        gcam = GradCAM(clasificadores[id_mapa])

    filas, conteo = [], {"hallazgo": 0, "normal": 0, "derivar": 0}
    for i, ruta in enumerate(imagenes, 1):
        print(f"[{i}/{len(imagenes)}] {ruta.name}")
        try:
            img, meta = cargar_imagen(ruta)
            if img is None:
                raise ValueError("no se pudo leer la imagen")
            pred = ablation.predecir_ensemble(clasificadores, img, pesos=pesos, tta=usar_tta,
                                              umbrales_operacion=cfg.get("umbrales_operacion"))
            senaladas = sorted(
                ((o, pred.probabilidades[o]) for o in CONSIDERAR
                 if pred.probabilidades[o] >= pred.umbral(o)),
                key=lambda kv: kv[1], reverse=True)
            rec = derivacion.evaluar(pred, senaladas, CONSIDERAR)
            conteo[rec.estado] = conteo.get(rec.estado, 0) + 1
            thumb = b64(preparar_entrada(img))
            heat = ""
            if gcam is not None and senaladas:
                try:
                    cam = gcam.generar(img, senaladas[0][0])
                    heat = b64(overlay.superponer(img, cam, alpha=0.5, lado=360, mascara=None))
                except Exception:
                    heat = ""
            filas.append(render_fila(ruta.name, pred, senaladas, rec, thumb, heat, meta))
        except Exception as e:  # noqa: BLE001  (un archivo malo no debe frenar el lote)
            filas.append(
                f'<div style="border:1px solid #6d2a2a;border-radius:12px;background:#241416;'
                f'color:#f0a0a0;padding:12px 16px;margin-bottom:12px">'
                f'{html.escape(ruta.name)} — no se pudo procesar ({html.escape(str(e))})</div>')

    salida = Path(args.salida)
    salida.write_text(documento(filas, conteo, len(imagenes), usar_tta), encoding="utf-8")
    print(f"\nListo. {conteo['hallazgo']} con hallazgo · {conteo['normal']} sin hallazgos · "
          f"{conteo['derivar']} derivadas.")
    print(f"Informe: {salida.resolve()}")


if __name__ == "__main__":
    main()
