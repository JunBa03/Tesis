"""Sistema visual de la demostración: tokens, estilos y piezas de layout.

Un único lugar para la identidad de la aplicación. El diseño sigue las convenciones de
las estaciones de lectura radiológica: fondo sobrio, imagen protagonista, informe
estructurado al costado, color usado con parquedad y jerarquía tipográfica clara. Si se
rehace la estética, se cambia acá y no en cada pantalla.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap del path: las páginas de Streamlit se ejecutan como scripts sueltos, así
# que hay que asegurar que la raíz del repositorio esté en sys.path para poder
# importar ``src`` y ``app``.
RAIZ = Path(__file__).resolve().parents[2]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import streamlit as st

# --- tokens de color (modo oscuro)
AZUL = "#3b82f6"
TINTA = "#f3f4f6"
TEXTO = "#c7ccd6"
GRIS = "#9aa4b2"
GRIS_CLARO = "#262b34"
AMBAR = "#f59e0b"
VERDE = "#22c55e"
ROJO = "#ef4444"

NIVEL_COLOR = {"alta": "#ef4444", "media": "#f59e0b", "baja": "#9aa4b2"}

CSS = """
<style>
  /* ---------- base ---------- */
  .block-container { padding-top: 3.4rem; padding-bottom: 3rem; max-width: 1180px; }
  #MainMenu, footer { visibility: hidden; }
  /* Indicador "Running" (el corredor) y el toolbar superior derecho. */
  [data-testid="stStatusWidget"], [data-testid="stToolbar"] { display: none !important; visibility: hidden !important; }
  /* Sidebar fijo: sin botón de colapsar ni control de reapertura. */
  [data-testid="stSidebarCollapseButton"], [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapsedControl"] { display: none !important; }
  section[data-testid="stSidebar"] { min-width: 244px !important; }
  /* Logo del sidebar más grande (st.logo). */
  [data-testid="stLogo"] { height: 2.7rem !important; width: auto !important;
      margin: 6px 4px 10px !important; }
  h1,h2,h3,h4 { color:#f3f4f6; letter-spacing:-0.01em; }
  h2 { font-weight:650; font-size:1.35rem; }
  h3 { font-weight:650; font-size:1.1rem; }
  hr { margin:1.3rem 0; border:none; border-top:1px solid #262b34; }

  /* ---------- encabezado de la aplicación ---------- */
  .topbar { display:flex; justify-content:space-between; align-items:center;
            padding:6px 2px 14px; margin:4px 0 18px; border-bottom:1px solid #262b34;
            flex-wrap:wrap; gap:8px; }
  .brand { display:flex; align-items:center; gap:13px; }
  .brand-mark { width:40px; height:40px; border-radius:10px;
                background:linear-gradient(135deg,#3b82f6,#1e40af); color:#fff;
                display:flex; align-items:center; justify-content:center;
                font-weight:800; font-size:15px; letter-spacing:0.02em;
                box-shadow:0 2px 8px rgba(37,99,235,0.35); }
  .brand-title { font-size:16px; font-weight:700; color:#f3f4f6; line-height:1.15; }
  .brand-sub { font-size:12.5px; color:#9aa4b2; }
  .pill { font-size:11.5px; font-weight:600; color:#93c5fd; background:#182233;
          border:1px solid #24354e; border-radius:20px; padding:5px 12px; }
  .page-ctx { font-size:14px; font-weight:600; color:#c7ccd6; letter-spacing:-0.01em; }

  /* ---------- aviso (slim, sin gritar) ---------- */
  .aviso { display:flex; gap:8px; align-items:center; background:#2a1e10;
           border:1px solid #6d4a15; border-radius:8px; padding:8px 13px;
           font-size:12.5px; color:#fcd9a8; margin-bottom:14px; }

  /* ---------- tarjetas de contenido ---------- */
  .bloque { border:1px solid #262b34; border-radius:12px; padding:6px 16px;
            margin-bottom:10px; background:#181c23; color:#c7ccd6; }

  /* filas de observación con barra de probabilidad */
  .obs { display:flex; justify-content:space-between; align-items:center;
         padding:9px 0; border-bottom:1px solid #23272f; }
  .obs:last-child { border-bottom:none; }
  .obs .nombre { font-size:13.5px; color:#c7ccd6; }
  .obs .bench { font-weight:650; color:#f3f4f6; }
  .obs .valor { font-variant-numeric:tabular-nums; font-weight:650; font-size:13.5px;
                color:#f3f4f6; }
  .barra { height:8px; background:#2a2f3a; border-radius:6px; overflow:hidden;
           margin-top:4px; }
  .barra > div { height:100%; border-radius:6px; }

  /* ---------- resultado principal ---------- */
  .resultado { border-radius:12px; padding:16px 18px; margin-bottom:12px;
               border:1px solid; }
  .resultado .rot { font-size:12px; text-transform:uppercase; letter-spacing:0.06em;
                    font-weight:700; opacity:0.85; }
  .resultado .dx { font-size:20px; font-weight:750; margin-top:2px; }
  .resultado .sub { font-size:13px; margin-top:5px; opacity:0.92; }
  .r-hallazgo { background:#2a1e0c; border-color:#7c5312; color:#fcd9a8; }
  .r-limpio { background:#0e2417; border-color:#1c6b3f; color:#a7f3d0; }
  .r-derivar { background:#14172e; border-color:#3b3f8f; color:#c7d2fe; }

  /* bloques de veredicto secundarios (comparación con la verdad, etc.) */
  .veredicto { border-radius:12px; padding:13px 17px; margin-bottom:12px;
               font-size:14px; border:1px solid; line-height:1.5; }
  .veredicto b { font-size:15px; }
  .v-limpio { background:#0e2417; border-color:#1c6b3f; color:#a7f3d0; }
  .v-hallazgo { background:#2a1e0c; border-color:#7c5312; color:#fcd9a8; }

  /* ---------- misc ---------- */
  .meta { color:#9aa4b2; font-size:12.5px; line-height:1.55; }

  /* ---------- landing / bienvenida ---------- */
  .landing { text-align:center; max-width:720px; margin:0 auto; }
  .hero-mark { width:92px; height:92px; border-radius:24px;
               background:linear-gradient(135deg,#3b82f6,#1d4ed8); color:#fff;
               display:flex; align-items:center; justify-content:center;
               font-weight:800; font-size:34px; letter-spacing:0.02em;
               margin:0 auto 28px; box-shadow:0 10px 30px rgba(37,99,235,0.4); }
  .welcome-tag { display:inline-block; font-size:12px; font-weight:700;
                 text-transform:uppercase; letter-spacing:0.24em; color:#8fb4f0;
                 margin-bottom:18px; }
  .hero-title { font-size:60px; font-weight:800; color:#f3f4f6; margin:0 0 18px;
                letter-spacing:-0.03em; line-height:1.0; }
  .hero-tagline { font-size:21px; color:#c7ccd6; margin:0 auto; max-width:560px;
                  line-height:1.45; }
  .hero-desc { font-size:15px; color:#8b94a3; line-height:1.6; max-width:600px;
               margin:18px auto 0; }
  .hero-hint { font-size:13px; color:#8fb4f0; margin-top:26px; }
  .landing-label { font-size:11px; font-weight:700; letter-spacing:0.2em;
                   text-transform:uppercase; color:#59616f; text-align:center;
                   margin:0 0 16px; }
  .landing-foot { text-align:center; color:#59616f; font-size:12px; margin-top:12px;
                  line-height:1.7; max-width:560px; margin-left:auto; margin-right:auto; }

  /* accesos de la landing: st.page_link como tarjeta amplia */
  [data-testid="stPageLink"] a { display:flex; flex-direction:column; justify-content:center;
      align-items:center; min-height:84px; border:1px solid #232a38; background:#161b24;
      border-radius:14px; padding:20px 16px; text-decoration:none !important;
      transition:transform .16s ease, border-color .16s ease, background .16s ease; }
  [data-testid="stPageLink"] a:hover { border-color:#3b82f6; background:#1a2130;
      transform:translateY(-3px); }
  [data-testid="stPageLink"] a p { color:#e6e9ef !important; font-weight:600 !important;
      font-size:16px !important; margin:0 !important; }
  [data-testid="stPageLink"] a [data-testid="stIconMaterial"],
  [data-testid="stPageLink"] a svg { display:none; }

  /* informe radiológico estructurado */
  .reporte { border:1px solid #262b34; border-radius:12px; background:#181c23;
             padding:6px 16px 12px; }
  .reporte .sec { padding:10px 0; border-bottom:1px solid #23272f; }
  .reporte .sec:last-child { border-bottom:none; }
  .reporte .rot { font-size:11px; text-transform:uppercase; letter-spacing:0.06em;
                  font-weight:700; color:#93c5fd; margin-bottom:3px; }
  .reporte div, .reporte li { color:#c7ccd6; font-size:13.5px; line-height:1.5; }
  .reporte ul { margin:2px 0 0; padding-left:18px; }
  .chip { display:inline-block; padding:2px 10px; border-radius:20px; font-size:11px;
          font-weight:700; color:#fff; letter-spacing:0.01em; }
  .tarjeta-modelo { border:1px solid #262b34; border-radius:12px; padding:14px 16px;
                    background:#181c23; height:100%; }
  .tarjeta-modelo h4 { margin:0 0 4px; font-size:15px; color:#f3f4f6; }
  .feature { border:1px solid #262b34; border-radius:12px; padding:16px 18px;
             background:#181c23; height:100%; }
  .feature .ic { width:34px; height:34px; border-radius:9px; background:#182233;
                 color:#93c5fd; display:flex; align-items:center; justify-content:center;
                 font-weight:800; margin-bottom:9px; }
  .feature h4 { margin:0 0 5px; font-size:15px; color:#f3f4f6; }
  .cola-item { border:1px solid #262b34; border-left-width:5px; border-radius:10px;
               padding:11px 16px; margin-bottom:8px; background:#181c23;
               display:flex; justify-content:space-between; align-items:center; }

  /* imágenes con marco suave y contenedores con borde nativo */
  [data-testid="stImage"] img { border-radius:10px; }
  [data-testid="stVerticalBlockBorderWrapper"] { border-radius:14px !important;
      border-color:#262b34 !important; box-shadow:0 1px 3px rgba(0,0,0,0.35); }
  [data-testid="stImage"] figcaption { color:#9aa4b2; font-size:12px; }
</style>
"""


def aplicar(titulo: str, sidebar: str = "expanded") -> None:
    """Configura la página y aplica el sistema visual. Llamar al inicio de cada pantalla.

    El sidebar queda fijo (expandido y sin botón de colapsar, ver CSS) y con el logo del
    producto arriba, de modo que la identidad viva ahí y no en el encabezado de cada página.
    """
    st.set_page_config(page_title=titulo, layout="wide",
                       initial_sidebar_state=sidebar)
    st.markdown(CSS, unsafe_allow_html=True)

    # Logo del producto en el sidebar (icono + nombre). Degrada con elegancia según la
    # versión de Streamlit: primero con `size`, luego sin él, y si no existe `st.logo`, nada.
    _logo = str(RAIZ / "app" / "assets" / "logo.svg")
    _icon = str(RAIZ / "app" / "assets" / "icon.svg")
    try:
        st.logo(_logo, icon_image=_icon, size="large")
    except TypeError:
        try:
            st.logo(_logo, icon_image=_icon)
        except Exception:
            pass
    except Exception:
        pass


def encabezado(subtitulo: str) -> None:
    """Barra de contexto de la página (sin logo ni nombre: eso vive en el sidebar)."""
    st.markdown(
        f'<div class="topbar"><div class="page-ctx">{subtitulo}</div>'
        f'<div class="pill">Investigación · CheXpert</div></div>',
        unsafe_allow_html=True)


def aviso() -> None:
    """Nota slim de que es una herramienta de investigación, no clínica."""
    st.markdown(
        '<div class="aviso"><b>Herramienta de investigación.</b>&nbsp;No es de uso '
        'clínico: devuelve probabilidades por observación, no diagnósticos. Entrenada y '
        'evaluada solo sobre CheXpert.</div>', unsafe_allow_html=True)


# Compatibilidad con las pantallas que aún llaman al nombre anterior.
banner_aviso = aviso
