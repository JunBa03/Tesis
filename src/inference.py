"""Inferencia sobre una radiografía.

Capa compartida por la aplicación y por el script de línea de comandos. Recibe una
imagen y devuelve las 14 probabilidades con los umbrales de decisión del modelo; no
toma decisiones clínicas.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageOps

from .data.labels import COMPETITION_TASKS, OBSERVATIONS
from .data.transforms import build_eval_transform
from .models import MultiLabelClassifier

logger = logging.getLogger("chexpert")


def preparar_entrada(imagen: Image.Image) -> Image.Image:
    """Normaliza una radiografía de cualquier origen al formato que espera el modelo.

    Pensada para imágenes **externas** al dataset (otro equipo, otra exportación, un
    DICOM convertido). Hace cuatro cosas antes de que la imagen entre al modelo:

    1. Corrige la orientación según el metadato EXIF (fotos o escaneos rotados).
    2. Lleva cualquier profundidad de bits (16 bits de un DICOM/PNG, flotante) a 8 bits.
    3. Fuerza escala de grises: una radiografía no tiene color, y un tinte sería ruido.
    4. Estira el contraste por percentiles (2--98 %) para que la distribución de
       intensidades se parezca a la de CheXpert, que es con lo que el modelo se entrenó.

    Sobre una imagen de CheXpert (ya en 8 bits, en gris y bien contrastada) el efecto
    es prácticamente nulo; sobre una externa es lo que hace que la predicción sea
    comparable. Devuelve un ``PIL.Image`` en RGB (gris replicado a tres canales).
    """
    try:
        imagen = ImageOps.exif_transpose(imagen)
    except Exception:
        pass

    # Profundidad de bits alta -> 8 bits por estiramiento min/max robusto.
    if imagen.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        arr = np.asarray(imagen).astype(np.float32)
        lo, hi = np.percentile(arr, [0.5, 99.5])
        if hi <= lo:
            lo, hi = float(arr.min()), float(arr.max())
        arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1) * 255.0
        imagen = Image.fromarray(arr.astype(np.uint8))

    gris = imagen.convert("L")

    # Estiramiento de contraste por percentiles: robusto a bordes negros y marcadores.
    arr = np.asarray(gris).astype(np.float32)
    lo, hi = np.percentile(arr, [2.0, 98.0])
    if hi - lo < 1.0:
        lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255)
        gris = Image.fromarray(arr.astype(np.uint8))

    return gris.convert("RGB")


@dataclass
class Prediccion:
    """Resultado de evaluar una imagen."""

    probabilidades: dict[str, float]
    experimento: str
    backbone: str
    incertidumbre: str
    perdida: str
    epoca: Optional[int] = None
    auc_validacion: Optional[float] = None
    umbrales: dict[str, float] = field(default_factory=dict)

    def ordenadas(self, solo_benchmark: bool = False) -> list[tuple[str, float]]:
        items = self.probabilidades.items()
        if solo_benchmark:
            items = [(k, v) for k, v in items if k in COMPETITION_TASKS]
        return sorted(items, key=lambda kv: kv[1], reverse=True)

    def umbral(self, obs: str) -> float:
        return self.umbrales.get(obs, 0.5)


class Clasificador:
    """Envuelve un checkpoint entrenado. Se construye una vez y se reutiliza."""

    def __init__(self, checkpoint: str | Path, device: Optional[str] = None,
                 umbrales: Optional[dict[str, float]] = None) -> None:
        self.ruta = Path(checkpoint)
        if not self.ruta.exists():
            raise FileNotFoundError(
                f"No existe el checkpoint {self.ruta}.\n"
                "Hay que entrenar al menos una configuración con scripts/train.py."
            )

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        ckpt = torch.load(self.ruta, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config", {})
        modelo_cfg = cfg.get("model", {})

        self.backbone = modelo_cfg.get("backbone", "densenet121")
        self.experimento = cfg.get("experiment", {}).get("name", self.ruta.stem)
        self.incertidumbre = cfg.get("labels", {}).get("uncertainty_policy", "?")
        self.perdida = cfg.get("loss", {}).get("name", "?")
        self.epoca = ckpt.get("epoch")
        self.auc_validacion = ckpt.get("best_metric")
        self.image_size = cfg.get("data", {}).get("image_size", 224)
        # Los modelos exportados para la demo traen los umbrales adentro, así no
        # dependen de la carpeta results/ ni del dataset.
        self.umbrales = umbrales or ckpt.get("umbrales", {}) or {}

        # pretrained=False: los pesos vienen del checkpoint, no de ImageNet. Bajarlos
        # sería descargar 30 MB al pedo y romper la demo si no hay internet.
        self.modelo = MultiLabelClassifier(
            backbone=self.backbone,
            pretrained=False,
            dropout=modelo_cfg.get("dropout", 0.2),
        )
        pesos = ckpt["model_state"]
        if ckpt.get("media_precision"):
            # El modelo exportado guarda los pesos en fp16 para ocupar la mitad.
            # Se vuelven a fp32 al cargar: en CPU la media precisión es más lenta,
            # y en GPU la ganancia acá es irrelevante para una imagen por vez.
            pesos = {k: (v.float() if v.is_floating_point() else v)
                     for k, v in pesos.items()}
        self.modelo.load_state_dict(pesos)
        self.modelo.to(self.device).eval()

        self.transform = build_eval_transform(self.image_size)
        logger.info("Clasificador listo | %s | %s | %s",
                    self.experimento, self.backbone, self.device)

    @torch.no_grad()
    def predecir(self, imagen: Image.Image | str | Path,
                 tta: bool = False) -> Prediccion:
        """Devuelve las 14 probabilidades para una imagen.

        Con ``tta=True`` promedia sobre pequeñas rotaciones (test-time augmentation),
        lo que estabiliza la predicción a costa de un par de pasadas más por el modelo.
        """
        if not isinstance(imagen, Image.Image):
            imagen = Image.open(imagen)
        # Normalización robusta para imágenes externas al dataset (orientación, bits,
        # gris, contraste). Sobre CheXpert es casi identidad; ver preparar_entrada.
        imagen = preparar_entrada(imagen)

        x = self.transform(imagen).unsqueeze(0).to(self.device)
        if tta:
            from .tta import probabilidades_tta
            probs = probabilidades_tta(self.modelo, x)[0].float().cpu().numpy()
        else:
            probs = torch.sigmoid(self.modelo(x))[0].float().cpu().numpy()

        return Prediccion(
            probabilidades={obs: float(p) for obs, p in zip(OBSERVATIONS, probs)},
            experimento=self.experimento,
            backbone=self.backbone,
            incertidumbre=self.incertidumbre,
            perdida=self.perdida,
            epoca=self.epoca,
            auc_validacion=self.auc_validacion,
            umbrales=self.umbrales,
        )

    @torch.no_grad()
    def calentar(self) -> None:
        """Primera pasada con una imagen negra.

        La primera inferencia siempre tarda más porque CUDA inicializa contexto y
        cuDNN elige algoritmos. En uso interactivo, eso son varios segundos de
        pantalla congelada la primera vez que alguien sube una imagen. Conviene
        pagarlos al arrancar la app.
        """
        x = torch.zeros(1, 3, self.image_size, self.image_size, device=self.device)
        self.modelo(x)
        if self.device.type == "cuda":
            torch.cuda.synchronize()


def buscar_checkpoints(directorio: str | Path = "checkpoints") -> list[Path]:
    """Checkpoints disponibles, ignorando los de estado de reanudación."""
    d = Path(directorio)
    if not d.exists():
        return []
    return sorted(p for p in d.glob("*.pt") if not p.stem.endswith("_last"))


def buscar_modelos(raiz: str | Path = ".") -> list[Path]:
    """Modelos utilizables, priorizando los exportados para la demostración.

    Si existen tanto `models/` como `checkpoints/`, se prefieren los modelos
    livianos, que traen los umbrales embebidos.
    """
    raiz = Path(raiz)
    modelos = buscar_checkpoints(raiz / "models")
    if modelos:
        return modelos
    return buscar_checkpoints(raiz / "checkpoints")


def cargar_umbrales(experimento: str, results_dir: str | Path = "results") -> dict:
    """Umbrales por observación calibrados durante el entrenamiento.

    Se calcularon maximizando F1 sobre la validación interna, nunca sobre el
    conjunto oficial. Si no están, se usa 0,5 para todo.
    """
    import json

    f = Path(results_dir) / experimento / "results.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8")).get("thresholds_from_val", {})
    except (json.JSONDecodeError, OSError):
        return {}
