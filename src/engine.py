"""Bucle de entrenamiento y evaluacion: AMP, early stopping y checkpointing."""
from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import mean_competition_auc, summarize
from .utils import AverageMeter

logger = logging.getLogger("chexpert")


@dataclass
class EarlyStopping:
    """Detiene el entrenamiento cuando la metrica de validacion deja de mejorar.

    Es la forma mas simple de regularizacion implicita: no modifica la arquitectura
    ni la perdida, solo corta el proceso antes de que el sobreajuste se consolide.
    """

    patience: int = 3
    min_delta: float = 1e-4
    best: float = -np.inf
    counter: int = 0
    should_stop: bool = False

    def step(self, value: float) -> bool:
        """Devuelve True si el valor recibido es el mejor hasta el momento."""
        if value > self.best + self.min_delta:
            self.best = value
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


@dataclass
class TrainState:
    epoch: int = 0
    best_metric: float = -np.inf
    best_epoch: int = -1
    history: list = field(default_factory=list)


def build_optimizer(cfg, model: nn.Module) -> torch.optim.Optimizer:
    name = cfg["train"]["optimizer"].lower()
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(
            params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
        )
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=cfg["train"]["lr"], momentum=0.9,
            weight_decay=cfg["train"]["weight_decay"],
        )
    raise ValueError(f"Optimizador desconocido: {name!r}")


def _build_scaler():
    """GradScaler compatible con las dos APIs.

    torch.cuda.amp.GradScaler quedo deprecado a favor de torch.amp.GradScaler("cuda").
    """
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):  # pragma: no cover
        return torch.cuda.amp.GradScaler()


def build_scheduler(cfg, optimizer):
    sched = cfg["train"]["scheduler"]
    if sched == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            factor=cfg["train"]["scheduler_factor"],
            patience=cfg["train"]["scheduler_patience"],
        )
    if sched == "cosine":
        # Recocido de coseno a lo largo de todas las épocas: baja la tasa de aprendizaje
        # de forma suave hasta casi cero, lo que suele exprimir un poco más de AUC en un
        # presupuesto fijo de épocas.
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg["train"]["epochs"], eta_min=cfg["train"]["lr"] * 0.02,
        )
    return None


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler=None,
    grad_clip: Optional[float] = None,
    max_batches: Optional[int] = None,
    epoch: int = 0,
    channels_last: bool = False,
) -> float:
    model.train()
    meter = AverageMeter()
    use_amp = scaler is not None and device.type == "cuda"
    mem_fmt = torch.channels_last if channels_last else torch.contiguous_format

    pbar = tqdm(loader, desc=f"epoca {epoch} [train]", leave=False)
    for i, (images, targets, mask) in enumerate(pbar):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device, non_blocking=True, memory_format=mem_fmt)
        targets = targets.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, targets, mask)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        meter.update(loss.item(), images.size(0))
        pbar.set_postfix(loss=f"{meter.avg:.4f}")

    return meter.avg


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    max_batches: Optional[int] = None,
    desc: str = "eval",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Devuelve (y_true, y_score, mask, loss_promedio)."""
    model.eval()
    meter = AverageMeter()
    ys, ps, ms = [], [], []

    for i, (images, targets, mask) in enumerate(tqdm(loader, desc=desc, leave=False)):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets_d = targets.to(device, non_blocking=True)
        mask_d = mask.to(device, non_blocking=True)

        logits = model(images)
        if criterion is not None:
            meter.update(criterion(logits, targets_d, mask_d).item(), images.size(0))

        ps.append(torch.sigmoid(logits).float().cpu().numpy())
        ys.append(targets.numpy())
        ms.append(mask.numpy())

    if not ys:
        raise RuntimeError("El loader no produjo ningun batch.")
    return np.concatenate(ys), np.concatenate(ps), np.concatenate(ms), meter.avg


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
    max_batches: Optional[int] = None,
    desc: str = "eval",
) -> Dict:
    y_true, y_score, mask, loss = predict(
        model, loader, device, criterion, max_batches, desc
    )
    result = summarize(y_true, y_score, mask)
    result["loss"] = loss
    result["n_samples"] = int(y_true.shape[0])
    return result, y_true, y_score, mask


def save_checkpoint(path: Path, model: nn.Module, optimizer, state: TrainState, cfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": state.epoch,
            "best_metric": state.best_metric,
            "best_epoch": state.best_epoch,
            "history": state.history,
            "config": dict(cfg),
        },
        path,
    )


def save_progress(path: Path, cfg, state: TrainState, status: str,
                  total_epochs: int) -> None:
    """Escribe un JSON chico con el avance de la corrida.

    El panel de experimentos lo lee para mostrar una configuracion mientras
    todavia esta entrenando. Es informativo: si se pierde, no afecta nada.
    """
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": cfg["experiment"]["name"],
        "status": status,
        "epoch": state.epoch,
        "total_epochs": total_epochs,
        "best_epoch": state.best_epoch,
        "best_val_auc_competition_mean": (
            None if state.best_metric == -np.inf else state.best_metric
        ),
        "history": state.history,
        "backbone": cfg["model"]["backbone"],
        "uncertainty_policy": cfg["labels"]["uncertainty_policy"],
        "loss": cfg["loss"]["name"],
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def save_resume_state(path: Path, model: nn.Module, optimizer, scheduler, scaler,
                      state: TrainState, stopper: EarlyStopping, cfg) -> None:
    """Guarda todo lo necesario para retomar exactamente donde se corto.

    Se escribe primero a un archivo temporal y despues se renombra: si el proceso
    muere en mitad del guardado, el estado anterior queda intacto en vez de
    corromperse. En una corrida de varios dias esto importa.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "epoch": state.epoch,
        "best_metric": state.best_metric,
        "best_epoch": state.best_epoch,
        "history": state.history,
        "stopper_counter": stopper.counter,
        "stopper_should_stop": stopper.should_stop,
        "config": dict(cfg),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, model: nn.Module, optimizer=None,
                    device: Optional[torch.device] = None) -> dict:
    ckpt = torch.load(path, map_location=device or "cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def fit(
    cfg,
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    criterion: nn.Module,
    device: torch.device,
    checkpoint_path: Path,
    resume: bool = False,
    progress_path: Optional[Path] = None,
) -> TrainState:
    """Entrena y selecciona el mejor modelo por AUC-ROC promedio sobre la validacion interna.

    El conjunto oficial de CheXpert no participa de ninguna decision de
    entrenamiento; se evalua una unica vez, al final, con el mejor checkpoint.
    """
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"
    scaler = _build_scaler() if use_amp else None
    stopper = EarlyStopping(patience=cfg["train"]["early_stopping_patience"])
    state = TrainState()

    # Estado de reanudacion: se reescribe DESPUES de cada epoca, mejore o no.
    # Distinto del checkpoint "mejor modelo", que solo se toca cuando el AUC mejora.
    # Con seis configuraciones corriendo varios dias, un corte de luz sin esto
    # significa perder la corrida entera.
    resume_path = checkpoint_path.with_name(checkpoint_path.stem + "_last.pt")
    start_epoch = 1
    best_weights = None

    if resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if scaler is not None and ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        if scheduler is not None and ckpt.get("scheduler_state"):
            scheduler.load_state_dict(ckpt["scheduler_state"])
        state.history = ckpt.get("history", [])
        state.best_metric = ckpt.get("best_metric", -np.inf)
        state.best_epoch = ckpt.get("best_epoch", -1)
        stopper.best = state.best_metric
        stopper.counter = ckpt.get("stopper_counter", 0)
        # Recuperar si el early stopping ya se habia disparado. Sin esto, una
        # corrida que ya habia cortado por convergencia se reactivaba al reanudar
        # y corria epocas de mas ignorando el early stopping.
        stopper.should_stop = (
            ckpt.get("stopper_should_stop", False)
            or stopper.counter >= stopper.patience
        )
        start_epoch = ckpt["epoch"] + 1
        logger.info(
            "Reanudando desde la epoca %d (mejor AUC hasta ahora: %.4f en la epoca %d)",
            start_epoch, state.best_metric, state.best_epoch,
        )

        # El mejor modelo real esta en el checkpoint separado, no en los pesos de
        # la ultima epoca. Se carga desde ahi para que, si la corrida ya termino o
        # las epocas restantes no mejoran, el modelo final sea el mejor de verdad.
        if checkpoint_path.exists():
            mejor = torch.load(checkpoint_path, map_location=device, weights_only=False)
            best_weights = copy.deepcopy(mejor["model_state"])
        else:
            best_weights = copy.deepcopy(model.state_dict())

        if stopper.should_stop or start_epoch > cfg["train"]["epochs"]:
            logger.info(
                "La corrida ya estaba completa (early stopping en la epoca %d, "
                "mejor AUC %.4f). No se reentrena.",
                state.best_epoch, state.best_metric,
            )
            model.load_state_dict(best_weights)
            if progress_path is not None:
                save_progress(progress_path, cfg, state, status="completa",
                              total_epochs=cfg["train"]["epochs"])
            return state
    elif resume:
        logger.info("No hay estado previo en %s: se empieza de cero.", resume_path)

    if best_weights is None:
        best_weights = copy.deepcopy(model.state_dict())
    if progress_path is not None:
        save_progress(progress_path, cfg, state, status="corriendo",
                      total_epochs=cfg["train"]["epochs"])

    for epoch in range(start_epoch, cfg["train"]["epochs"] + 1):
        state.epoch = epoch
        t0 = time.time()

        train_loss = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler,
            grad_clip=cfg["train"]["grad_clip"],
            max_batches=cfg["train"]["max_train_batches"],
            epoch=epoch,
            channels_last=bool(cfg["train"].get("channels_last", False))
                          and device.type == "cuda",
        )

        val_res, *_ = evaluate(
            model, loaders["val_internal"], device, criterion,
            max_batches=cfg["train"]["max_eval_batches"],
            desc=f"epoca {epoch} [val]",
        )
        val_auc = val_res["auc_competition_mean"] or 0.0

        if scheduler is not None:
            # ReduceLROnPlateau ajusta según la métrica; el resto avanza por época.
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()

        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_res["loss"],
            "val_auc_competition_mean": val_auc,
            "val_auc_all_mean": val_res["auc_all_mean"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(time.time() - t0, 1),
        }
        state.history.append(entry)
        logger.info(
            "epoca %d | train_loss=%.4f | val_loss=%.4f | val_AUC(5)=%.4f | lr=%.2e | %.1fs",
            epoch, train_loss, val_res["loss"], val_auc,
            entry["lr"], entry["seconds"],
        )

        if stopper.step(val_auc):
            state.best_metric = val_auc
            state.best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            save_checkpoint(checkpoint_path, model, optimizer, state, cfg)
            logger.info("  -> nuevo mejor modelo (AUC=%.4f), checkpoint guardado", val_auc)

        # Estado de reanudacion, se escribe siempre.
        save_resume_state(resume_path, model, optimizer, scheduler, scaler, state,
                          stopper, cfg)
        # Progreso liviano para el panel: permite mirar una corrida en curso sin
        # esperar a que termine ni abrir el checkpoint.
        if progress_path is not None:
            save_progress(progress_path, cfg, state, status="corriendo",
                          total_epochs=cfg["train"]["epochs"])

        if stopper.should_stop:
            logger.info(
                "Early stopping en la epoca %d (sin mejora durante %d epocas)",
                epoch, stopper.patience,
            )
            break

    model.load_state_dict(best_weights)
    if progress_path is not None:
        save_progress(progress_path, cfg, state, status="completa",
                      total_epochs=cfg["train"]["epochs"])
    return state
