"""Carga y validacion de configuraciones YAML con herencia simple (`extends`)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

import yaml

VALID_POLICIES = {"u_zeros", "u_ones", "u_ignore"}
VALID_BACKBONES = {"densenet121", "resnet50"}
VALID_LOSSES = {"bce", "weighted_bce", "focal"}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


class Config(dict):
    """dict con acceso por punto: cfg.train.epochs."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path) -> Config:
    """Lee un YAML resolviendo `extends` de forma recursiva y valida los campos criticos."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    parent = raw.pop("extends", None)
    if parent is not None:
        base = load_config(path.parent / parent)
        raw = _deep_merge(dict(base), raw)

    cfg = Config(raw)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    policy = cfg["labels"]["uncertainty_policy"]
    if policy not in VALID_POLICIES:
        raise ValueError(f"uncertainty_policy invalida: {policy!r}. Opciones: {sorted(VALID_POLICIES)}")

    mode = cfg["labels"].get("u_ignore_mode", "row")
    if mode not in {"row", "mask"}:
        raise ValueError(f"u_ignore_mode invalido: {mode!r}. Opciones: ['row', 'mask']")

    backbone = cfg["model"]["backbone"]
    if backbone not in VALID_BACKBONES:
        raise ValueError(f"backbone invalido: {backbone!r}. Opciones: {sorted(VALID_BACKBONES)}")

    loss = cfg["loss"]["name"]
    if loss not in VALID_LOSSES:
        raise ValueError(f"loss invalida: {loss!r}. Opciones: {sorted(VALID_LOSSES)}")

    frac = cfg["data"]["internal_val_fraction"]
    if not 0.0 < frac < 0.5:
        raise ValueError(f"internal_val_fraction debe estar en (0, 0.5), recibido {frac}")


def apply_overrides(cfg: Config, overrides: list[str]) -> Config:
    """Aplica overrides del tipo 'train.epochs=2' sobre una configuracion ya cargada."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override mal formado: {item!r}. Usar clave.anidada=valor")
        key, value = item.split("=", 1)
        node: Any = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = yaml.safe_load(value)
    _validate(cfg)
    return cfg
