"""Capa de datos.

`labels` y `splits` solo dependen de numpy/pandas, de modo que pueden importarse
sin torch instalado (util para scripts de preparacion y verificacion). Los
simbolos de `dataset`, que si requieren torch, se resuelven de forma diferida.
"""
from .io import read_chexpert_csv
from .labels import (COMPETITION_TASKS, OBSERVATIONS, REFERENCE_AUC,
                     build_label_matrix, positive_counts, prevalence_report)
from .splits import (add_patient_column, extract_patient_id, filter_frontal,
                     patient_level_split, split_summary)

_LAZY = {
    "CheXpertDataset": "dataset",
    "build_datasets": "dataset",
    "build_dataloaders": "dataset",
    "build_train_transform": "transforms",
    "build_eval_transform": "transforms",
}


def __getattr__(name: str):
    if name in _LAZY:
        import importlib
        module = importlib.import_module(f".{_LAZY[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OBSERVATIONS", "COMPETITION_TASKS", "REFERENCE_AUC",
    "build_label_matrix", "prevalence_report", "positive_counts",
    "patient_level_split", "filter_frontal", "split_summary",
    "extract_patient_id", "add_patient_column",
    "CheXpertDataset", "build_datasets", "build_dataloaders", "read_chexpert_csv",
    "build_train_transform", "build_eval_transform",
]
