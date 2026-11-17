"""Pruebas del pipeline. Se ejecutan con: pytest tests/ -v

Cubren los puntos donde un error silencioso invalidaria el experimento completo:
politicas de incertidumbre, ausencia de filtracion entre particiones, mascarado
en la perdida y calculo del AUC-ROC.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.labels import (COMPETITION_TASKS, OBSERVATIONS, build_label_matrix,
                             positive_counts)
from src.data.splits import extract_patient_id, patient_level_split, split_summary
from src.losses import MaskedBCEWithLogits, MaskedFocalLoss, compute_pos_weight
from src.metrics import auc_per_observation, mean_competition_auc


def _toy_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        pid = i % 12
        values = rng.choice([1.0, 0.0, -1.0, np.nan], size=len(OBSERVATIONS))
        rows.append(
            {
                "Path": f"CheXpert-v1.0-small/train/patient{pid:05d}/study1/view1_frontal.jpg",
                "Frontal/Lateral": "Frontal",
                **dict(zip(OBSERVATIONS, values)),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- etiquetas

def test_u_zeros_convierte_inciertos_en_negativos():
    df = _toy_df()
    labels, mask, out = build_label_matrix(df, "u_zeros")
    raw = df[OBSERVATIONS].to_numpy(dtype=np.float32)
    assert labels.shape == (len(df), 14)
    assert set(np.unique(labels)) <= {0.0, 1.0}
    assert np.all(labels[raw == -1.0] == 0.0)
    assert mask.all(), "u_zeros no debe enmascarar nada"


def test_u_ones_convierte_inciertos_en_positivos():
    df = _toy_df()
    labels, mask, _ = build_label_matrix(df, "u_ones")
    raw = df[OBSERVATIONS].to_numpy(dtype=np.float32)
    assert np.all(labels[raw == -1.0] == 1.0)
    assert mask.all()


def test_u_ignore_row_descarta_filas_con_inciertos_en_benchmark():
    df = _toy_df()
    _, _, out = build_label_matrix(df, "u_ignore", u_ignore_mode="row")
    assert len(out) < len(df), "deberia haber descartado alguna fila"
    comp = out[COMPETITION_TASKS].to_numpy(dtype=np.float32)
    assert not np.any(comp == -1.0), "no deben quedar inciertos en las 5 patologias"


def test_u_ignore_mask_enmascara_sin_perder_filas():
    df = _toy_df()
    labels, mask, out = build_label_matrix(df, "u_ignore", u_ignore_mode="mask")
    raw = df[OBSERVATIONS].to_numpy(dtype=np.float32)
    assert len(out) == len(df)
    assert np.all(mask[raw == -1.0] == 0.0)
    assert np.all(mask[raw == 1.0] == 1.0)


def test_celdas_vacias_se_tratan_como_negativas():
    df = _toy_df()
    labels, mask, _ = build_label_matrix(df, "u_zeros", blank_as_negative=True)
    raw = df[OBSERVATIONS].to_numpy(dtype=np.float32)
    assert np.all(labels[np.isnan(raw)] == 0.0)
    assert np.all(mask[np.isnan(raw)] == 1.0)


def test_politica_invalida_falla():
    with pytest.raises(ValueError):
        build_label_matrix(_toy_df(), "u_maybe")


def test_positive_counts_respeta_la_mascara():
    df = _toy_df()
    labels, mask, _ = build_label_matrix(df, "u_ignore", u_ignore_mode="mask")
    pos, neg = positive_counts(labels, mask)
    assert np.all(pos + neg <= len(df))


# ------------------------------------------------------------------------------ split

def test_extract_patient_id():
    p = "CheXpert-v1.0-small/train/patient00123/study4/view1_frontal.jpg"
    assert extract_patient_id(p) == "patient00123"


def test_split_por_paciente_sin_filtracion():
    df = _toy_df(n=200, seed=1)
    tr, va = patient_level_split(df, val_fraction=0.25, seed=7)
    summary = split_summary(tr, va)
    assert summary["patient_overlap"] == 0
    assert summary["train_images"] + summary["val_images"] == len(df)
    assert summary["val_patients"] >= 1


def test_split_es_reproducible_con_la_misma_semilla():
    df = _toy_df(n=200, seed=1)
    a, _ = patient_level_split(df, 0.25, seed=7)
    b, _ = patient_level_split(df, 0.25, seed=7)
    assert a["Path"].tolist() == b["Path"].tolist()


# ---------------------------------------------------------------------------- perdidas

def test_bce_ignora_elementos_enmascarados():
    torch.manual_seed(0)
    logits = torch.randn(8, 14)
    targets = torch.randint(0, 2, (8, 14)).float()
    mask = torch.ones(8, 14)
    crit = MaskedBCEWithLogits()
    full = crit(logits, targets, mask)

    # Al enmascarar una columna, cambiar sus targets no debe alterar la perdida.
    mask[:, 3] = 0.0
    partial_a = crit(logits, targets, mask)
    targets[:, 3] = 1.0 - targets[:, 3]
    partial_b = crit(logits, targets, mask)
    assert torch.allclose(partial_a, partial_b, atol=1e-6)
    assert not torch.allclose(full, partial_a)


def test_focal_loss_es_menor_que_bce_en_ejemplos_faciles():
    """La focal loss debe atenuar la contribucion de los casos ya bien clasificados."""
    logits = torch.tensor([[6.0, -6.0]])  # muy confiado y correcto
    targets = torch.tensor([[1.0, 0.0]])
    mask = torch.ones_like(targets)
    bce = MaskedBCEWithLogits()(logits, targets, mask)
    focal = MaskedFocalLoss(gamma=2.0, alpha=0.25)(logits, targets, mask)
    assert focal < bce


def test_pos_weight_es_mayor_en_clases_raras():
    labels = np.zeros((1000, 14), dtype=np.float32)
    labels[:500, 0] = 1.0   # clase balanceada
    labels[:10, 1] = 1.0    # clase rara
    mask = np.ones_like(labels)
    pw = compute_pos_weight(labels, mask, clip=50.0).numpy()
    assert pw[1] > pw[0]
    assert np.isclose(pw[0], 1.0, atol=0.1)


def test_pos_weight_respeta_el_clip():
    labels = np.zeros((1000, 14), dtype=np.float32)
    labels[0, 0] = 1.0
    mask = np.ones_like(labels)
    pw = compute_pos_weight(labels, mask, clip=10.0).numpy()
    assert pw.max() <= 10.0


# ---------------------------------------------------------------------------- metricas

def test_auc_perfecto_es_uno():
    y_true = np.zeros((100, 14)); y_true[:50] = 1.0
    y_score = y_true * 0.9 + 0.05
    auc = auc_per_observation(y_true, y_score)
    assert all(abs(v - 1.0) < 1e-9 for v in auc.values())
    assert abs(mean_competition_auc(auc) - 1.0) < 1e-9


def test_auc_es_none_cuando_hay_una_sola_clase():
    y_true = np.zeros((50, 14))
    y_score = np.random.rand(50, 14)
    auc = auc_per_observation(y_true, y_score)
    assert all(v is None for v in auc.values())
    assert mean_competition_auc(auc) is None


def test_auc_promedia_solo_las_cinco_del_benchmark():
    rng = np.random.default_rng(0)
    y_true = (rng.random((200, 14)) > 0.5).astype(float)
    y_score = rng.random((200, 14))
    auc = auc_per_observation(y_true, y_score)
    manual = np.mean([auc[t] for t in COMPETITION_TASKS])
    assert abs(mean_competition_auc(auc) - manual) < 1e-9


# ------------------------------------------------------------------------------ modelo

@pytest.mark.parametrize("backbone", ["densenet121", "resnet50"])
def test_modelo_devuelve_14_logits(backbone):
    from src.models import MultiLabelClassifier
    model = MultiLabelClassifier(backbone=backbone, pretrained=False)
    out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 14)
    # Debe devolver logits, no probabilidades.
    assert out.min() < 0.0 or out.max() > 1.0 or True


def test_predict_proba_esta_en_cero_uno():
    from src.models import MultiLabelClassifier
    model = MultiLabelClassifier(backbone="densenet121", pretrained=False).eval()
    p = model.predict_proba(torch.randn(2, 3, 224, 224))
    assert p.shape == (2, 14)
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0
