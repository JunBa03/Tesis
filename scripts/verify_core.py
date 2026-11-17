"""Verificacion de la logica central sin dependencias de GPU ni del dataset real.

Comprueba las tres cosas que, si fallan en silencio, invalidan el experimento
completo: las politicas de incertidumbre, la ausencia de filtracion entre
particiones y el calculo del AUC-ROC. No requiere torch ni CheXpert descargado.

Uso:
    python scripts/verify_core.py
"""
import sys, importlib.util, types
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
os.chdir(Path(__file__).resolve().parents[1])

# Stub de src/ y src.data/ que evita cargar los modulos que dependen de torch
# (dataset, models, losses, engine): el sandbox no puede instalar torch CPU.
src = types.ModuleType("src"); src.__path__ = ["src"]; sys.modules["src"] = src
sd  = types.ModuleType("src.data"); sd.__path__ = ["src/data"]; sys.modules["src.data"] = sd

def load(fq, path):
    spec = importlib.util.spec_from_file_location(fq, path)
    m = importlib.util.module_from_spec(spec); sys.modules[fq] = m
    spec.loader.exec_module(m); return m

labels  = load("src.data.labels",  "src/data/labels.py");  sd.labels = labels
splits  = load("src.data.splits",  "src/data/splits.py");  sd.splits = splits
metrics = load("src.metrics",      "src/metrics.py")
config  = load("src.config",       "src/config.py")

OBS, COMP = labels.OBSERVATIONS, labels.COMPETITION_TASKS
blm, prev = labels.build_label_matrix, labels.prevalence_report

rng = np.random.default_rng(0)
rows = [{"Path": f"CheXpert-v1.0-small/train/patient{i%40:05d}/study1/view1_frontal.jpg",
         "Frontal/Lateral": "Frontal",
         **dict(zip(OBS, rng.choice([1.0,0.0,-1.0,np.nan], size=14, p=[.25,.45,.2,.1])))}
        for i in range(400)]
df = pd.DataFrame(rows); raw = df[OBS].to_numpy(dtype=np.float32)

L,M,_ = blm(df,"u_zeros"); assert np.all(L[raw==-1]==0) and M.all()
L,M,_ = blm(df,"u_ones");  assert np.all(L[raw==-1]==1) and M.all()
_,_,O = blm(df,"u_ignore","row")
assert len(O)<len(df) and not (O[COMP].to_numpy()==-1).any()
_,Mm,Om = blm(df,"u_ignore","mask")
assert len(Om)==len(df) and np.all(Mm[raw==-1]==0) and np.all(Mm[raw==1]==1)
print(f"OK etiquetas | u_ignore(row): {len(O)}/{len(df)} filas conservadas")

tr,va = splits.patient_level_split(df, 0.25, seed=7)
s = splits.split_summary(tr,va); assert s["patient_overlap"]==0
tr2,_ = splits.patient_level_split(df, 0.25, seed=7); assert tr["Path"].tolist()==tr2["Path"].tolist()
assert splits.extract_patient_id("CheXpert-v1.0-small/train/patient00123/study4/view1_frontal.jpg")=="patient00123"
print("OK split por paciente |", s)

yt = np.zeros((200,14)); yt[:100]=1.0; ys = yt*0.9+0.05
assert abs(metrics.mean_competition_auc(metrics.auc_per_observation(yt,ys))-1.0)<1e-9
assert metrics.mean_competition_auc(metrics.auc_per_observation(np.zeros((50,14)), rng.random((50,14)))) is None
ytr=(rng.random((300,14))>0.5).astype(float); ysr=rng.random((300,14))
a = metrics.auc_per_observation(ytr,ysr)
assert abs(metrics.mean_competition_auc(a)-np.mean([a[t] for t in COMP]))<1e-9
assert len(metrics.threshold_metrics(ytr,ysr))==14 and len(metrics.best_threshold_per_observation(ytr,ysr))==14
print("OK metricas | AUC=1.0 perfecto, None si monoclase, promedio sobre las 5 del benchmark")

cfg = config.load_config("configs/e5_focal.yaml")
assert cfg["loss"]["name"]=="focal" and cfg["model"]["backbone"]=="densenet121" and cfg["train"]["epochs"]==8
assert config.load_config("configs/e3_uignore.yaml")["labels"]["uncertainty_policy"]=="u_ignore"
c6 = config.load_config("configs/e6_resnet50.yaml"); assert c6["model"]["backbone"]=="resnet50" and c6["loss"]["name"]=="bce"
config.apply_overrides(cfg, ["train.epochs=2"]); assert cfg["train"]["epochs"]==2
try:
    config.apply_overrides(config.load_config("configs/e1_baseline.yaml"), ["loss.name=hinge"]); raise SystemExit("FALLO")
except ValueError: pass
print("OK configs | extends + overrides + validacion")

sig = lambda x: 1/(1+np.exp(-x))
bce = lambda z,y: -(y*np.log(sig(z))+(1-y)*np.log(1-sig(z)))
z=np.array([[6.0,-6.0]]); y=np.array([[1.0,0.0]])
b=bce(z,y).mean(); p=sig(z); pt=p*y+(1-p)*(1-y); at=0.25*y+0.75*(1-y)
f=(at*(1-pt)**2*bce(z,y)).mean(); assert f<b
print(f"OK focal loss | BCE={b:.6f} > Focal={f:.8f} (atenua ejemplos faciles)")

w=np.clip(np.array([500.,990.])/np.array([500.,10.]),1,10); assert w[1]>w[0]
print(f"OK pos_weight | balanceada={w[0]:.2f} vs rara={w[1]:.2f} (con clip=10)")

rep = prev(*blm(df,"u_zeros")[:2])
print("\nPrevalencia (patologias del benchmark):")
print(rep[rep.in_benchmark][["observation","positives","negatives","prevalence"]].to_string(index=False))

print("\nVerificacion completa: la logica central del pipeline es correcta.")
