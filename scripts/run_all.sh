#!/usr/bin/env bash
# Corre las seis configuraciones del diseno experimental de forma secuencial
# y consolida las tablas comparativas.
#
#   bash scripts/run_all.sh              # usa la ruta por defecto de configs/base.yaml
#   bash scripts/run_all.sh /otra/ruta
set -euo pipefail

DATA_ROOT="${1:-../data/raw}"
cd "$(dirname "$0")/.."

python scripts/prepare_splits.py --config configs/e1_baseline.yaml --set data.root="$DATA_ROOT"

for cfg in configs/e1_baseline.yaml configs/e2_uones.yaml configs/e3_uignore.yaml \
           configs/e4_weighted_bce.yaml configs/e5_focal.yaml configs/e6_resnet50.yaml; do
  echo "=================================================================="
  echo ">> $cfg"
  echo "=================================================================="
  # --resume permite relanzar el script entero despues de un corte sin perder trabajo:
  # las configuraciones ya completas terminan de inmediato.
  python scripts/train.py --config "$cfg" --resume --set data.root="$DATA_ROOT"
done

python scripts/compare_results.py --results-dir results
