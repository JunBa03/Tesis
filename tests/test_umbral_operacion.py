"""El umbral de operación del sistema reemplaza al promedio de umbrales de los modelos."""
from app.core import ablation
from src.data.labels import OBSERVATIONS


class _Pred:
    def __init__(self, valor, umbral):
        self.probabilidades = {o: valor for o in OBSERVATIONS}
        self._u = umbral

    def umbral(self, obs):
        return self._u


class _Clf:
    def __init__(self, valor):
        self.valor = valor

    def predecir(self, imagen, tta=False):
        return _Pred(self.valor, 0.30)


def test_umbral_operacion_y_acuerdo():
    clfs = {f"E{i}": _Clf(0.10 + 0.02 * i) for i in range(1, 8)}   # 0,12 … 0,24
    ens = ablation.predecir_ensemble(clfs, None, umbrales_operacion={"Cardiomegaly": 0.094})
    assert ens.umbral("Cardiomegaly") == 0.094
    assert ens.acuerdo("Cardiomegaly") == 7          # todos superan el umbral de operación
    assert abs(ens.umbral("Edema") - 0.30) < 1e-9    # sin umbral propio: promedio de modelos
    assert ens.acuerdo("Edema") == 0


def test_sin_umbral_operacion_no_cambia_nada():
    clfs = {f"E{i}": _Clf(0.5) for i in range(1, 8)}
    ens = ablation.predecir_ensemble(clfs, None)
    assert abs(ens.umbral("Cardiomegaly") - 0.30) < 1e-9
