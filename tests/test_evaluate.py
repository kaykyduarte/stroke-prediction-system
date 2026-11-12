import numpy as np
import pytest

from src.stroke_prediction.models.evaluate import (
    calcular_metricas,
    selecionar_limiar,
)


def test_calcular_metricas() -> None:
    y_real = np.array([0, 0, 0, 1, 1, 1])
    probabilidades = np.array([0.10, 0.20, 0.80, 0.90, 0.70, 0.30])

    resultado = calcular_metricas(
        y_real,
        probabilidades,
        limiar=0.50,
    )

    assert resultado["VN"] == 2
    assert resultado["FP"] == 1
    assert resultado["FN"] == 1
    assert resultado["VP"] == 2

    assert resultado["Precision"] == pytest.approx(2 / 3)
    assert resultado["Recall"] == pytest.approx(2 / 3)
    assert resultado["Accuracy"] == pytest.approx(4 / 6)


def test_selecionar_limiar() -> None:
    y_real = np.array([1, 0, 1, 0, 1])
    probabilidades = np.array([0.90, 0.80, 0.70, 0.60, 0.50])

    limiar = selecionar_limiar(
        y_real,
        probabilidades,
        recall_minimo=0.60,
    )

    assert limiar == pytest.approx(0.70)

    metricas = calcular_metricas(y_real, probabilidades, limiar)

    assert metricas["Recall"] >= 0.60
    assert metricas["Recall"] == pytest.approx(2 / 3)
    assert metricas["Precision"] == pytest.approx(2 / 3)


def test_selecionar_limiar_sem_positivos():
    y_real = np.array([0, 0, 0, 0])
    probabilidades = np.array([0.10, 0.30, 0.60, 0.90])

    resultado = selecionar_limiar(
        y_real,
        probabilidades,
        recall_minimo=0.80,
    )

    assert resultado is None