from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.stroke_prediction.models.final_evaluation import avaliar_final


def test_avaliar_final() -> None:
    X_test = pd.DataFrame({"age": [30, 40, 50, 60]})
    y_test = np.array([0, 0, 1, 1])

    pipeline = Mock()
    pipeline.classes_ = np.array([0, 1])

    pipeline.predict_proba.return_value = np.array([
        [0.90, 0.10],
        [0.20, 0.80],
        [0.10, 0.90],
        [0.60, 0.40],
    ])

    resultados = avaliar_final(
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        limiar=0.50,
    )

    assert resultados["VN"] == 1
    assert resultados["FP"] == 1
    assert resultados["FN"] == 1
    assert resultados["VP"] == 1

    assert resultados["Accuracy"] == pytest.approx(0.50)
    assert resultados["Precision"] == pytest.approx(0.50)
    assert resultados["Recall"] == pytest.approx(0.50)

    np.testing.assert_array_almost_equal(
        resultados["Matriz de confusao"],
        [[1,1], [1, 1]]
    )

    assert resultados["Average Precision"] == pytest.approx(5 / 6)

    pipeline.predict_proba.assert_called_once_with(X_test)
    pipeline.fit.assert_not_called()