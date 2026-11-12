from unittest.mock import Mock

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from src.stroke_prediction.models.predict import prever


def test_prever_multiplos_registros():
    dados = pd.DataFrame(
        {"age": [30, 50, 70]},
        index=[101, 205, 309],
    )

    pipeline = Mock()
    pipeline.classes_ = np.array([0, 1])
    pipeline.predict_proba.return_value = np.array([
        [0.80, 0.20],
        [0.40, 0.60],
        [0.50, 0.50],
    ])

    artefato = {
        "pipeline": pipeline,
        "limiar": 0.50,
        "versao_artefato": "1.0",
    }

    resultado = prever(artefato, dados)

    esperado = pd.DataFrame(
        {
            "probabilidade": [0.20, 0.60, 0.50],
            "classificacao": [0, 1, 1],           
        },
        index=[101, 205, 309],
    )

    assert_frame_equal(resultado, esperado)
    pipeline.predict_proba.assert_called_once_with(dados)
    pipeline.fit.assert_not_called()