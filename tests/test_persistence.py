import joblib
import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.stroke_prediction.models.persistence import (
    carregar_modelo,
    salvar_modelo,
)


def test_salvar_e_carregar_modelo(tmp_path):
    X = pd.DataFrame({
        "age": [20, 30, 40, 50, 60, 70],
        "bmi": [20.0, 22.0, 24.0, 26.0, 28.0, 30.0],
    })
    y = np.array([0, 0, 0, 1, 1, 1])

    pipeline = Pipeline(
        steps=[
            ("escalador", StandardScaler()),
            ("classificador", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(X, y)

    probabilidades_antes = pipeline.predict_proba(X)

    caminho = tmp_path / "stroke_model.joblib"

    salvar_modelo(
        pipeline=pipeline,
        limiar=0.40,
        caminho=caminho,
    )

    assert caminho.is_file()

    artefato = carregar_modelo(caminho)

    assert artefato["versao_artefato"] == "1.0"
    assert artefato["limiar"] == pytest.approx(0.40)

    probabilidades_depois = artefato["pipeline"].predict_proba(X)

    np.testing.assert_allclose(
        probabilidades_depois,
        probabilidades_antes,
        rtol=1e-10,
        atol=1e-12,
    )


def test_carregar_modelo_chaves_ausentes(tmp_path):
    caminho = tmp_path / "artefato_invalido.joblib"

    joblib.dump(
        {
            "pipeline": None,
            "versao_artefato": "1.0",
        },
        caminho,
    )

    with pytest.raises(ValueError, match="limiar"):
        carregar_modelo(caminho)