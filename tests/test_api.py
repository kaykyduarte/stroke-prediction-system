from unittest.mock import Mock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from fastapi.exceptions import ResponseValidationError

from src.stroke_prediction.api.main import app
from src.stroke_prediction.api.schemas import StrokeInput


@pytest.fixture
def entrada_valida():
    return {
        "gender": "Female",
        "age": 65,
        "hypertension": 1,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 180.5,
        "bmi": None,
        "smoking_status": "formerly smoked",
    }


@pytest.fixture
def client_com_modelo():
    artefato = {
        "pipeline": Mock(),
        "limiar": 0.50,
        "versao_artefato": "1.0",
    }

    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        return_value=artefato,
    ):
        with TestClient(app) as client:
            yield client


def test_health_retorna_200():
    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        side_effect=FileNotFoundError("Arquivo não encontrado"),
    ): 
        with TestClient(app) as client:
            resposta = client.get("/health")

        assert resposta.status_code == 200
        assert resposta.json() == {"status": "ok"}


def test_ready_retorna_503_sem_artefato():
    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        side_effect=FileNotFoundError("Arquivo não encontrado"),
    ) as carregar_mock: 
        with TestClient(app) as client:
            assert app.state.artefato is None

            resposta = client.get("/ready")

            assert resposta.status_code == 503
            assert resposta.json() == {"status": "not_ready"}

            carregar_mock.assert_called_once()


def test_ready_retorna_200_com_artefato():
    artefato_simulado = {
        "pipeline": object(),
        "limiar": 0.50,
        "versao_artefato": "1.0",
    }

    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        return_value=artefato_simulado,
    ) as carregar_mock:
        with TestClient(app) as client:
            assert app.state.artefato is artefato_simulado

            resposta = client.get("/ready")

            assert resposta.status_code == 200
            assert resposta.json() == {"status": "ready"}

            client.get("/ready")
            carregar_mock.assert_called_once()


def test_predict_retorna_200(client_com_modelo, entrada_valida):
    resultado_simulado = pd.DataFrame({
        "probabilidade": [0.60],
        "classificacao": [1],
    })

    with patch(
        "src.stroke_prediction.api.main.prever",
        return_value=resultado_simulado,
    ) as prever_mock:
        resposta = client_com_modelo.post(
            "/predict",
            json=entrada_valida,
        )

        assert resposta.status_code == 200
        assert resposta.json() == {
            "score": 0.60,
            "classificacao": 1,
            "versao_artefato": "1.0",
        }

        prever_mock.assert_called_once()

        _, dados = prever_mock.call_args.args

        esperado = pd.DataFrame([
            StrokeInput.model_validate(entrada_valida).model_dump()
        ])

        pd.testing.assert_frame_equal(dados, esperado)



def test_predict_retorna_503_sem_modelo(entrada_valida):
    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        side_effect=FileNotFoundError("Modelo ausente"),
    ):
        with TestClient(app) as client:
            with patch(
                "src.stroke_prediction.api.main.prever"
            ) as prever_mock:
                resposta = client.post("/predict", json=entrada_valida)

                assert resposta.status_code == 503
                assert resposta.json() == {
                    "detail": "Modelo indisponível."
                }
                prever_mock.assert_not_called()


def test_predict_rejeita_bmi_invalido(
    client_com_modelo, entrada_valida
):
    dados = {**entrada_valida, "bmi": "valor_invalido"}

    with patch(
        "src.stroke_prediction.api.main.prever"
    ) as prever_mock:
        resposta = client_com_modelo.post("/predict", json=dados)

        assert resposta.status_code == 422
        assert any(
            erro["loc"] == ["body", "bmi"]
            for erro in resposta.json()["detail"]
        )
        prever_mock.assert_not_called()


def test_predict_rejeita_campo_obrigatorio_ausente(
    client_com_modelo, entrada_valida
):
    dados = entrada_valida.copy()
    del dados["age"]

    with patch(
        "src.stroke_prediction.api.main.prever"
    ) as prever_mock:
        resposta = client_com_modelo.post("/predict", json=dados)

        assert resposta.status_code == 422
        assert any(
            erro["loc"] == ["body", "age"]
            and erro["type"] == "missing"
            for erro in resposta.json()["detail"]
        )
        prever_mock.assert_not_called()


def test_predict_rejeita_score_invalido(
    client_com_modelo, entrada_valida
):
    resultado_invalido = pd.DataFrame({
        "probabilidade": [1.45],
        "classificacao": [1],
    })

    with patch(
        "src.stroke_prediction.api.main.prever",
        return_value=resultado_invalido,
    ):
        resposta = client_com_modelo.post(
            "/predict",
            json=entrada_valida,
        )

    assert resposta.status_code == 500
    assert resposta.json() == {
        "detail": "Erro interno na resposta do modelo."
    }
