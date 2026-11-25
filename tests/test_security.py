from typing import Annotated

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.stroke_prediction.api.security import validar_chave_monitor


@pytest.fixture
def client_seguranca():
    app_teste = FastAPI()

    @app_teste.get("/protegido")
    def rota_protegida(
        _: Annotated[None, Depends(validar_chave_monitor)],
    ):
        return {"status": "autorizado"}

    with TestClient(app_teste) as client:
        yield client


@pytest.mark.parametrize(
    "chave_configurada, chave_enviada, status_esperado, json_esperado",
    [
        (
            None,
            "chave-ficticia-teste",
            503,
            {"detail": "Acesso ao monitor indisponível."},
        ),
        (
            "chave-ficticia-teste",
            None,
            403,
            {"detail": "Acesso negado."},
        ),
        (
            "chave-ficticia-teste",
            "chave-incorreta",
            403,
            {"detail": "Acesso negado."},
        ),
        (
            "chave-ficticia-teste",
            "chave-ficticia-teste",
            200,
            {"status": "autorizado"},
        ),
    ],
    ids=[
        "sem-configuracao",
        "sem-cabecalho",
        "chave-incorreta",
        "chave-correta",
    ],
)
def test_validar_chave_monitor(
    client_seguranca,
    monkeypatch,
    chave_configurada,
    chave_enviada,
    status_esperado,
    json_esperado,
):
    if chave_configurada is None:
        monkeypatch.delenv("MONITOR_API_KEY", raising=False)
    else:
        monkeypatch.setenv("MONITOR_API_KEY", chave_configurada)

    headers = {}

    if chave_enviada is not None:
        headers["X-Monitor-Key"] = chave_enviada

    resposta = client_seguranca.get(
        "/protegido",
        headers=headers,
    )

    assert resposta.status_code == status_esperado
    assert resposta.json() == json_esperado