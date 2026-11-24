import pytest

from src.stroke_prediction.api.alert_state import ControleAlertas, ControleAlertasHisterese


def test_controle_alertas_transicoes():
    controle = ControleAlertas()

    assert controle.atualizar(False) is False

    assert controle.atualizar(True) is True

    assert controle.atualizar(True) is False
    assert controle.atualizar(True) is False

    assert controle.atualizar(False) is False

    assert controle.atualizar(True) is True


def test_controle_alertas_oscilacao():
    controle = ControleAlertas()
    condicoes = [True, False, True, False, True]

    resultados = [
        controle.atualizar(condicao)
        for condicao in condicoes
    ]

    assert resultados == [True, False, True, False, True]
    assert resultados.count(True) == 3


def test_histerese_gera_duas_notificacoes():
    controle = ControleAlertasHisterese()
    taxas = [12.0, 8.0, 11.0, 4.0, 12.0]

    resultados = [
        controle.atualizar(taxa)
        for taxa in taxas
    ]

    assert resultados == [True, False, False, False, True]
    assert resultados.count(True) == 2


def test_histerese_sem_dados_preserva_estado():
    controle = ControleAlertasHisterese()

    assert controle.atualizar(None) is False
    assert controle.atualizar(8.0) is False
    assert controle.atualizar(12.0) is True

    assert controle.atualizar(None) is False
    assert controle.atualizar(8.0) is False
    assert controle.atualizar(12.0) is False

    assert controle.atualizar(4.0) is False
    assert controle.atualizar(12.0) is True


def test_histerese_ativa_no_limiar_exato():
    controle = ControleAlertasHisterese(
        limiar_ativacao=10.0,
        limiar_recuperacao=5.0,
    )

    assert controle.atualizar(10.0) is True
    assert controle.atualizar(10.0) is False


def test_histerese_recupera_no_limiar_exato():
    controle = ControleAlertasHisterese(
        limiar_ativacao=10.0,
        limiar_recuperacao=5.0,
    )

    assert controle.atualizar(12.0) is True
    assert controle.atualizar(5.0) is False

    # Uma nova notificação comprova que houve recuperação
    assert controle.atualizar(10.0) is True


def test_histerese_rejeita_limites_invalidos():
    with pytest.raises(ValueError):
        ControleAlertasHisterese(
            limiar_ativacao=5.0,
            limiar_recuperacao=10.0,
        )