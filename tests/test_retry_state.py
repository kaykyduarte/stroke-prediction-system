import pytest

from src.stroke_prediction.api.retry_state import ControleReinicializacoes


def test_contador_inicial_permite_reinicio():
    controle = ControleReinicializacoes()

    assert controle.pode_reiniciar() is True


def test_limite_apos_tres_reinicializacoes():
    controle = ControleReinicializacoes(maximo_tentativas=3)

    assert controle.registrar_reinicio() == 1
    assert controle.registrar_reinicio() == 2
    assert controle.registrar_reinicio() == 3

    assert controle.pode_reiniciar() is False


def test_quarta_reinicializacao_lanca_erro():
    controle = ControleReinicializacoes(maximo_tentativas=3)

    for _ in range(3):
        controle.registrar_reinicio()

    with pytest.raises(RuntimeError):
        controle.registrar_reinicio()

    assert controle.pode_reiniciar() is False


def test_reiniciar_contagem_libera_novas_tentativas():
    controle = ControleReinicializacoes(maximo_tentativas=3)

    for _ in range(3):
        controle.registrar_reinicio()

    controle.reiniciar_contagem()

    assert controle.pode_reiniciar() is True
    assert controle.registrar_reinicio() == 1


@pytest.mark.parametrize("maximo_tentativas", [0, -1])
def test_rejeita_limite_invalido(maximo_tentativas):
    with pytest.raises(ValueError):
        ControleReinicializacoes(
            maximo_tentativas=maximo_tentativas
        )