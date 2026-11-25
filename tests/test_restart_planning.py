import asyncio
from unittest.mock import Mock

from src.stroke_prediction.api.restart_planning import planejar_reinicio
from src.stroke_prediction.api.retry_state import ControleReinicializacoes


def test_tarefa_ativa_nao_planeja_reinicio():
    tarefa = Mock(spec=asyncio.Task)
    tarefa.done.return_value = False
    controle = ControleReinicializacoes()

    assert planejar_reinicio(tarefa, controle) is None

    tarefa.cancelled.assert_not_called()

    # Nenhuma tentativa foi consumida.
    assert controle.registrar_reinicio() == 1


def test_tarefa_cancelada_nao_planeja_reinicio():
    tarefa = Mock(spec=asyncio.Task)
    tarefa.done.return_value = True
    tarefa.cancelled.return_value = True
    controle = ControleReinicializacoes()

    assert planejar_reinicio(tarefa, controle) is None
    assert controle.registrar_reinicio() == 1


def test_tarefa_encerrada_planeja_atrasos_exponenciais():
    controle = ControleReinicializacoes(maximo_tentativas=3)

    tarefas = [Mock(spec=asyncio.Task) for _ in range(3)]
    for tarefa in tarefas:
        tarefa.done.return_value = True
        tarefa.cancelled.return_value = False

    atrasos = [planejar_reinicio(tarefa, controle) for tarefa in tarefas]

    assert atrasos == [1.0, 2.0, 4.0]
    assert controle.pode_reiniciar() is False


def test_limite_atingido_nao_planeja_reinicio():
    tarefa = Mock(spec=asyncio.Task)
    tarefa.done.return_value = True
    tarefa.cancelled.return_value = False
    controle = ControleReinicializacoes(maximo_tentativas=3)

    for _ in range(3):
        controle.registrar_reinicio()

    assert planejar_reinicio(tarefa, controle) is None
    assert controle.pode_reiniciar() is False
