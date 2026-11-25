import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.stroke_prediction.api.monitor_health import EstadoMonitor
from src.stroke_prediction.api.restart_recovery import (
    recuperar_tarefa_encerrada,
)
from src.stroke_prediction.api.retry_state import ControleReinicializacoes


@pytest.mark.parametrize(
    "encerrada, cancelada, limite_atingido",
    [
        (False, False, False),
        (True, True, False),
        (True, False, True),
    ],
    ids=[
        "tarefa-ativa",
        "tarefa-cancelada",
        "limite-atingido",
    ],
)
def test_recuperacao_sem_tentativa_autorizada(
    encerrada, cancelada, limite_atingido
):
    tarefa = Mock(spec=asyncio.Task)
    tarefa.done.return_value = encerrada
    tarefa.cancelled.return_value = cancelada

    controle = ControleReinicializacoes(maximo_tentativas=3)
    estado = Mock(spec=EstadoMonitor)
    criar_tarefa = Mock()

    if limite_atingido:
        for _ in range(3):
            controle.registrar_reinicio()

    with patch(
        "src.stroke_prediction.api.restart_recovery.executar_reinicio",
        new_callable=AsyncMock,
    ) as executar_mock:
        resultado = asyncio.run(
            recuperar_tarefa_encerrada(
                tarefa=tarefa,
                controle=controle,
                estado_monitor=estado,
                criar_tarefa=criar_tarefa,
            )
        )

        assert resultado is None
        executar_mock.assert_not_awaited()
        criar_tarefa.assert_not_called()
        estado.iniciar_nova_execucao.assert_not_called()


def test_recuperacao_retorna_nova_tarefa():
    tarefa = Mock(spec=asyncio.Task)
    tarefa.done.return_value = True
    tarefa.cancelled.return_value = False

    controle = ControleReinicializacoes()
    estado = Mock(spec=EstadoMonitor)
    criar_tarefa = Mock()
    nova_tarefa = Mock(spec=asyncio.Task)

    with patch(
        "src.stroke_prediction.api.restart_recovery.executar_reinicio",
        new_callable=AsyncMock,
        return_value=nova_tarefa,
    ) as executar_mock:
        resultado = asyncio.run(
            recuperar_tarefa_encerrada(
                tarefa=tarefa,
                controle=controle,
                estado_monitor=estado,
                criar_tarefa=criar_tarefa,
            )
        )

        assert resultado is nova_tarefa

        executar_mock.assert_awaited_once_with(
            atraso=1.0,
            estado_monitor=estado,
            criar_tarefa=criar_tarefa,
        )
