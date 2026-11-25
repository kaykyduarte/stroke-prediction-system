import asyncio
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from src.stroke_prediction.api.monitor_health import EstadoMonitor
from src.stroke_prediction.api.restart_executor import executar_reinicio


def test_reinicio_aguarda_prepara_cria_e_retorna_tarefa():
    estado = Mock(spec=EstadoMonitor)
    tarefa = Mock(spec=asyncio.Task)
    criar_tarefa = Mock(return_value=tarefa)
    sleep_mock = AsyncMock()

    # Reúne as chamadas para verificar sua ordem.
    chamadas = Mock()
    chamadas.attach_mock(sleep_mock, "sleep")
    chamadas.attach_mock(
        estado.iniciar_nova_execucao,
        "preparar_estado",
    )
    chamadas.attach_mock(criar_tarefa, "criar_tarefa")

    async def executar():
        with patch(
            "src.stroke_prediction.api.restart_executor.asyncio.sleep",
            new=sleep_mock,
        ):
            return await executar_reinicio(
                atraso=4.0,
                estado_monitor=estado,
                criar_tarefa=criar_tarefa,
            )

    resultado = asyncio.run(executar())

    sleep_mock.assert_awaited_once_with(4.0)

    assert chamadas.mock_calls == [
        call.sleep(4.0),
        call.preparar_estado(),
        call.criar_tarefa(),
    ]
    assert resultado is tarefa


def test_cancelamento_durante_espera_nao_inicia_execucao():
    estado = Mock(spec=EstadoMonitor)
    criar_tarefa = Mock()
    sleep_mock = AsyncMock(
        side_effect=asyncio.CancelledError()
    )

    async def executar():
        with patch(
            "src.stroke_prediction.api.restart_executor.asyncio.sleep",
            new=sleep_mock,
        ):
            with pytest.raises(asyncio.CancelledError):
                await executar_reinicio(
                    atraso=4.0,
                    estado_monitor=estado,
                    criar_tarefa=criar_tarefa,
                )

    asyncio.run(executar())

    sleep_mock.assert_awaited_once_with(4.0)
    estado.iniciar_nova_execucao.assert_not_called()
    criar_tarefa.assert_not_called()


@pytest.mark.parametrize(
    "atraso",
    [0.0, -1.0, float("inf"), float("-inf"), float("nan")],
)
def test_reinicio_rejeita_atraso_invalido(atraso):
    estado = Mock(spec=EstadoMonitor)
    criar_tarefa = Mock()
    sleep_mock = AsyncMock()

    async def executar():
        with patch(
            "src.stroke_prediction.api.restart_executor.asyncio.sleep",
            new=sleep_mock,
        ):
            with pytest.raises(ValueError):
                await executar_reinicio(
                    atraso=atraso,
                    estado_monitor=estado,
                    criar_tarefa=criar_tarefa,
                )

    asyncio.run(executar())

    sleep_mock.assert_not_awaited()
    estado.iniciar_nova_execucao.assert_not_called()
    criar_tarefa.assert_not_called()


def test_cancelamento_real_durante_backoff():
    async def executar():
        estado = Mock(spec=EstadoMonitor)
        criar_tarefa = Mock()

        espera_iniciada = asyncio.Event()
        sleep_real = asyncio.sleep

        async def sleep_observado(atraso):
            espera_iniciada.set()
            await sleep_real(atraso)

        with patch(
            "src.stroke_prediction.api.restart_executor.asyncio.sleep",
            new=sleep_observado,
        ):
            tarefa = asyncio.create_task(
                executar_reinicio(
                    atraso=60.0,
                    estado_monitor=estado,
                    criar_tarefa=criar_tarefa,
                )
            )

            try:
                await asyncio.wait_for(
                    espera_iniciada.wait(),
                    timeout=2,
                )

                assert not tarefa.done()
            finally:
                tarefa.cancel()

                with pytest.raises(asyncio.CancelledError):
                    await tarefa

            assert tarefa.done()
            assert tarefa.cancelled()

            estado.iniciar_nova_execucao.assert_not_called()
            criar_tarefa.assert_not_called()

    asyncio.run(executar())