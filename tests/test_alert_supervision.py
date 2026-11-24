import asyncio
from unittest.mock import patch

import pytest

from src.stroke_prediction.api.alert_supervision import supervisionar_termino


def test_supervisao_ignora_tarefa_cancelada():
    async def executar():
        async def aguardar():
            await asyncio.Event().wait()

        tarefa = asyncio.create_task(aguardar())
        tarefa.cancel()

        with pytest.raises(asyncio.CancelledError):
            await tarefa

        with patch(
            "src.stroke_prediction.api.alert_supervision.logger.error"
        ) as error_mock:
            supervisionar_termino(tarefa)

            assert tarefa.cancelled()
            error_mock.assert_not_called()

    asyncio.run(executar())


def test_supervisao_registra_excecao():
    async def executar():
        async def falhar():
            raise RuntimeError("Falha simulada do monitor")

        tarefa = asyncio.create_task(falhar())

        with pytest.raises(RuntimeError) as erro_capturado:
            await tarefa

        with patch(
            "src.stroke_prediction.api.alert_supervision.logger.error"
        ) as error_mock:
            supervisionar_termino(tarefa)

            error_mock.assert_called_once()

            mensagem = error_mock.call_args.args[0]
            tipo, erro, traceback = error_mock.call_args.kwargs["exc_info"]

            assert "exceção inesperada" in mensagem
            assert tipo is RuntimeError
            assert erro is erro_capturado.value
            assert traceback is not None

    asyncio.run(executar())


def test_supervisao_registra_termino_normal_inesperado():
    async def executar():
        async def terminar():
            return None

        tarefa = asyncio.create_task(terminar())
        await tarefa

        with patch(
            "src.stroke_prediction.api.alert_supervision.logger.error"
        ) as error_mock:
            supervisionar_termino(tarefa)

            error_mock.assert_called_once_with(
                "Monitor de alertas encerrado normalmente de forma inesperada."
            )

    asyncio.run(executar())