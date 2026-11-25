import asyncio
import logging
from unittest.mock import AsyncMock, Mock, patch, call

import pytest

from src.stroke_prediction.api.alert_monitor import monitorar_alertas
from src.stroke_prediction.api.alert_state import ControleAlertasHisterese
from src.stroke_prediction.api.monitor_health import EstadoMonitor
from src.stroke_prediction.api.retry_state import ControleReinicializacoes


def test_monitor_registra_apenas_um_novo_incidente():
    metricas = Mock()
    metricas.consultar.return_value = {"total": 100, "5xx": 20}

    controle = ControleAlertasHisterese()

    async def executar():
        sleep_mock = AsyncMock(
            side_effect=[None, None, asyncio.CancelledError()]
        )

        with patch(
            "src.stroke_prediction.api.alert_monitor.asyncio.sleep",
            new=sleep_mock,
        ):
            with patch(
                "src.stroke_prediction.api.alert_monitor.logger.warning"
            ) as warning_mock:
                with pytest.raises(asyncio.CancelledError):
                    await monitorar_alertas(
                        metricas_janela=metricas,
                        controle=controle,
                        estado_monitor=EstadoMonitor(),
                    )

                assert metricas.consultar.call_count == 2
                warning_mock.assert_called_once()

    asyncio.run(executar())


def test_monitor_permite_cancelamento():
    async def executar():
        metricas = Mock()
        controle = ControleAlertasHisterese()

        entrou_na_espera = asyncio.Event()
        liberar_espera = asyncio.Event()

        async def sleep_controlado(intervalo):
            entrou_na_espera.set()
            await liberar_espera.wait()

        with patch(
            "src.stroke_prediction.api.alert_monitor.asyncio.sleep",
            new=sleep_controlado,
        ):
            tarefa = asyncio.create_task(
                monitorar_alertas(
                    metricas_janela=metricas,
                    controle=controle,
                    estado_monitor=EstadoMonitor(),
                )
            )

            try:
                # Confirma que o monitor iniciou e está aguardando
                await asyncio.wait_for(
                    entrou_na_espera.wait(),
                    timeout=2,
                )
            finally:
                tarefa.cancel()

                with pytest.raises(asyncio.CancelledError):
                    await tarefa

            assert tarefa.done()
            assert tarefa.cancelled()

            # O cancelamento ocorreu antes da primeira avaliação
            metricas.consultar.assert_not_called()

    asyncio.run(executar())


def test_monitor_continua_apos_falha():
    metricas = Mock()
    metricas.consultar.side_effect = [
        KeyError("total"),
        {"total": 100, "5xx": 20},
        asyncio.CancelledError(),
    ]

    controle = ControleAlertasHisterese()

    async def executar():
        with patch(
            "src.stroke_prediction.api.alert_monitor.asyncio.sleep",
            new=AsyncMock(),
        ):
            with patch(
                "src.stroke_prediction.api.alert_monitor.logger"
            ) as logger_mock:
                with pytest.raises(asyncio.CancelledError):
                    await monitorar_alertas(
                        metricas_janela=metricas,
                        controle=controle,
                        estado_monitor=EstadoMonitor(),
                    )

                assert metricas.consultar.call_count == 3
                logger_mock.exception.assert_called_once()
                logger_mock.warning.assert_called_once()

    asyncio.run(executar())


def test_monitor_registra_falhas_e_recuperacao():
    metricas = Mock()
    metricas.consultar.side_effect = [
        KeyError("total"),
        KeyError("total"),
        KeyError("total"),
        {"total": 100, "5xx": 0},
        {"total": 100, "5xx": 0},
    ]

    controle = ControleAlertasHisterese()
    estado_monitor = EstadoMonitor()
    estados_observados = []

    async def sleep_controlado(intervalo):
        # Cada nova espera ocorre após concluir a avaliação anterior.
        if metricas.consultar.call_count > 0:
            estados_observados.append(estado_monitor.consultar())

        # Encerra depois das cinco avaliações.
        if len(estados_observados) == 5:
            raise asyncio.CancelledError()

    async def executar():
        with patch(
            "src.stroke_prediction.api.alert_monitor.asyncio.sleep",
            new=sleep_controlado,
        ):
            with patch(
                "src.stroke_prediction.api.alert_monitor.logger"
            ) as logger_mock:
                with pytest.raises(asyncio.CancelledError):
                    await monitorar_alertas(
                        metricas_janela=metricas,
                        controle=controle,
                        estado_monitor=estado_monitor,
                    )

                assert metricas.consultar.call_count == 5
                assert logger_mock.exception.call_count == 3

    asyncio.run(executar())

    assert estados_observados == [
        {"falhas_consecutivas": 1, "sucessos_consecutivos": 0, "degradado": False},
        {"falhas_consecutivas": 2, "sucessos_consecutivos": 0, "degradado": False},
        {"falhas_consecutivas": 3, "sucessos_consecutivos": 0, "degradado": True},
        {"falhas_consecutivas": 0, "sucessos_consecutivos": 1, "degradado": True},
        {"falhas_consecutivas": 0, "sucessos_consecutivos": 2, "degradado": False},
    ]


def test_monitor_consulta_recuperacao_apos_registrar_sucesso():
    metricas = Mock()
    metricas.consultar.return_value = {"total": 100, "5xx": 0}

    controle = ControleAlertasHisterese()
    estado = Mock(spec=EstadoMonitor)
    reinicios = ControleReinicializacoes()

    async def executar():
        sleep_mock = AsyncMock(
            side_effect=[None, asyncio.CancelledError()]
        )

        with (
            patch(
                "src.stroke_prediction.api.alert_monitor.asyncio.sleep",
                new=sleep_mock,
            ),
            patch(
                "src.stroke_prediction.api.alert_monitor.confirmar_recuperacao",
                return_value=False,
            ) as confirmar_mock,
        ):
            ordem = Mock()
            ordem.attach_mock(estado.registrar_sucesso, "sucesso")
            ordem.attach_mock(confirmar_mock, "confirmar")

            with pytest.raises(asyncio.CancelledError):
                await monitorar_alertas(
                    metricas_janela=metricas,
                    controle=controle,
                    estado_monitor=estado,
                    controle_reinicios=reinicios,
                )

            assert ordem.mock_calls == [
                call.sucesso(),
                call.confirmar(
                    estado_monitor=estado,
                    controle=reinicios,
                ),
            ]
            estado.registrar_falha.assert_not_called()

    asyncio.run(executar())


def test_monitor_nao_consulta_recuperacao_apos_falha():
    metricas = Mock()
    metricas.consultar.side_effect = KeyError("total")

    controle = ControleAlertasHisterese()
    estado = Mock(spec=EstadoMonitor)
    reinicios = ControleReinicializacoes()

    async def executar():
        sleep_mock = AsyncMock(
            side_effect=[None, asyncio.CancelledError()]
        )

        with (
            patch(
                "src.stroke_prediction.api.alert_monitor.asyncio.sleep",
                new=sleep_mock,
            ),
            patch(
                "src.stroke_prediction.api.alert_monitor.confirmar_recuperacao"
            ) as confirmar_mock,
            patch(
                "src.stroke_prediction.api.alert_monitor.logger.exception"
            ),
        ):
            with pytest.raises(asyncio.CancelledError):
                await monitorar_alertas(
                    metricas_janela=metricas,
                    controle=controle,
                    estado_monitor=estado,
                    controle_reinicios=reinicios,
                )

            estado.registrar_falha.assert_called_once_with()
            estado.registrar_sucesso.assert_not_called()
            confirmar_mock.assert_not_called()

    asyncio.run(executar())


def test_falha_na_confirmacao_nao_invalida_avaliacao(caplog):
    metricas = Mock()
    metricas.consultar.return_value = {"total": 100, "5xx": 0}
    controle = ControleAlertasHisterese()
    estado = Mock(spec=EstadoMonitor)
    reinicios = ControleReinicializacoes()
    falha = RuntimeError("Falha simulada na confirmação da recuperação")

    async def executar():
        with (
            patch(
                "src.stroke_prediction.api.alert_monitor.asyncio.sleep",
                new=AsyncMock(
                    side_effect=[None, None, asyncio.CancelledError()]
                ),
            ),
            patch(
                "src.stroke_prediction.api.alert_monitor.confirmar_recuperacao",
                side_effect=[falha, False],
            ) as confirmar_mock,
        ):
            with pytest.raises(asyncio.CancelledError):
                await monitorar_alertas(
                    metricas_janela=metricas,
                    controle=controle,
                    estado_monitor=estado,
                    controle_reinicios=reinicios,
                )

            assert metricas.consultar.call_count == 2
            assert estado.registrar_sucesso.call_count == 2
            estado.registrar_falha.assert_not_called()
            assert confirmar_mock.call_count == 2

    with caplog.at_level(
        logging.ERROR,
        logger="src.stroke_prediction.api.alert_monitor",
    ):
        asyncio.run(executar())

    erros = [
        registro
        for registro in caplog.records
        if registro.name == "src.stroke_prediction.api.alert_monitor"
        and registro.levelno == logging.ERROR
    ]
    assert len(erros) == 1
    assert "confirmar a recuperação" in erros[0].getMessage()
    assert erros[0].exc_info[1] is falha
    assert erros[0].exc_info[2] is not None


def _executar_avaliacoes_simuladas(instantes, respostas):
    metricas = Mock()
    metricas.consultar.side_effect = respostas

    controle_alertas = ControleAlertasHisterese()
    reinicios = ControleReinicializacoes(maximo_tentativas=3)

    for _ in range(3):
        reinicios.registrar_reinicio()

    disponibilidade = []
    estados = []

    with patch(
        "src.stroke_prediction.api.monitor_health.monotonic",
        return_value=instantes[0],
    ) as relogio:
        estado = EstadoMonitor()
        indice = 0

        async def sleep_controlado(intervalo):
            nonlocal indice

            # A nova espera indica que a avaliação anterior terminou.
            if indice > 0:
                disponibilidade.append(reinicios.pode_reiniciar())
                estados.append(estado.consultar())

            if indice == len(instantes):
                raise asyncio.CancelledError()

            relogio.return_value = instantes[indice]
            indice += 1

        async def executar():
            with patch(
                "src.stroke_prediction.api.alert_monitor.asyncio.sleep",
                new=sleep_controlado,
            ):
                with pytest.raises(asyncio.CancelledError):
                    await monitorar_alertas(
                        metricas_janela=metricas,
                        controle=controle_alertas,
                        estado_monitor=estado,
                        controle_reinicios=reinicios,
                    )

        asyncio.run(executar())

    assert metricas.consultar.call_count == len(instantes)

    return disponibilidade, estados


def test_monitor_libera_reinicios_apos_120_segundos():
    disponibilidade, estados = _executar_avaliacoes_simuladas(
        instantes=[1000.0, 1060.0, 1120.0],
        respostas=[
            {"total": 100, "5xx": 0},
            {"total": 100, "5xx": 0},
            {"total": 100, "5xx": 0},
        ],
    )

    assert disponibilidade == [False, False, True]
    assert estados[-1]["sucessos_consecutivos"] == 3


def test_monitor_nao_libera_reinicios_apos_lacuna():
    disponibilidade, estados = _executar_avaliacoes_simuladas(
        instantes=[1000.0, 1060.0, 1170.0],
        respostas=[
            {"total": 100, "5xx": 0},
            {"total": 100, "5xx": 0},
            {"total": 100, "5xx": 0},
        ],
    )

    assert disponibilidade == [False, False, False]

    # Houve três avaliações bem-sucedidas, mas apenas uma
    # pertence à sequência atual, iniciada após a lacuna.
    assert estados[-1]["sucessos_consecutivos"] == 1


def test_monitor_falha_interrompe_periodo_de_estabilidade():
    disponibilidade, estados = _executar_avaliacoes_simuladas(
        instantes=[
            1000.0,  # Sucesso
            1060.0,  # Falha
            1120.0,  # Primeiro sucesso da nova sequência
            1180.0,  # 60 segundos da nova sequência
            1240.0,  # 120 segundos da nova sequência
        ],
        respostas=[
            {"total": 100, "5xx": 0},
            RuntimeError("Falha simulada na avaliação"),
            {"total": 100, "5xx": 0},
            {"total": 100, "5xx": 0},
            {"total": 100, "5xx": 0},
        ],
    )

    assert disponibilidade == [False, False, False, False, True]

    assert [
        estado["sucessos_consecutivos"]
        for estado in estados
    ] == [1, 0, 1, 2, 3]

    assert estados[1]["falhas_consecutivas"] == 1
