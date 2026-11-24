from unittest.mock import patch

import pytest

from src.stroke_prediction.api.monitor_health import EstadoMonitor


def test_monitor_inicializado():
    estado = EstadoMonitor()

    assert estado.consultar() == {
        "falhas_consecutivas": 0,
        "sucessos_consecutivos": 0,
        "degradado": False,
    }


def test_monitor_degradado_apos_tres_falhas():
    estado = EstadoMonitor()

    for quantidade in range(1, 4):
        estado.registrar_falha()

        assert estado.consultar() == {
            "falhas_consecutivas": quantidade,
            "sucessos_consecutivos": 0,
            "degradado": quantidade >= 3,
        }


def test_monitor_recupera_apos_dois_sucessos():
    estado = EstadoMonitor()

    for _ in range(3):
        estado.registrar_falha()

    estado.registrar_sucesso()

    assert estado.consultar() == {
        "falhas_consecutivas": 0,
        "sucessos_consecutivos": 1,
        "degradado": True,
    }

    estado.registrar_sucesso()

    assert estado.consultar() == {
        "falhas_consecutivas": 0,
        "sucessos_consecutivos": 2,
        "degradado": False,
    }


def test_falha_interrompe_recuperacao():
    estado = EstadoMonitor()

    for _ in range(3):
        estado.registrar_falha()

    estado.registrar_sucesso()
    estado.registrar_falha()

    assert estado.consultar() == {
        "falhas_consecutivas": 1,
        "sucessos_consecutivos": 0,
        "degradado": True,
    }

    estado.registrar_sucesso()
    assert estado.consultar()["degradado"] is True

    estado.registrar_sucesso()
    assert estado.consultar()["degradado"] is False


def test_monitor_detecta_inatividade_e_registro_reinicia_relogio():
    with patch(
        "src.stroke_prediction.api.monitor_health.monotonic",
        return_value=1000.0,
    ) as relogio:
        estado = EstadoMonitor()

        relogio.return_value = 1060.0
        assert estado.esta_inativo(90) is False

        # Exatamente no limite, ainda não está inativo
        relogio.return_value = 1090.0
        assert estado.esta_inativo(90) is False

        relogio.return_value = 1091.0
        assert estado.esta_inativo(90) is True

        estado.registrar_falha()

        assert estado.esta_inativo(90) is False
        assert estado.consultar()["falhas_consecutivas"] == 1


@pytest.mark.parametrize(
    "avaliacoes, instante_consulta, esperado",
    [
        (
            [],
            1000.0,
            False,
        ),
        (
            [(1000.0, True), (1030.0, True)],
            1030.0,
            False,
        ),
        (
            [
                (1000.0, True),
                (1030.0, True),
                (1060.0, True),
                (1090.0, True),
                (1120.0, True),
            ],
            1120.0,
            True,
        ),
        (
            [
                (1000.0, True),
                (1030.0, True),
                (1060.0, True),
                (1090.0, True),
                (1120.0, True),
                (1150.0, False),
            ],
            1150.0,
            False,
        ),
        (
            [
                (1000.0, True),
                (1030.0, True),
                (1060.0, True),
                (1090.0, True),
                (1120.0, True),
            ],
            1211.0,
            False,
        ),
    ],
    ids=[
        "recem-inicializado",
        "dois-sucessos-em-30-segundos",
        "estavel-apos-120-segundos",
        "falha-interrompe-estabilidade",
        "avaliacao-atrasada",
    ],
)
def test_monitor_estabilidade(
    avaliacoes, instante_consulta, esperado
):
    with patch(
        "src.stroke_prediction.api.monitor_health.monotonic",
        return_value=1000.0,
    ) as relogio:
        estado = EstadoMonitor()

        for instante, sucesso in avaliacoes:
            relogio.return_value = instante

            if sucesso:
                estado.registrar_sucesso()
            else:
                estado.registrar_falha()

        relogio.return_value = instante_consulta

        assert estado.esta_estavel() is esperado


def test_lacuna_de_inatividade_reinicia_estabilidade():
    with patch(
        "src.stroke_prediction.api.monitor_health.monotonic",
        return_value=1000.0,
    ) as relogio:
        estado = EstadoMonitor()
        estado.registrar_sucesso()

        relogio.return_value = 1150.0
        estado.registrar_sucesso()

        assert estado.esta_estavel() is False

        assert estado.consultar()["sucessos_consecutivos"] == 1


def test_nova_execucao_nao_herda_estabilidade():
    with patch(
        "src.stroke_prediction.api.monitor_health.monotonic",
        return_value=1000.0,
    ) as relogio:
        estado = EstadoMonitor()

        # Monitor A: primeiro sucesso
        estado.registrar_sucesso()
        assert estado.esta_estavel() is False

        # Monitor A: sucessos durante 120 segundos
        for instante in [1030.0, 1060.0, 1090.0, 1120.0]:
            relogio.return_value = instante
            estado.registrar_sucesso()

        assert estado.esta_estavel() is True
        assert estado.consultar()["sucessos_consecutivos"] == 5

        # Monitor B: nova execução, sem estabilidade herdada
        relogio.return_value = 1130.0
        estado.iniciar_nova_execucao()

        assert estado.esta_estavel() is False
        assert estado.consultar()["sucessos_consecutivos"] == 0
        assert estado.esta_inativo() is False

        # Monitor B: primeiro sucesso da sua própria sequência
        relogio.return_value = 1160.0
        estado.registrar_sucesso()

        assert estado.esta_estavel() is False
        assert estado.consultar()["sucessos_consecutivos"] == 1


def test_nova_execucao_preserva_degradacao():
    with patch(
        "src.stroke_prediction.api.monitor_health.monotonic",
        return_value=1000.0,
    ):
        estado = EstadoMonitor()

        for _ in range(3):
            estado.registrar_falha()

        assert estado.consultar()["degradado"] is True

        estado.iniciar_nova_execucao()

        assert estado.consultar() == {
            "falhas_consecutivas": 3,
            "sucessos_consecutivos": 0,
            "degradado": True,
        }

        estado.registrar_sucesso()

        assert estado.consultar()["sucessos_consecutivos"] == 1
        assert estado.consultar()["degradado"] is True

        estado.registrar_sucesso()

        assert estado.consultar()["sucessos_consecutivos"] == 2
        assert estado.consultar()["degradado"] is False