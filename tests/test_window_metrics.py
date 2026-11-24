from unittest.mock import patch

import pytest

from src.stroke_prediction.api.window_metrics import MetricasJanela


def test_agrupamento_no_mesmo_segundo():
    metricas = MetricasJanela()

    with patch(
        "src.stroke_prediction.api.window_metrics.monotonic",
        return_value=1000.0,
    ):
        for status_code in [200, 500, 503]:
            metricas.registrar(status_code)

        assert metricas.consultar() == {
            "total": 3,
            "5xx": 2,
        }


def test_expiracao_sem_novas_requisicoes():
    metricas = MetricasJanela(janela_segundos=300)

    with patch(
        "src.stroke_prediction.api.window_metrics.monotonic",
        return_value=1000.0,
    ) as relogio:
        metricas.registrar(500)

        assert metricas.consultar() == {
            "total": 1,
            "5xx": 1,
        }

        relogio.return_value = 1299.0

        assert metricas.consultar() == {
            "total": 1,
            "5xx": 1,
        }

        relogio.return_value = 1300.0

        assert metricas.consultar() == {
            "total": 0,
            "5xx": 0,
        }


def test_janela_invalida():
    with pytest.raises(ValueError):
        MetricasJanela(janela_segundos=0)