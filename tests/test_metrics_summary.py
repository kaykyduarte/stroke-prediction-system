import pytest

from src.stroke_prediction.api.metrics_summary import (
    calcular_taxa_5xx, 
    criar_resumo_metricas,
    deve_alertar_5xx,
)


@pytest.mark.parametrize(
    "total, erros_5xx, esperado",
    [
        (1000, 30, 3.0),
        (100, 20, 20.0),
        (500, 0, 0.0),
        (0, 0, None),
    ],
)
def test_calcular_taxa_5xx(total, erros_5xx, esperado):
    resultado = calcular_taxa_5xx(total, erros_5xx)

    if esperado is None:
        assert resultado is None
    else:
        assert resultado == pytest.approx(esperado)


@pytest.mark.parametrize(
    "total, erros_5xx",
    [
        (-10, 2),
        (10, -2),
        (10, 15),
        (0, 1),
    ],
)
def test_calcular_taxa_5xx_rejeita_valores_invalidos(total, erros_5xx):
    with pytest.raises(ValueError):
        calcular_taxa_5xx(total, erros_5xx)


def test_criar_resumo_metricas():
    historico = {
        "total": 1000,
        "2xx": 900,
        "4xx": 70,
        "5xx": 30,
    }
    janela = {"total": 100, "5xx": 20}

    historico_original = historico.copy()
    janela_original = janela.copy()

    resultado = criar_resumo_metricas(historico, janela)

    assert resultado == {
        "historico": {
            "total": 1000,
            "erros_5xx": 30,
            "taxa_5xx": 3.0,
        },
        "ultimos_5_minutos": {
            "total": 100,
            "erros_5xx": 20,
            "taxa_5xx": 20.0,
        },
    }

    assert historico == historico_original
    assert janela == janela_original


def test_criar_resumo_metricas_com_janela_vazia():
    historico = {
        "total": 1000,
        "2xx": 900,
        "4xx": 70,
        "5xx": 30,
    }
    janela = {"total": 0, "5xx": 0}

    resultado = criar_resumo_metricas(historico, janela)

    assert resultado == {
        "historico": {
            "total": 1000,
            "erros_5xx": 30,
            "taxa_5xx": 3.0,
        },
        "ultimos_5_minutos": {
            "total": 0,
            "erros_5xx": 0,
            "taxa_5xx": None,
        },
    }


@pytest.mark.parametrize(
    "total, erros_5xx, esperado",
    [
        (1, 1, False),
        (20, 2, True),
        (100, 5, False),
        (100, 25, True),
    ],
)
def test_deve_alertar_5xx(total, erros_5xx, esperado):
    janela = {"total": total, "5xx": erros_5xx}
    original = janela.copy()

    resultado = deve_alertar_5xx(
        janela,
        limite_percentual=10.0,
        minimo_requisicoes=20,
    )

    assert resultado is esperado
    assert janela == original