import pytest

from src.stroke_prediction.api.retry_policy import calcular_atraso


@pytest.mark.parametrize(
    "tentativa, esperado",
    [
        (1, 1.0),
        (2, 2.0),
        (3, 4.0),
        (4, 8.0),
        (5, 16.0),
    ],
)
def test_calcular_atraso_exponencial(tentativa, esperado):
    assert calcular_atraso(tentativa) == pytest.approx(esperado)


@pytest.mark.parametrize("tentativa", [4, 5, 6, 10])
def test_calcular_atraso_respeita_maximo(tentativa):
    resultado = calcular_atraso(
        tentativa,
        atraso_inicial=2.0,
        atraso_maximo=16.0,
    )

    assert resultado == pytest.approx(16.0)


@pytest.mark.parametrize("tentativa", [0, -1, -10])
def test_calcular_atraso_rejeita_tentativa_invalida(tentativa):
    with pytest.raises(ValueError):
        calcular_atraso(tentativa)


@pytest.mark.parametrize(
    "atraso_inicial, atraso_maximo",
    [
        (-1.0, 30.0),
        (1.0, 0.0),
        (30.0, 10.0),
        (float("inf"), 30.0),
        (float("nan"), 30.0),
        (1.0, float("inf")),
        (1.0, float("nan")),
    ],
)
def test_calcular_atraso_rejeita_atrasos_invalidos(
    atraso_inicial, atraso_maximo
):
    with pytest.raises(ValueError):
        calcular_atraso(
            tentativa=1,
            atraso_inicial=atraso_inicial,
            atraso_maximo=atraso_maximo,
        )