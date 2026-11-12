import pytest
from pydantic import ValidationError

from src.stroke_prediction.api.schemas import StrokeInput, StrokeOutput


@pytest.fixture
def registro_valido():
    return {
        "gender": "Female",
        "age": 50,
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 95.0,
        "bmi": None,
        "smoking_status": "Unknown",
    }


def test_stroke_input_aceita_bmi_nulo(registro_valido):
    entrada = StrokeInput.model_validate(registro_valido)

    assert entrada.bmi is None


def test_stroke_input_rejeita_bmi_invalido(registro_valido):
    dados = {**registro_valido, "bmi": "valor_invalido"}

    with pytest.raises(ValidationError) as erro:
        StrokeInput.model_validate(dados)

    assert any(
        detalhe["loc"] == ("bmi",)
        for detalhe in erro.value.errors()
    )


@pytest.mark.parametrize(
    "campo, valor_invalido",
    [
        ("age", -1),
        ("avg_glucose_level", -10),
        ("bmi", -5),
    ],
)
def test_stroke_input_rejeita_valores_negativos(
    registro_valido, campo, valor_invalido
):
    dados = {**registro_valido, campo: valor_invalido}

    with pytest.raises(ValidationError) as erro:
        StrokeInput.model_validate(dados)

    assert any(
        detalhe["loc"] == (campo,)
        for detalhe in erro.value.errors()
    )


def test_stroke_input_aceita_idade_limite(registro_valido):
    dados = {**registro_valido, "age": 125}

    entrada = StrokeInput.model_validate(dados)

    assert entrada.age == 125


def test_stroke_input_rejeita_idade_acima_limite(registro_valido):
    dados = {**registro_valido, "age": 150}

    with pytest.raises(ValidationError) as erro:
        StrokeInput.model_validate(dados)

    assert any(
        detalhe["loc"] == ("age",)
        for detalhe in erro.value.errors()
    )


def test_stroke_input_rejeita_categoria_invalida(registro_valido):
    dados = {
        **registro_valido,
        "work_type": "categoria_invalida",
    }

    with pytest.raises(ValidationError) as erro:
        StrokeInput.model_validate(dados)

    assert any(
        detalhe["loc"] == ("work_type",)
        and detalhe["type"] == "literal_error"
        for detalhe in erro.value.errors()
    )


def test_stroke_input_aceita_smoking_status_unknown(registro_valido):
    dados = {
        **registro_valido,
        "smoking_status": "Unknown",
    }

    entrada = StrokeInput.model_validate(dados)

    assert entrada.smoking_status == "Unknown"


def test_stroke_output_aceita_resposta_valida():
    resposta = StrokeOutput.model_validate({
        "score": 0.75,
        "classificacao": 1,
        "versao_artefato": "1.0",
    })

    assert resposta.score == 0.75
    assert resposta.classificacao == 1
    assert resposta.versao_artefato == "1.0"


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_stroke_output_rejeita_score_fora_intervalo(score):
    with pytest.raises(ValidationError) as erro:
        StrokeOutput.model_validate({
            "score": score,
            "classificacao": 1,
            "versao_artefato": "1.0",
        })

    assert any(
        detalhe["loc"] == ("score",)
        for detalhe in erro.value.errors()
    )