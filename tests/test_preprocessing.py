import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import pytest

from src.stroke_prediction.data.preprocessing import (
    dividir_dados, 
    imputar_bmi,
    codificar_categorias,
    criar_preprocessador,
)


@pytest.fixture
def dados():
    return pd.DataFrame({
        "id": range(1, 21),
        "age": range(30, 50),
        "bmi": [25.0] * 20,
        "stroke": [0] * 15 + [1] * 5,
    })


def test_dividir_dados_proporcao(dados):
    X_train, X_test, y_train, y_test = dividir_dados(dados)

    assert len(X_train) == 16
    assert len(X_test) == 4
    assert len(y_train) == 16
    assert len(y_test) == 4


def test_dividir_dados_features(dados):
    X_train, X_test, _, _ = dividir_dados(dados)

    for features in (X_train, X_test):
        assert "id" not in features.columns
        assert "stroke" not in features.columns
        assert set(features.columns) == {"age", "bmi"}


def test_dividir_dados_estratificacao(dados):
    _, _, y_train, y_test = dividir_dados(dados)

    proporcao_original = dados["stroke"].value_counts(normalize=True)

    for conjunto in (y_train, y_test):
        proporcoes = conjunto.value_counts(normalize=True)

        for classe, proporcao in proporcao_original.items():
            assert proporcoes.get(classe, 0) == pytest.approx(
                proporcao, abs=0.05
            )


def test_imputar_bmi():
    X_train = pd.DataFrame({
        "bmi": [20.0, 25.0, 30.0, None],
        "age": [30, 40, 50, 60],
    })

    X_test = pd.DataFrame({
        "bmi": [90.0, None],
        "age": [45, 55],
    })

    treino_original = X_train.copy(deep=True)
    test_original = X_test.copy(deep=True)

    treino_tratado, test_tratado = imputar_bmi(X_train, X_test)

    treino_esperado = pd.DataFrame({
        "bmi": [20.0, 25.0, 30.0, 25.0],
        "age": [30, 40, 50, 60],
    })
    teste_esperado = pd.DataFrame({
        "bmi": [90.0, 25.0],
        "age": [45, 55],
    })

    assert_frame_equal(treino_tratado, treino_esperado)
    assert_frame_equal(test_tratado, teste_esperado)

    assert_frame_equal(X_train, treino_original)
    assert_frame_equal(X_test, test_original)


def test_codificar_categorias():
    X_train = pd.DataFrame({
        "gender": ["Male", "Female", "Female"],
        "ever_married": ["Yes", "No", "Yes"],
        "work_type": ["Private", "Self-employed", "Private"],
        "Residence_type": ["Urban", "Rural", "Urban"],
        "smoking_status": ["never smoked", "smokes", "Unknown"],
        "age": [30, 40, 50],
    }, index=[10, 20, 30])

    X_test = pd.DataFrame({
        "gender": ["Female", "Male"],
        "ever_married": ["No", "Yes"],
        # "Govt_job" não aparece no treino
        "work_type": ["Govt_job", "Private"],
        "Residence_type": ["Rural", "Urban"],
        "smoking_status": ["Unknown", "smokes"],
        "age": [45, 55],
    }, index=[40, 50])

    treino_original = X_train.copy(deep=True)
    test_original = X_test.copy(deep=True)

    treino_codificado, teste_codificado = codificar_categorias(X_train, X_test)

    categoricas = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]

    colunas_esperadas = {
        "age",
        "gender_Female",
        "gender_Male",
        "ever_married_No",
        "ever_married_Yes",
        "work_type_Private",
        "work_type_Self-employed",
        "Residence_type_Rural",
        "Residence_type_Urban",
        "smoking_status_Unknown",
        "smoking_status_never smoked",
        "smoking_status_smokes",
    }

    for resultado in (treino_codificado, teste_codificado):
        assert set(categoricas).isdisjoint(resultado.columns)
        assert set(resultado.columns) == colunas_esperadas

        valores_codificados = resultado.drop(columns="age")
        assert valores_codificados.isin([0, 1]).all().all()

    assert_frame_equal(treino_codificado[["age"]], X_train[["age"]])
    assert_frame_equal(teste_codificado[["age"]], X_test[["age"]])

    assert treino_codificado.columns.equals(teste_codificado.columns)

    assert_frame_equal(X_train, treino_original)
    assert_frame_equal(X_test, test_original)

    assert "work_type_Govt_job" not in teste_codificado.columns

    colunas_work_type = [
        "work_type_Private",
        "work_type_Self-employed",
    ]
    assert (teste_codificado.loc[40, colunas_work_type] == 0).all()

    assert teste_codificado.loc[50, "work_type_Private"] == 1


def test_integracao_preprocessamento():
    X_train = pd.DataFrame({
        "gender": ["Male", "Female", "Female", "Male"],
        "ever_married": ["Yes", "No", "Yes", "No"],
        "work_type": ["Private", "Self-employed", "Private", "Private"],
        "Residence_type": ["Urban", "Rural", "Urban", "Rural"],
        "smoking_status": ["Unknown", "smokes", "never smoked", "Unknown"],
        "age": [30, 40, 50, 60],
        "hypertension": [0, 1, 0, 1],
        "heart_disease": [0, 0, 1, 0],
        "avg_glucose_level": [80.0, 95.0, 110.0, 120.0],
        "bmi": [20.0, 25.0, 30.0, None],
    })

    X_test = pd.DataFrame({
        "gender": ["Female", "Male"],
        "ever_married": ["No", "Yes"],
        "work_type": ["Govt_job", "Private"],
        "Residence_type": ["Rural", "Urban"],
        "smoking_status": ["Unknown", "never smoked"],
        "age": [45, 55],
        "hypertension": [0, 1],
        "heart_disease": [0, 0],
        "avg_glucose_level": [100.0, 130.0],
        "bmi": [90.0, None],
    })

    preprocessador = criar_preprocessador()

    treino_transformado = preprocessador.fit_transform(X_train)
    test_transformado = preprocessador.transform(X_test)

    assert treino_transformado.shape[1] == test_transformado.shape[1]

    assert not np.isnan(treino_transformado).any()
    assert not np.isnan(test_transformado).any()

    assert treino_transformado.shape[0] == len(X_train)
    assert test_transformado.shape[0] == len(X_test)
