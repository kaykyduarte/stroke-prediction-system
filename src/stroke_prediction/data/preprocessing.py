import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def dividir_dados(df, test_size=0.2, random_state=42):
    X = df.drop(columns=["id", "stroke"])
    y = df["stroke"]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def imputar_bmi(X_train, X_test):

    X_train_transformado = X_train.copy()
    X_test_transformado = X_test.copy()

    imputer = SimpleImputer(strategy="median")

    X_train_transformado["bmi"] = imputer.fit_transform(
        X_train[["bmi"]]
    ).ravel()

    X_test_transformado["bmi"] = imputer.transform(
        X_test[["bmi"]]
    ).ravel()

    return X_train_transformado, X_test_transformado


def codificar_categorias(X_train, X_test):
    categorias = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]
    numericas = X_train.columns.drop(categorias).tolist()

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False,
    )

    treino_codificado = encoder.fit_transform(X_train[categorias])
    teste_codificado = encoder.transform(X_test[categorias])

    nomes_colunas = encoder.get_feature_names_out(categorias)

    X_train_transformado = pd.concat(
        [
            X_train[numericas].copy(),
            pd.DataFrame(
                treino_codificado,
                columns=nomes_colunas,
                index=X_train.index,
            ),
        ],
        axis=1,
    )

    X_test_transformado = pd.concat(
        [
            X_test[numericas].copy(),
            pd.DataFrame(
                teste_codificado,
                columns=nomes_colunas,
                index=X_test.index,
            ),
        ],
        axis=1,
    )

    return X_train_transformado, X_test_transformado


def criar_preprocessador():
    categoricas = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]

    pipeline_bmi = Pipeline(
        steps=[
            ("imputador", SimpleImputer(strategy="median")),
            ("escalador", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("bmi", pipeline_bmi, ["bmi"]),
            (
                "numericas",
                StandardScaler(),
                ["age", "avg_glucose_level"],
            ),
            (
                "categoricas",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categoricas,
            ),
        ],
        remainder="passthrough",
    )




