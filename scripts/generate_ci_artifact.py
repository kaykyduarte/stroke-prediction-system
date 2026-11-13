import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.stroke_prediction.data.preprocessing import criar_preprocessador
from src.stroke_prediction.models.persistence import salvar_modelo


def main():
    parser = argparse.ArgumentParser(
        description="Geraum artefato sintético para testes de integração."
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Caminho de saída. Arquivos existentes não serão substituídos."
    )
    args = parser.parse_args()

    destino = args.output.resolve()

    if destino.exists():
        parser.error(f"O arquivo de destino já existe: {destino}")

    X = pd.DataFrame({
        "gender": ["Female", "Male", "Other", "Female", "Male", "Female"],
        "age": [25, 45, 35, 65, 75, 55],
        "hypertension": [0, 0, 0, 1, 1, 0],
        "heart_disease": [0, 0, 0, 0, 1, 1],
        "ever_married": ["No", "Yes", "No", "Yes", "Yes", "Yes"],
        "work_type": [
            "Private",
            "Self-employed",
            "Govt_job",
            "Private",
            "Self-employed",
            "Govt_job",
        ],
        "Residence_type": ["Urban", "Rural", "Urban", "Rural", "Urban", "Rural"],
        "avg_glucose_level": [85.0, 100.0, 95.0, 160.0, 190.0, 140.0],
        "bmi": [22.0, None, 24.0, 28.0, 30.0, 26.0],
        "smoking_status": [
            "never smoked",
            "Unknown",
            "formerly smoked",
            "smokes",
            "formerly smoked",
            "never smoked",
        ],
    })

    y = [0, 0, 0, 1, 1, 1]

    pipeline = Pipeline(
        steps=[
            ("preprocessador", criar_preprocessador()),
            ("classificador", LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline.fit(X, y)

    limiar = 0.50

    destino.parent.mkdir(parents=True, exist_ok=True)

    try:
        with destino.open("xb"):
            pass
    except FileExistsError:
        parser.error(f"O arquivo de destino já existe: {destino}")

    salvar_modelo(
        pipeline=pipeline,
        limiar=limiar,
        caminho=destino,
    )

    print(f"Artefato sintético criado: {destino}")
    print("Uso exclusivo em testes de integração e CI. Não utilizar em produção.")


if __name__ == "__main__":
    main()