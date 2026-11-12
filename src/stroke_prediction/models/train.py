from pathlib import Path
import warnings

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score

from src.stroke_prediction.data.preprocessing import (
    dividir_dados,
    criar_preprocessador,
)
from src.stroke_prediction.models.evaluate import (
    calcular_metricas,
    selecionar_limiar,
)

from src.stroke_prediction.models.final_evaluation import avaliar_final

from src.stroke_prediction.models.persistence import salvar_modelo


def carregar_dados():
    """Carrega o dataset original."""
    raiz = Path(__file__).resolve().parents[3]
    caminho = raiz / "data" / "raw" / "healthcare-dataset-stroke-data.csv"
    
    return pd.read_csv(caminho)


def criar_modelos():
    """cria pipeline para modelo com peso e sem peso"""
    classificadores = {
        "Baseline": DummyClassifier(strategy="most_frequent"),
        "Modelo A - sem pesos": LogisticRegression(max_iter=1000),
        "Modelo B - balanced": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
        ),
    }

    return {
        nome: Pipeline(
            steps=[
                ("preprocessador", criar_preprocessador()),
                ("classificador", classificador),
            ]
        )
        for nome, classificador in classificadores.items()
    }


def treinar_modelo(nome, modelo, X_train, y_train):
    """treina o modelo e informa avisso de convergencia."""
    with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always", ConvergenceWarning)
            modelo.fit(X_train, y_train)

    houve_aviso = False

    for aviso in avisos:
        if issubclass(aviso.category, ConvergenceWarning):
            houve_aviso = True
            print(f"\n{nome} = ConvergenceWarning: {aviso.message}")
        else:
            warnings.showwarning(
                aviso.message,
                aviso.category,
                aviso.filename,
                aviso.lineno,
            )

    if not houve_aviso:
        print(f"\n{nome}: nenhum ConvergenceWarning.")

    return modelo


def exibir_tabela(titulo, resultados):
    """exibi resultados com formatacao."""
    print(f"\n{titulo}")

    if not resultados:
        print("Nenhum resultado disponivel.")
        return

    tabela = pd.DataFrame(resultados)

    formatadores = {
        coluna: "{:.2%}".format
        for coluna in ["Accuracy", "Recall", "Precision", "Average Precision"]
        if coluna in tabela.columns
    }

    if "Limiar" in tabela.columns:
        formatadores["Limiar"] = "{:.6f}".format

    print(
        tabela.to_string(
            index=False,
            formatters=formatadores,
        )
    )


def main(avaliar_teste=False):
    df = carregar_dados()

    # -> X_restante	Features dos 80% disponíveis para treino e validação
    #   -> X_test	Features dos 20% reservados para teste
    #   -> y_restante	Valores reais de stroke dos 80% restantes
    #    -> y_test	Valores reais de stroke do teste 
    X_restante, X_test, y_restante, y_test = dividir_dados(
        df,
        test_size=0.2,
        random_state=42,
    )

    # Variáveis	Finalidade	Percentual
    #  X_train, y_train /	Aprender o pré-processamento e treinar os modelos /	60%
    #  X_validacao, y_validacao / Comparar o desempenho dos modelos / 20%
    #  X_test, y_test / Avaliação futura /	20%
    X_train, X_validacao, y_train, y_validacao = train_test_split(
        X_restante,
        y_restante,
        test_size=0.25,
        random_state=42,
        stratify=y_restante,
    )

    print(f"Treino: {len(X_train)} registros")
    print(f"Validacao: {len(X_validacao)} registros")
    print(f"Teste reservado: {len(X_test)} registros")

    modelos = criar_modelos()

    resultados_ap = []
    resultados_limiares = []
    resultados_recall_80 = []
    limiares_selecionados = {}

    for nome, modelo in modelos.items():
        treinar_modelo(nome, modelo, X_train, y_train)

        indice_classe_1 = list(modelo.classes_).index(1)
        probabilidades = modelo.predict_proba(X_validacao)[:, indice_classe_1]

        resultados_ap.append({
            "Modelo": nome,
            "Average Precision": average_precision_score(
                y_validacao,
                probabilidades,
            ),
        })

        for limiar in [0.50, 0.30, 0.20, 0.10]:
            metricas = calcular_metricas(
                y_validacao,
                probabilidades,
                limiar,
            )

            resultados_limiares.append({
                "Modelo": nome,
                **metricas,
            })
        
        # Busca um limiar com recall >= 80 no modelo de regressao
        if isinstance(
            modelo.named_steps["classificador"],
            LogisticRegression,
        ):
            limiar = selecionar_limiar(
                y_validacao,
                probabilidades,
                recall_minimo=0.80,
            )

            if limiar is None:
                print(f"{nome}: nenhum limiar elegível.")
                continue

            limiares_selecionados[nome] = limiar

            metricas = calcular_metricas(
                y_validacao,
                probabilidades,
                limiar,
            )

            resultados_recall_80.append({
                "Modelo": nome,
                **metricas,
            })

    nome_escolhido = "Modelo B - balanced"

    modelo_final = modelos[nome_escolhido]
    limiar_final = limiares_selecionados[nome_escolhido]

    if avaliar_teste:
        resultados_finais = avaliar_final(
            pipeline=modelo_final,
            X_test=X_test,
            y_test=y_test,
            limiar=limiar_final,
        )

        print("\nAvaliação final — teste reservado")
        print(f"Modelo: {nome_escolhido}")
        print(f"Limiar completo: {limiar_final!r}")
        print(f"Accuracy: {resultados_finais['Accuracy']:.2%}")
        print(f"Precision: {resultados_finais['Precision']:.2%}")
        print(f"Recall: {resultados_finais['Recall']:.2%}")
        print(
            f"Average Precision: "
            f"{resultados_finais['Average Precision']:.2%}"
        )

        print("Matriz de confusão:")
        for linha in resultados_finais["Matriz de confusao"]:
            print(linha)

    raiz = Path(__file__).resolve().parents[3]
    destino = raiz / "artifacts" / "stroke_model.joblib"

    salvar_modelo(
        pipeline=modelo_final,
        limiar=limiar_final,
        caminho=destino,
    )

    if not destino.is_file():
        raise RuntimeError(f"O artefato não foi criado: {destino}")

    print(f"Artefato salvo em: {destino}")
    print(f"Limiar salvo: {limiar_final!r}")


if __name__ == "__main__":
    main()