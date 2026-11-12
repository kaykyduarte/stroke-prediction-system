import pandas as pd


def prever(artefato, dados):
    """Gera probabilidades e classes usando pipeline ja treinado."""
    pipeline = artefato["pipeline"]
    limiar = artefato["limiar"]

    indice_classe_1 = list(pipeline.classes_).index(1)

    probabilidades = pipeline.predict_proba(
        dados
    )[:, indice_classe_1]

    classificacoes = (probabilidades >= limiar).astype(int)

    return pd.DataFrame(
        {
            "probabilidade": probabilidades,
            "classificacao": classificacoes,
        },
        index=dados.index,
    )