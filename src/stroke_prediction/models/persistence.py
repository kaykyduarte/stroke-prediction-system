from pathlib import Path

import joblib


def salvar_modelo(pipeline, limiar, caminho):
    """Salva o pipeline treinado e seu limiar, sem realizar treinamento."""
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)

    artefato = {
        "pipeline": pipeline,
        "limiar": float(limiar),
        "versao_artefato": "1.0",
    }

    joblib.dump(artefato, caminho)

    return caminho


def carregar_modelo(caminho):
    """Carrega um artefato confiavel e verifica sua estrutura."""
    artefato = joblib.load(Path(caminho))

    if not isinstance(artefato, dict):
        raise ValueError("O artefato deve ser um dicionario.")

    chaves_obrigatorias = {"pipeline", "limiar", "versao_artefato"}
    chaves_ausentes = chaves_obrigatorias - artefato.keys()

    if chaves_ausentes:
        raise ValueError(
            f"Artefato invalido. Chaves ausentes: {sorted(chaves_ausentes)}"
        )

    return artefato