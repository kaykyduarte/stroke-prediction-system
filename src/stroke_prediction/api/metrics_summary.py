def calcular_taxa_5xx(total: int, erros_5xx: int) -> float | None:
    """Retorna o percentual de respostas 5xx ou None quando nao ha dados."""
    if total < 0 or erros_5xx < 0:
        raise ValueError("Os contadores não podem ser negativos.")

    if erros_5xx > total:
        raise ValueError("A quantidade de erros não pode superar o total.")
    
    if total == 0:
        return None

    return (erros_5xx / total) * 100


def criar_resumo_metricas(historico: dict, janela: dict) -> dict:
    return {
        "historico": {
            "total": historico["total"],
            "erros_5xx": historico["5xx"],
            "taxa_5xx": calcular_taxa_5xx(
                historico["total"],
                historico["5xx"],
            ),
        },
        "ultimos_5_minutos": {
            "total": janela["total"],
            "erros_5xx": janela["5xx"],
            "taxa_5xx": calcular_taxa_5xx(
                janela["total"],
                janela["5xx"],
            ),    
        },
    }


def deve_alertar_5xx(
        janela: dict,
        limite_percentual: float = 10.0,
        minimo_requisicoes: int = 20,
) -> bool:
    total = janela["total"]
    taxa = calcular_taxa_5xx(total, janela["5xx"])

    return (
        total >= minimo_requisicoes
        and taxa is not None
        and taxa >= limite_percentual
    )