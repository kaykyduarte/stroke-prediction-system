from src.stroke_prediction.api.alert_state import ControleAlertasHisterese
from src.stroke_prediction.api.metrics_summary import calcular_taxa_5xx


def avaliar_alerta_janela(
        janela: dict[str, int],
        controle: ControleAlertasHisterese,
        minimo_requisicoes: int = 20,
) -> bool:
    if type(minimo_requisicoes) is not int or minimo_requisicoes <= 0:
        raise ValueError("O mínimo de requisições deve ser um inteiro positivo.")

    total = janela["total"]
    erros_5xx = janela["5xx"]

    taxa = calcular_taxa_5xx(total, erros_5xx)

    if total < minimo_requisicoes:
        return controle.atualizar(None)

    return controle.atualizar(taxa)