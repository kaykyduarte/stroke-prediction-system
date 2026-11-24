from src.stroke_prediction.api.alert_evaluation import avaliar_alerta_janela
from src.stroke_prediction.api.alert_state import ControleAlertasHisterese


def test_avaliar_alerta_janela_transicoes():
    controle = ControleAlertasHisterese(
        limiar_ativacao=10.0,
        limiar_recuperacao=5.0,
    )

    # 1. Amostra insuficiente
    assert avaliar_alerta_janela(
        {"total": 5, "5xx": 3}, controle
    ) is False

    # 2. Novo incidente: taxa de 10%
    assert avaliar_alerta_janela(
        {"total": 20, "5xx": 2}, controle
    ) is True

    # 3. Taxa de 0%, mas volume insuficiente: preservar incidente
    assert avaliar_alerta_janela(
        {"total": 5, "5xx": 0}, controle
    ) is False

    # Confirma que o incidente continua ativo:
    # voltar a 10% não pode gerar uma nova notificação.
    assert avaliar_alerta_janela(
        {"total": 20, "5xx": 2}, controle
    ) is False

    # 4. Taxa intermediária: manter incidente
    assert avaliar_alerta_janela(
        {"total": 100, "5xx": 8}, controle
    ) is False

    # 5. Volume suficiente e taxa de 4%: recuperar
    assert avaliar_alerta_janela(
        {"total": 100, "5xx": 4}, controle
    ) is False

    # 6. Novo incidente após recuperação
    assert avaliar_alerta_janela(
        {"total": 100, "5xx": 12}, controle
    ) is True