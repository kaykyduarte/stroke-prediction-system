from src.stroke_prediction.api.monitor_health import EstadoMonitor
from src.stroke_prediction.api.retry_state import ControleReinicializacoes


def confirmar_recuperacao(
    estado_monitor: EstadoMonitor,
    controle: ControleReinicializacoes,
) -> bool:
    if not estado_monitor.esta_estavel():
        return False

    controle.reiniciar_contagem()
    return True