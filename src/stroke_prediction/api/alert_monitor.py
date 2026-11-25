import asyncio
import logging
import math

from src.stroke_prediction.api.alert_evaluation import avaliar_alerta_janela
from src.stroke_prediction.api.alert_state import ControleAlertasHisterese
from src.stroke_prediction.api.monitor_health import EstadoMonitor
from src.stroke_prediction.api.window_metrics import MetricasJanela
from src.stroke_prediction.api.recovery_policy import confirmar_recuperacao
from src.stroke_prediction.api.retry_state import ControleReinicializacoes


logger = logging.getLogger(__name__)


async def monitorar_alertas(
        metricas_janela: MetricasJanela,
        controle: ControleAlertasHisterese,
        estado_monitor: EstadoMonitor,
        intervalo_segundos: float = 30,
        controle_reinicios: ControleReinicializacoes | None = None,
) -> None:
    if not math.isfinite(intervalo_segundos) or intervalo_segundos <= 0:
        raise ValueError("O intervalo deve ser positivo e finito.")

    while True:
        await asyncio.sleep(intervalo_segundos)

        try:
            janela = metricas_janela.consultar()

            novo_incidente = avaliar_alerta_janela(
                janela=janela,
                controle=controle,
            )

            if novo_incidente:
                logger.warning(
                    "Novo incidente de respostas 5xx na janela: "
                    "total=%s erros_5xx=%s",
                    janela["total"],
                    janela["5xx"],                
                )

            estado_monitor.registrar_sucesso()

        except Exception:
            estado_monitor.registrar_falha()
            
            logger.exception(
                "Falha inesperada ao avaliar alertas; "
                "o monitor tentará novamente no próximo ciclo."                
            )
            continue

        if controle_reinicios is not None:
            try:
                confirmar_recuperacao(
                    estado_monitor=estado_monitor,
                    controle=controle_reinicios,
                )
            except Exception:
                logger.exception(
                    "Falha inesperada ao confirmar a recuperação; "
                    "a avaliação das métricas foi concluída com sucesso."
                )
