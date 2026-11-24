import asyncio
from collections.abc import Callable

from src.stroke_prediction.api.monitor_health import EstadoMonitor
from src.stroke_prediction.api.restart_executor import executar_reinicio
from src.stroke_prediction.api.restart_planning import planejar_reinicio
from src.stroke_prediction.api.retry_state import ControleReinicializacoes


async def recuperar_tarefa_encerrada(
        tarefa: asyncio.Task,
        controle: ControleReinicializacoes,
        estado_monitor: EstadoMonitor,
        criar_tarefa: Callable[[], asyncio.Task],
) -> asyncio.Task | None:
    atraso = planejar_reinicio(tarefa, controle)

    if atraso is None:
        return None

    return await executar_reinicio(
        atraso=atraso,
        estado_monitor=estado_monitor,
        criar_tarefa=criar_tarefa,
    )