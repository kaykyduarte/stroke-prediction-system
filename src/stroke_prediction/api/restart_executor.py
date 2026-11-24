import asyncio
import math
from collections.abc import Callable

from src.stroke_prediction.api.monitor_health import EstadoMonitor


async def executar_reinicio(
    atraso: float,
    estado_monitor: EstadoMonitor,
    criar_tarefa: Callable[[], asyncio.Task],
) -> asyncio.Task:
    if not math.isfinite(atraso) or atraso <= 0:
        raise ValueError("O atraso deve ser positivo e finito.")

    await asyncio.sleep(atraso)

    estado_monitor.iniciar_nova_execucao()
    return criar_tarefa()