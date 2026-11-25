import asyncio

from src.stroke_prediction.api.retry_policy import calcular_atraso
from src.stroke_prediction.api.retry_state import ControleReinicializacoes


def planejar_reinicio(
    tarefa: asyncio.Task,
    controle: ControleReinicializacoes,
) -> float | None:
    if not tarefa.done():
        return None

    if tarefa.cancelled():
        return None

    try:
        tentativa = controle.registrar_reinicio()
    except RuntimeError:
        return None

    return calcular_atraso(tentativa)
