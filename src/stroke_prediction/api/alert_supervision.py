import asyncio
import logging


logger = logging.getLogger(__name__)


def supervisionar_termino(tarefa: asyncio.Task) -> None:
    """Inspeciona uma tarefa que ja terminou."""
    if tarefa.cancelled():
        return

    erro = tarefa.exception()

    if erro is not None:
        logger.error(
            "Monitor de alertas encerrado por uma exceção inesperada.",
            exc_info=(type(erro), erro, erro.__traceback__),
        )
    else:
        logger.error(
            "Monitor de alertas encerrado normalmente de forma inesperada."
        )