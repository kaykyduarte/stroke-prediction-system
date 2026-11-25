import os
from secrets import compare_digest
from typing import Annotated

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader


cabecalho_monitor = APIKeyHeader(
    name="X-Monitor-Key",
    auto_error=False,
)


def validar_chave_monitor(
        chave_recebida: Annotated[
            str | None,
            Security(cabecalho_monitor),
        ],
) -> None:
    chave_esperada = os.environ.get("MONITOR_API_KEY")

    if not chave_esperada:
        raise HTTPException(
            status_code=503,
            detail="Acesso ao monitor indisponível.",
        )

    if not chave_recebida:
        raise HTTPException(
            status_code=403,
            detail="Acesso negado.",
        )

    if not compare_digest(
        chave_recebida.encode("utf-8"),
        chave_esperada.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=403,
            detail="Acesso negado."
        )