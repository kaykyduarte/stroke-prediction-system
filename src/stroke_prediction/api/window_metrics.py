from collections import deque
from threading import Lock
from time import monotonic


class MetricasJanela:
    def __init__(self, janela_segundos: int = 300):
        if type(janela_segundos) is not int or janela_segundos <= 0:
            raise ValueError("A janela deve ser um inteiro positivo.")

        self._janela_segundos = janela_segundos
        self._buckets: deque[dict[str, int]] = deque()
        self._lock = Lock()

    def _expirar(self, segundo_atual: int) -> None:
        """Remove buckets fora da janela. deve ser chamado com lock."""
        limite = segundo_atual - self._janela_segundos

        while self._buckets and self._buckets[0]["segundo"] <= limite:
            self._buckets.popleft()

    def registrar(self, status_code: int) -> None:
        with self._lock:
            segundos_atual = int(monotonic())
            self._expirar(segundos_atual)

            if (
                not self._buckets
                or self._buckets[-1]["segundo"] != segundos_atual
            ):
                self._buckets.append({
                    "segundo": segundos_atual,
                    "total": 0,
                    "5xx": 0,
                })

            bucket = self._buckets[-1]
            bucket["total"] += 1

            if 500 <= status_code < 600:
                bucket["5xx"] += 1


    def consultar(self) -> dict[str, int]:
        with self._lock:
            segundo_atual = int(monotonic())
            self._expirar(segundo_atual)

            return {
                "total": sum(
                    bucket["total"] for bucket in self._buckets
                ),
                "5xx": sum(
                    bucket["5xx"] for bucket in self._buckets
                ),
            }