from collections import Counter
from threading import Lock


class MetricasAPI:
    def __init__(self):
        self._contadores = Counter({
            "total": 0,
            "2xx": 0,
            "4xx": 0,
            "5xx": 0
        })
        self._lock = Lock()

    def registrar(self, status_code: int) -> None:
        """Registra uma resposta HTTP sem armazenar dados da requisicao."""
        with self._lock:
            self._contadores["total"] += 1

            if 200 <= status_code < 300:
                self._contadores["2xx"] += 1
            elif 400 <= status_code < 500:
                self._contadores["4xx"] += 1
            elif 500 <= status_code < 600:
                self._contadores["5xx"] += 1

    def consultar(self) -> dict[str, int]:
        """Retorna uma copia do contadores atuais."""
        with self._lock:
            return dict(self._contadores)