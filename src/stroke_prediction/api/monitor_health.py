from threading import Lock
from time import monotonic


class EstadoMonitor:
    def __init__(self):
        self._falhas_consecutivas = 0
        self._sucessos_consecutivos = 0
        self._degradado = False
        self._ultima_avaliacao = monotonic()
        self._inicio_sucessos: float | None = None
        self._lock = Lock()

    def registrar_sucesso(self) -> None:
        with self._lock:
            agora = monotonic()
            
            if agora - self._ultima_avaliacao > 90:
                self._inicio_sucessos = None
                self._sucessos_consecutivos = 0

            self._ultima_avaliacao = agora

            if self._inicio_sucessos is None:
                self._inicio_sucessos = agora

            self._falhas_consecutivas = 0
            self._sucessos_consecutivos += 1

            if self._sucessos_consecutivos >= 2:
                self._degradado = False

    def registrar_falha(self) -> None:
        with self._lock:
            self._ultima_avaliacao = monotonic()
            self._inicio_sucessos = None
            self._sucessos_consecutivos = 0
            self._falhas_consecutivas += 1

            if self._falhas_consecutivas >= 3:
                self._degradado = True

    def esta_inativo(self, limite_segundos: float = 90) -> bool:
        with self._lock:
            tempo_decorrido = monotonic() - self._ultima_avaliacao
            return tempo_decorrido > limite_segundos

    def consultar(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "falhas_consecutivas": self._falhas_consecutivas,
                "sucessos_consecutivos": self._sucessos_consecutivos,
                "degradado": self._degradado,
            }

    def esta_estavel(
        self,
        tempo_minimo_segundos: float = 120,
        minimo_sucessos: int = 2,
    ) -> bool:
        with self._lock:
            if self._inicio_sucessos is None:
                return False

            agora = monotonic()

            sucessos_suficientes = (
                self._sucessos_consecutivos >= minimo_sucessos
            )
            periodo_suficiente = (
                agora - self._inicio_sucessos >= tempo_minimo_segundos
            )
            avaliacao_recente = (
                agora - self._ultima_avaliacao <= 90
            )

            return (
                sucessos_suficientes
                and periodo_suficiente
                and avaliacao_recente
            )

    def iniciar_nova_execucao(self) -> None:
        with self._lock:
            self._inicio_sucessos = None
            self._sucessos_consecutivos = 0
            self._ultima_avaliacao = monotonic()
