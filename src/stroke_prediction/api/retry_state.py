from threading import Lock


class ControleReinicializacoes:
    def __init__(self, maximo_tentativas: int = 3):
        if maximo_tentativas < 1:
            raise ValueError("O máximo de tentativas deve ser pelo menos 1.")

        self._maximo_tentativas = maximo_tentativas
        self._tentativas = 0
        self._lock = Lock()

    def pode_reiniciar(self) -> bool:
        with self._lock:
            return self._tentativas < self._maximo_tentativas

    def registrar_reinicio(self) -> int:
        with self._lock:
            if self._tentativas >= self._maximo_tentativas:
                raise RuntimeError(
                    "Limite de reinicializações atingido."
                )

            self._tentativas += 1
            return self._tentativas

    def reiniciar_contagem(self) -> None:
        with self._lock:
            self._tentativas = 0