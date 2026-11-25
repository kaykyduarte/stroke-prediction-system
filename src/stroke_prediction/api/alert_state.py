from threading import Lock


class ControleAlertas:
    def __init__(self):
        self._alerta_ativo = False
        self._lock = Lock()

    def atualizar(self, condicao_alerta: bool) -> bool:
        """Retorna True somente quando um novo incidente comeca."""
        with self._lock:
            novo_incidente = condicao_alerta and not self._alerta_ativo

            self._alerta_ativo = condicao_alerta

            return novo_incidente


class ControleAlertasHisterese:
    def __init__(
            self,
            limiar_ativacao: float = 10.0,
            limiar_recuperacao: float = 5.0,
    ):
        if not 0 <= limiar_recuperacao < limiar_ativacao <= 100:
            raise ValueError(
                "Os limiares devem satisfazer: "
                "0 <= recuperação < ativação <= 100."                
            )

        self._limiar_ativacao = limiar_ativacao
        self._limiar_recuperacao = limiar_recuperacao
        self._alerta_ativo = False
        self._lock = Lock()

    def atualizar(self, taxa_5xx: float | None) -> bool:
        if taxa_5xx is None:
            return False
        
        if not 0 <= taxa_5xx <= 100:
            raise ValueError("A taxa deve estar entre 0 e 100.")

        with self._lock:
            if self._alerta_ativo:
                if taxa_5xx <= self._limiar_recuperacao:
                    self._alerta_ativo = False

                return False

            if taxa_5xx >= self._limiar_ativacao:
                self._alerta_ativo = True
                return True

            return False