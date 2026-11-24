from unittest.mock import Mock

from src.stroke_prediction.api.monitor_health import EstadoMonitor
from src.stroke_prediction.api.recovery_policy import confirmar_recuperacao
from src.stroke_prediction.api.retry_state import ControleReinicializacoes


def test_monitor_instavel_preserva_contador():
    estado = Mock(spec=EstadoMonitor)
    estado.esta_estavel.return_value = False

    controle = ControleReinicializacoes(maximo_tentativas=3)
    assert controle.registrar_reinicio() == 1

    resultado = confirmar_recuperacao(estado, controle)

    assert resultado is False
    estado.esta_estavel.assert_called_once_with()

    # A próxima tentativa continua sendo a segunda.
    assert controle.registrar_reinicio() == 2


def test_monitor_estavel_reinicia_contador():
    estado = Mock(spec=EstadoMonitor)
    estado.esta_estavel.return_value = True

    controle = ControleReinicializacoes(maximo_tentativas=3)

    for _ in range(3):
        controle.registrar_reinicio()

    assert controle.pode_reiniciar() is False

    resultado = confirmar_recuperacao(estado, controle)

    assert resultado is True
    estado.esta_estavel.assert_called_once_with()

    assert controle.pode_reiniciar() is True
    assert controle.registrar_reinicio() == 1