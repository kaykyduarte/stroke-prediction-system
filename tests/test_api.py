from unittest.mock import AsyncMock, Mock, patch
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Barrier

import asyncio
import logging
import re
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from types import SimpleNamespace

from src.stroke_prediction.api.main import app
from src.stroke_prediction.api.schemas import StrokeInput
from src.stroke_prediction.api.monitor_health import EstadoMonitor
from src.stroke_prediction.api.retry_state import ControleReinicializacoes
from src.stroke_prediction.api.main import criar_tarefa_monitor
from src.stroke_prediction.api.main import agendar_recuperacao
from src.stroke_prediction.api.main import tratar_termino_monitor
from src.stroke_prediction.api.restart_recovery import recuperar_tarefa_encerrada
from src.stroke_prediction.api.alert_monitor import (
    monitorar_alertas as monitorar_alertas_real,
)
from src.stroke_prediction.api.recovery_policy import (
    confirmar_recuperacao as confirmar_recuperacao_real,
)


@pytest.fixture
def entrada_valida():
    return {
        "gender": "Female",
        "age": 65,
        "hypertension": 1,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 180.5,
        "bmi": None,
        "smoking_status": "formerly smoked",
    }


@pytest.fixture
def client_com_modelo():
    artefato = {
        "pipeline": Mock(),
        "limiar": 0.50,
        "versao_artefato": "1.0",
    }

    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        return_value=artefato,
    ):
        with TestClient(app) as client:
            yield client


def test_health_retorna_200():
    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        side_effect=FileNotFoundError("Arquivo não encontrado"),
    ): 
        with TestClient(app) as client:
            resposta = client.get("/health")

        assert resposta.status_code == 200
        assert resposta.json() == {"status": "ok"}


def test_ready_retorna_503_sem_artefato():
    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        side_effect=FileNotFoundError("Arquivo não encontrado"),
    ) as carregar_mock: 
        with TestClient(app) as client:
            assert app.state.artefato is None

            resposta = client.get("/ready")

            assert resposta.status_code == 503
            assert resposta.json() == {"status": "not_ready"}

            carregar_mock.assert_called_once()


def test_ready_retorna_200_com_artefato():
    artefato_simulado = {
        "pipeline": object(),
        "limiar": 0.50,
        "versao_artefato": "1.0",
    }

    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        return_value=artefato_simulado,
    ) as carregar_mock:
        with TestClient(app) as client:
            assert app.state.artefato is artefato_simulado

            resposta = client.get("/ready")

            assert resposta.status_code == 200
            assert resposta.json() == {"status": "ready"}

            client.get("/ready")
            carregar_mock.assert_called_once()


def test_predict_retorna_200(client_com_modelo, entrada_valida):
    resultado_simulado = pd.DataFrame({
        "probabilidade": [0.60],
        "classificacao": [1],
    })

    with patch(
        "src.stroke_prediction.api.main.prever",
        return_value=resultado_simulado,
    ) as prever_mock:
        resposta = client_com_modelo.post(
            "/predict",
            json=entrada_valida,
        )

        assert resposta.status_code == 200
        assert resposta.json() == {
            "score": 0.60,
            "classificacao": 1,
            "versao_artefato": "1.0",
        }

        prever_mock.assert_called_once()

        _, dados = prever_mock.call_args.args

        esperado = pd.DataFrame([
            StrokeInput.model_validate(entrada_valida).model_dump()
        ])

        pd.testing.assert_frame_equal(dados, esperado)



def test_predict_retorna_503_sem_modelo(entrada_valida):
    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        side_effect=FileNotFoundError("Modelo ausente"),
    ):
        with TestClient(app) as client:
            with patch(
                "src.stroke_prediction.api.main.prever"
            ) as prever_mock:
                resposta = client.post("/predict", json=entrada_valida)

                assert resposta.status_code == 503
                assert resposta.json() == {
                    "detail": "Modelo indisponível."
                }
                prever_mock.assert_not_called()


def test_predict_rejeita_bmi_invalido(
    client_com_modelo, entrada_valida
):
    dados = {**entrada_valida, "bmi": "valor_invalido"}

    with patch(
        "src.stroke_prediction.api.main.prever"
    ) as prever_mock:
        resposta = client_com_modelo.post("/predict", json=dados)

        assert resposta.status_code == 422
        assert any(
            erro["loc"] == ["body", "bmi"]
            for erro in resposta.json()["detail"]
        )
        prever_mock.assert_not_called()


def test_predict_rejeita_campo_obrigatorio_ausente(
    client_com_modelo, entrada_valida
):
    dados = entrada_valida.copy()
    del dados["age"]

    with patch(
        "src.stroke_prediction.api.main.prever"
    ) as prever_mock:
        resposta = client_com_modelo.post("/predict", json=dados)

        assert resposta.status_code == 422
        assert any(
            erro["loc"] == ["body", "age"]
            and erro["type"] == "missing"
            for erro in resposta.json()["detail"]
        )
        prever_mock.assert_not_called()


def test_predict_rejeita_score_invalido(
    client_com_modelo, entrada_valida
):
    resultado_invalido = pd.DataFrame({
        "probabilidade": [1.45],
        "classificacao": [1],
    })

    with patch(
        "src.stroke_prediction.api.main.prever",
        return_value=resultado_invalido,
    ):
        resposta = client_com_modelo.post(
            "/predict",
            json=entrada_valida,
        )

    assert resposta.status_code == 500
    assert resposta.json() == {
        "detail": "Erro interno na resposta do modelo."
    }


def test_middleware_registra_latencia(caplog):
    nome_logger = "src.stroke_prediction.api.main"

    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        side_effect=FileNotFoundError("Modelo ausente"),
    ):
        with caplog.at_level(logging.INFO, logger=nome_logger):
            with TestClient(app) as client:
                resposta = client.get("/health")

    assert resposta.status_code == 200

    registros = [
        registro
        for registro in caplog.records
        if registro.name == nome_logger
        and registro.levelno == logging.INFO
    ]

    padrao = re.compile(
        r"HTTP GET /health status=200 duration_ms=(\d+\.\d+)"
    )

    correspondencias = [
        resultado
        for registro in registros
        if (resultado := padrao.fullmatch(registro.getMessage()))
    ]

    assert len(correspondencias) == 1, (
        "Esperado um log INFO de latência para GET /health"
    )

    duracao_ms = float(correspondencias[0].group(1))

    assert duracao_ms >= 0


def test_middleware_contabiliza_requisicoes():
    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        side_effect=FileNotFoundError("Modelo ausente"),
    ):
        with TestClient(app) as client:
            resposta_health = client.get("/health")
            resposta_ready = client.get("/ready")
            resposta_inexistente = client.get("/rota-inexistente")

            assert resposta_health.status_code == 200
            assert resposta_ready.status_code == 503
            assert resposta_inexistente.status_code == 404

            assert app.state.metricas.consultar() == {
                "total": 3,
                "2xx": 1,
                "4xx": 1,
                "5xx": 1,
            }


def test_middleware_atualiza_janela_temporal():
    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        side_effect=FileNotFoundError("Modelo ausente"),
    ):
        with patch(
            "src.stroke_prediction.api.window_metrics.monotonic",
            return_value=1000.0,
        ):
            with TestClient(app) as client:
                assert client.get("/health").status_code == 200
                assert client.get("/ready").status_code == 503
                assert client.get("/rota-inexistente").status_code == 404

                assert app.state.metricas.consultar() == {
                    "total": 3,
                    "2xx": 1,
                    "4xx": 1,
                    "5xx": 1,
                }

                assert app.state.metricas_janela.consultar() == {
                    "total": 3,
                    "5xx": 1,
                }


def test_lifespan_inicia_e_cancela_monitor():
    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        side_effect=FileNotFoundError("Modelo ausente"),
    ):
        with TestClient(app):
            tarefa = app.state.tarefa_monitor

            assert isinstance(tarefa, asyncio.Task)
            assert not tarefa.done()

        # Sair do contexto executa o shutdown e aguarda o monitor
        assert tarefa.done()
        assert tarefa.cancelled()
        assert app.state.artefato is None


def test_lifespan_inicializa_estado_monitor():
    iniciou = Event()

    async def monitor_simulado(**kwargs):
        iniciou.set()

        # Permanece aguardando até o shutdown cancelar a tarefa
        await asyncio.Event().wait()

    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        side_effect=FileNotFoundError("Modelo ausente"),
    ):
        with patch(
            "src.stroke_prediction.api.main.monitorar_alertas",
            new_callable=AsyncMock,
            side_effect=monitor_simulado,
        ) as monitor_mock:
            with TestClient(app):
                estado = app.state.estado_monitor
                tarefa = app.state.tarefa_monitor

                assert isinstance(estado, EstadoMonitor)
                assert estado.consultar() == {
                    "falhas_consecutivas": 0,
                    "sucessos_consecutivos": 0,
                    "degradado": False,
                }

                assert iniciou.wait(timeout=2)

                monitor_mock.assert_awaited_once_with(
                    metricas_janela=app.state.metricas_janela,
                    controle=app.state.controle_alertas,
                    estado_monitor=estado,
                    intervalo_segundos=30,
                    controle_reinicios=app.state.controle_reinicios,
                )

                assert (
                    monitor_mock.await_args.kwargs["controle_reinicios"]
                    is app.state.controle_reinicios
                )

                assert not tarefa.done()

            assert tarefa.done()
            assert tarefa.cancelled()
            assert app.state.artefato is None


def test_monitor_health_exige_autenticacao(monkeypatch):
    monkeypatch.delenv("MONITOR_API_KEY", raising=False)

    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        side_effect=FileNotFoundError("Modelo ausente"),
    ):
        with TestClient(app) as client:
            # Chave esperada não configurada
            resposta = client.get("/internal/monitor-health")
            assert resposta.status_code == 503

            monkeypatch.setenv(
                "MONITOR_API_KEY",
                "chave-ficticia-teste",
            )

            # Cabeçalho ausente
            resposta = client.get("/internal/monitor-health")
            assert resposta.status_code == 403

            # Chave incorreta
            resposta = client.get(
                "/internal/monitor-health",
                headers={"X-Monitor-Key": "chave-incorreta"},
            )
            assert resposta.status_code == 403

            # Chave correta
            resposta = client.get(
                "/internal/monitor-health",
                headers={"X-Monitor-Key": "chave-ficticia-teste"},
            )
            assert resposta.status_code == 200
            assert resposta.json() == {
                "falhas_consecutivas": 0,
                "sucessos_consecutivos": 0,
                "degradado": False,
                "inativo": False,
                "tarefa_ativa": True,
            }

            # A rota protegida não aparece na documentação pública
            schema = client.get("/openapi.json").json()
            assert "/internal/monitor-health" not in schema["paths"]


def test_monitor_degradado_nao_afeta_readiness(monkeypatch):
    monkeypatch.setenv(
        "MONITOR_API_KEY",
        "chave-ficticia-teste",
    )

    artefato = {
        "pipeline": Mock(),
        "limiar": 0.50,
        "versao_artefato": "1.0",
    }

    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        return_value=artefato,
    ):
        with TestClient(app) as client:
            for _ in range(3):
                app.state.estado_monitor.registrar_falha()

            resposta_ready = client.get("/ready")

            assert resposta_ready.status_code == 200
            assert resposta_ready.json() == {"status": "ready"}

            resposta_monitor = client.get(
                "/internal/monitor-health",
                headers={"X-Monitor-Key": "chave-ficticia-teste"},
            )

            assert resposta_monitor.status_code == 200
            assert resposta_monitor.json() == {
                "falhas_consecutivas": 3,
                "sucessos_consecutivos": 0,
                "degradado": True,
                "inativo": False,
                "tarefa_ativa": True,
            }


def test_monitor_health_detecta_inatividade(monkeypatch):
    monkeypatch.setenv(
        "MONITOR_API_KEY",
        "chave-ficticia-teste",
    )

    artefato = {
        "pipeline": Mock(),
        "limiar": 0.50,
        "versao_artefato": "1.0",
    }

    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        return_value=artefato,
    ):
        with patch(
            "src.stroke_prediction.api.monitor_health.monotonic",
            return_value=1000.0,
        ) as relogio:
            with TestClient(app) as client:
                for segundo, esperado in [
                    (1060.0, False),
                    (1090.0, False),
                    (1091.0, True),
                ]:
                    relogio.return_value = segundo

                    resposta = client.get(
                        "/internal/monitor-health",
                        headers={
                            "X-Monitor-Key": "chave-ficticia-teste"
                        },
                    )

                    assert resposta.status_code == 200
                    assert resposta.json() == {
                        "falhas_consecutivas": 0,
                        "sucessos_consecutivos": 0,
                        "degradado": False,
                        "inativo": esperado,
                        "tarefa_ativa": True,
                    }


def test_monitor_health_detecta_tarefa_encerrada(monkeypatch):
    monkeypatch.setenv("MONITOR_API_KEY", "chave-ficticia-teste")

    artefato = {
        "pipeline": Mock(),
        "limiar": 0.50,
        "versao_artefato": "1.0",
    }
    tarefa_simulada = Mock()
    tarefa_simulada.done.return_value = True

    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        return_value=artefato,
    ):
        with TestClient(app) as client:
            tarefa_original = app.state.tarefa_monitor

            with patch.object(app.state, "tarefa_monitor", tarefa_simulada):
                resposta = client.get(
                    "/internal/monitor-health",
                    headers={"X-Monitor-Key": "chave-ficticia-teste"},
                )

                assert resposta.status_code == 200
                assert resposta.json()["tarefa_ativa"] is False
                tarefa_simulada.done.assert_called_once_with()

            assert app.state.tarefa_monitor is tarefa_original

        assert tarefa_original.done()
        assert tarefa_original.cancelled()


def test_shutdown_nao_registra_falha_na_supervisao():
    artefato = {
        "pipeline": Mock(),
        "limiar": 0.50,
        "versao_artefato": "1.0",
    }

    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        return_value=artefato,
    ):
        with patch(
            "src.stroke_prediction.api.alert_supervision.logger.error"
        ) as error_mock:
            with TestClient(app):
                tarefa = app.state.tarefa_monitor

                assert not tarefa.done()

            # O shutdown já cancelou e aguardou a tarefa.
            assert tarefa.done()
            assert tarefa.cancelled()

            error_mock.assert_not_called()


def test_shutdown_apos_falha_do_monitor():
    artefato = {
        "pipeline": Mock(),
        "limiar": 0.50,
        "versao_artefato": "1.0",
    }
    falha = RuntimeError("Falha inesperada do monitor")
    erro_registrado = Event()

    def sinalizar_log(*args, **kwargs):
        erro_registrado.set()

    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        return_value=artefato,
    ):
        with patch(
            "src.stroke_prediction.api.main.monitorar_alertas",
            new_callable=AsyncMock,
            side_effect=falha,
        ):
            with patch(
                "src.stroke_prediction.api.alert_supervision.logger.error",
                side_effect=sinalizar_log,
            ) as error_mock:
                with TestClient(app) as client:
                    tarefa = app.state.tarefa_monitor

                    resposta = client.get("/health")
                    assert resposta.status_code == 200

                    # Aguarda a supervisão sem usar sleep.
                    assert erro_registrado.wait(timeout=2)
                    error_mock.assert_called_once()

                    tipo, erro, traceback = (
                        error_mock.call_args.kwargs["exc_info"]
                    )

                    assert tipo is RuntimeError
                    assert erro is falha
                    assert traceback is not None

                # O contexto deve encerrar sem propagar RuntimeError.
                assert tarefa.done()
                assert not tarefa.cancelled()
                assert app.state.artefato is None


def test_reinicializacoes_concorrentes_respeitam_limite():
    controle = ControleReinicializacoes(maximo_tentativas=3)
    barreira = Barrier(10)

    def tentar_reinicio():
        # Aguarda as dez threads antes de iniciar as tentativas.
        barreira.wait(timeout=10)

        try:
            return controle.registrar_reinicio()
        except RuntimeError:
            return None

    with ThreadPoolExecutor(max_workers=10) as executor:
        tarefas = [
            executor.submit(tentar_reinicio)
            for _ in range(10)
        ]

        resultados = [
            tarefa.result(timeout=10)
            for tarefa in tarefas
        ]

    autorizadas = [
        resultado
        for resultado in resultados
        if resultado is not None
    ]
    bloqueadas = resultados.count(None)

    assert len(autorizadas) == 3
    assert sorted(autorizadas) == [1, 2, 3]
    assert bloqueadas == 7
    assert controle.pode_reiniciar() is False


def test_lifespan_inicia_nova_execucao_do_monitor():
    artefato = {
        "pipeline": Mock(),
        "limiar": 0.50,
        "versao_artefato": "1.0",
    }

    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        return_value=artefato,
    ):
        with patch.object(
            EstadoMonitor,
            "iniciar_nova_execucao",
            autospec=True,
        ) as iniciar_mock:
            with TestClient(app):
                estado = app.state.estado_monitor
                tarefa = app.state.tarefa_monitor

                iniciar_mock.assert_called_once_with(estado)
                assert not tarefa.done()

            assert tarefa.done()
            assert tarefa.cancelled()
            iniciar_mock.assert_called_once_with(estado)


def test_fabrica_cria_tarefa_com_supervisao():
    tarefas_criadas = []
    supervisao_executada = Event()

    async def monitor_simulado(**kwargs):
        await asyncio.Event().wait()

    def observar_criacao(aplicacao):
        tarefa = criar_tarefa_monitor(aplicacao)
        tarefas_criadas.append(tarefa)
        return tarefa

    artefato = {
        "pipeline": Mock(),
        "limiar": 0.50,
        "versao_artefato": "1.0",
    }

    with (
        patch(
            "src.stroke_prediction.api.main.carregar_modelo",
            return_value=artefato,
        ),
        patch(
            "src.stroke_prediction.api.main.monitorar_alertas",
            new_callable=AsyncMock,
            side_effect=monitor_simulado,
        ),
        patch(
            "src.stroke_prediction.api.main.supervisionar_termino",
            side_effect=lambda tarefa: supervisao_executada.set(),
        ) as supervisor_mock,
        patch(
            "src.stroke_prediction.api.main.criar_tarefa_monitor",
            side_effect=observar_criacao,
        ) as fabrica_mock,
    ):
        with TestClient(app):
            fabrica_mock.assert_called_once_with(app)

            assert len(tarefas_criadas) == 1
            tarefa = tarefas_criadas[0]

            assert isinstance(tarefa, asyncio.Task)
            assert app.state.tarefa_monitor is tarefa
            assert not tarefa.done()

        # O shutdown cancela e aguarda a tarefa.
        assert tarefa.done()
        assert tarefa.cancelled()

        # Confirma que o callback registrado foi executado.
        assert supervisao_executada.wait(timeout=2)
        supervisor_mock.assert_called_once_with(tarefa)


def test_lifespan_inicializa_controle_recuperacao():
    artefato = {
        "pipeline": Mock(),
        "limiar": 0.50,
        "versao_artefato": "1.0",
    }

    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        return_value=artefato,
    ):
        with TestClient(app) as client:
            controle = app.state.controle_reinicios
            tarefa_monitor = app.state.tarefa_monitor

            assert isinstance(controle, ControleReinicializacoes)
            assert controle.pode_reiniciar() is True
            assert app.state.tarefa_recuperacao is None

            assert isinstance(tarefa_monitor, asyncio.Task)
            assert not tarefa_monitor.done()

            resposta = client.get("/health")
            assert resposta.status_code == 200

        assert tarefa_monitor.done()
        assert tarefa_monitor.cancelled()


def test_shutdown_cancela_recuperacao_pendente():
    recuperacao_iniciada = Event()

    async def recuperacao_simulada():
        recuperacao_iniciada.set()
        await asyncio.Event().wait()

    def criar_monitor_com_recuperacao(aplicacao):
        # Executado no event loop do TestClient durante o startup.
        tarefa_monitor = criar_tarefa_monitor(aplicacao)

        aplicacao.state.tarefa_recuperacao = asyncio.create_task(
            recuperacao_simulada()
        )

        return tarefa_monitor

    artefato = {
        "pipeline": Mock(),
        "limiar": 0.50,
        "versao_artefato": "1.0",
    }

    with (
        patch(
            "src.stroke_prediction.api.main.carregar_modelo",
            return_value=artefato,
        ),
        patch(
            "src.stroke_prediction.api.main.criar_tarefa_monitor",
            side_effect=criar_monitor_com_recuperacao,
        ),
    ):
        with TestClient(app):
            assert recuperacao_iniciada.wait(timeout=2)

            tarefa_recuperacao = app.state.tarefa_recuperacao
            tarefa_monitor = app.state.tarefa_monitor

            assert isinstance(tarefa_recuperacao, asyncio.Task)
            assert not tarefa_recuperacao.done()
            assert not tarefa_monitor.done()

        assert tarefa_recuperacao.done()
        assert tarefa_recuperacao.cancelled()

        assert tarefa_monitor.done()
        assert tarefa_monitor.cancelled()

        assert app.state.artefato is None


def test_lifespan_sinaliza_encerramento():
    artefato = {
        "pipeline": Mock(),
        "limiar": 0.50,
        "versao_artefato": "1.0",
    }

    with patch(
        "src.stroke_prediction.api.main.carregar_modelo",
        return_value=artefato,
    ):
        with TestClient(app):
            tarefa_monitor = app.state.tarefa_monitor

            assert app.state.encerrando is False
            assert not tarefa_monitor.done()

        assert app.state.encerrando is True
        assert tarefa_monitor.done()
        assert tarefa_monitor.cancelled()
        assert app.state.artefato is None


def test_nao_agenda_recuperacao_durante_shutdown():
    async def executar():
        app_teste = FastAPI()
        app_teste.state.encerrando = True
        app_teste.state.tarefa_recuperacao = None

        tarefa = Mock(spec=asyncio.Task)

        with patch(
            "src.stroke_prediction.api.main.recuperar_tarefa_encerrada",
            new_callable=AsyncMock,
        ) as recuperar_mock:
            agendar_recuperacao(app_teste, tarefa)

            recuperar_mock.assert_not_called()
            assert app_teste.state.tarefa_recuperacao is None

    asyncio.run(executar())


def test_nao_agenda_segunda_recuperacao_ativa():
    async def executar():
        app_teste = FastAPI()
        app_teste.state.encerrando = False

        recuperacao_existente = Mock(spec=asyncio.Task)
        recuperacao_existente.done.return_value = False
        app_teste.state.tarefa_recuperacao = recuperacao_existente

        tarefa = Mock(spec=asyncio.Task)

        with patch(
            "src.stroke_prediction.api.main.recuperar_tarefa_encerrada",
            new_callable=AsyncMock,
        ) as recuperar_mock:
            agendar_recuperacao(app_teste, tarefa)

            recuperar_mock.assert_not_called()
            recuperacao_existente.done.assert_called_once_with()

            assert (
                app_teste.state.tarefa_recuperacao
                is recuperacao_existente
            )

    asyncio.run(executar())


def test_agenda_recuperacao_quando_permitida():
    async def executar():
        app_teste = FastAPI()
        app_teste.state.encerrando = False
        app_teste.state.tarefa_recuperacao = None

        controle = ControleReinicializacoes(maximo_tentativas=3)
        estado = EstadoMonitor()

        app_teste.state.controle_reinicios = controle
        app_teste.state.estado_monitor = estado

        tarefa_encerrada = Mock(spec=asyncio.Task)
        tarefa_encerrada.done.return_value = True
        tarefa_encerrada.cancelled.return_value = False

        nova_tarefa = Mock(spec=asyncio.Task)

        with patch(
            "src.stroke_prediction.api.restart_recovery.executar_reinicio",
            new_callable=AsyncMock,
            return_value=nova_tarefa,
        ) as executar_mock:
            agendar_recuperacao(app_teste, tarefa_encerrada)

            tarefa_recuperacao = app_teste.state.tarefa_recuperacao

            assert isinstance(tarefa_recuperacao, asyncio.Task)

            resultado = await tarefa_recuperacao

            assert resultado is nova_tarefa
            assert tarefa_recuperacao.done()
            assert not tarefa_recuperacao.cancelled()

            executar_mock.assert_awaited_once()

            argumentos = executar_mock.call_args.kwargs

            assert argumentos["atraso"] == 1.0
            assert argumentos["estado_monitor"] is estado
            assert callable(argumentos["criar_tarefa"])

            # Confirma que a função fornecida usa a fábrica correta.
            with patch(
                "src.stroke_prediction.api.main.criar_tarefa_monitor",
                return_value=nova_tarefa,
            ) as criar_mock:
                retorno = argumentos["criar_tarefa"]()

                assert retorno is nova_tarefa
                criar_mock.assert_called_once_with(app_teste)

    asyncio.run(executar())


def test_termino_com_erro_supervisiona_e_agenda_recuperacao():
    async def executar():
        app_teste = FastAPI()
        app_teste.state.encerrando = False

        async def falhar():
            raise RuntimeError("Falha simulada do monitor")

        tarefa = asyncio.create_task(falhar())

        with pytest.raises(RuntimeError):
            await tarefa

        with (
            patch(
                "src.stroke_prediction.api.alert_supervision.logger.error"
            ) as error_mock,
            patch(
                "src.stroke_prediction.api.main.agendar_recuperacao"
            ) as agendar_mock,
        ):
            tratar_termino_monitor(app_teste, tarefa)

            error_mock.assert_called_once()

            tipo, erro, traceback = (
                error_mock.call_args.kwargs["exc_info"]
            )
            assert tipo is RuntimeError
            assert isinstance(erro, RuntimeError)
            assert traceback is not None

            agendar_mock.assert_called_once_with(app_teste, tarefa)

    asyncio.run(executar())


def test_cancelamento_supervisiona_sem_agendar_recuperacao():
    async def executar():
        app_teste = FastAPI()
        app_teste.state.encerrando = False

        async def aguardar():
            await asyncio.Event().wait()

        tarefa = asyncio.create_task(aguardar())
        tarefa.cancel()

        with pytest.raises(asyncio.CancelledError):
            await tarefa

        with (
            patch(
                "src.stroke_prediction.api.main.supervisionar_termino"
            ) as supervisor_mock,
            patch(
                "src.stroke_prediction.api.main.agendar_recuperacao"
            ) as agendar_mock,
        ):
            tratar_termino_monitor(app_teste, tarefa)

            supervisor_mock.assert_called_once_with(tarefa)
            agendar_mock.assert_not_called()

    asyncio.run(executar())


def test_erro_durante_shutdown_nao_agenda_recuperacao():
    async def executar():
        app_teste = FastAPI()
        app_teste.state.encerrando = True

        async def falhar():
            raise RuntimeError("Falha simulada durante o shutdown")

        tarefa = asyncio.create_task(falhar())

        with pytest.raises(RuntimeError) as erro_capturado:
            await tarefa

        assert not tarefa.cancelled()

        with (
            patch(
                "src.stroke_prediction.api.alert_supervision.logger.error"
            ) as error_mock,
            patch(
                "src.stroke_prediction.api.main.agendar_recuperacao"
            ) as agendar_mock,
        ):
            tratar_termino_monitor(app_teste, tarefa)

            error_mock.assert_called_once()
            tipo, erro, traceback = error_mock.call_args.kwargs["exc_info"]
            assert tipo is RuntimeError
            assert erro is erro_capturado.value
            assert traceback is not None
            agendar_mock.assert_not_called()

    asyncio.run(executar())


def test_recuperacao_automatica_substitui_monitor_com_falha():
    monitor_b_iniciado = Event()
    tarefas_executadas = []

    async def monitor_simulado(**kwargs):
        tarefas_executadas.append(asyncio.current_task())

        if len(tarefas_executadas) == 1:
            raise RuntimeError("Falha simulada")

        monitor_b_iniciado.set()
        await asyncio.Event().wait()

    artefato = {
        "pipeline": Mock(),
        "limiar": 0.50,
        "versao_artefato": "1.0",
    }

    with (
        patch(
            "src.stroke_prediction.api.main.carregar_modelo",
            return_value=artefato,
        ),
        patch(
            "src.stroke_prediction.api.main.monitorar_alertas",
            new_callable=AsyncMock,
            side_effect=monitor_simulado,
        ) as monitor_mock,
    ):
        with TestClient(app):
            assert monitor_b_iniciado.wait(timeout=4), (
                "O Monitor B não iniciou após a recuperação."
            )

            monitor_mock.assert_awaited()
            assert monitor_mock.await_count == 2
            assert monitor_mock.call_count == 2

            tarefa_a, tarefa_b = tarefas_executadas

            assert tarefa_a is not tarefa_b
            assert tarefa_a.done()
            assert not tarefa_a.cancelled()

            assert app.state.tarefa_monitor is tarefa_b
            assert not tarefa_b.done()

        assert tarefa_b.done()
        assert tarefa_b.cancelled()
        assert app.state.artefato is None


def test_recuperacao_automatica_respeita_limite_de_tentativas(monkeypatch):
    monkeypatch.setenv("MONITOR_API_KEY", "chave-ficticia-teste")

    limite_processado = Event()
    atrasos = []
    execucoes = []

    async def monitor_simulado(**kwargs):
        execucoes.append(asyncio.current_task())
        raise RuntimeError("Falha simulada")

    async def executor_simulado(
        atraso,
        estado_monitor,
        criar_tarefa,
    ):
        atrasos.append(atraso)
        estado_monitor.iniciar_nova_execucao()
        return criar_tarefa()

    async def observar_recuperacao(**kwargs):
        # Executa a orquestração real, incluindo o planejador.
        resultado = await recuperar_tarefa_encerrada(**kwargs)

        if resultado is None:
            limite_processado.set()

        return resultado

    async def aguardar_fim_recuperacao():
        return await app.state.tarefa_recuperacao

    artefato = {
        "pipeline": Mock(),
        "limiar": 0.50,
        "versao_artefato": "1.0",
    }

    with (
        patch(
            "src.stroke_prediction.api.main.carregar_modelo",
            return_value=artefato,
        ),
        patch(
            "src.stroke_prediction.api.main.monitorar_alertas",
            new_callable=AsyncMock,
            side_effect=monitor_simulado,
        ) as monitor_mock,
        patch(
            "src.stroke_prediction.api.restart_recovery.executar_reinicio",
            new_callable=AsyncMock,
            side_effect=executor_simulado,
        ) as executor_mock,
        patch(
            "src.stroke_prediction.api.main.recuperar_tarefa_encerrada",
            new_callable=AsyncMock,
            side_effect=observar_recuperacao,
        ) as recuperar_mock,
    ):
        with TestClient(app) as client:
            assert limite_processado.wait(timeout=4), (
                "A recuperação não chegou ao limite de tentativas."
            )

            # Executa a espera no event loop do TestClient.
            # O sinal acima pode ocorrer imediatamente antes do retorno.
            assert client.portal.call(aguardar_fim_recuperacao) is None

            assert atrasos == [1.0, 2.0, 4.0]
            assert executor_mock.await_count == 3

            assert monitor_mock.await_count == 4
            assert monitor_mock.call_count == 4
            assert len(execucoes) == 4
            assert len(set(execucoes)) == 4

            # Cada falha foi processada, inclusive a quarta,
            # cuja recuperação foi recusada.
            assert recuperar_mock.await_count == 4
            assert app.state.controle_reinicios.pode_reiniciar() is False

            assert app.state.tarefa_monitor is execucoes[-1]
            assert all(tarefa.done() for tarefa in execucoes)
            assert app.state.tarefa_recuperacao.done()

            resposta = client.get("/health")
            assert resposta.status_code == 200
            assert resposta.json() == {"status": "ok"}

            resposta_ready = client.get("/ready")
            assert resposta_ready.status_code == 200
            assert resposta_ready.json() == {"status": "ready"}

            resposta_monitor = client.get(
                "/internal/monitor-health",
                headers={"X-Monitor-Key": "chave-ficticia-teste"},
            )
            assert resposta_monitor.status_code == 200
            assert resposta_monitor.json()["tarefa_ativa"] is False

            assert monitor_mock.call_count == 4
            assert monitor_mock.await_count == 4
            assert len(execucoes) == 4

        assert monitor_mock.call_count == 4
        assert app.state.artefato is None


def test_falha_na_recuperacao_fica_observavel(caplog):
    falha_recuperacao = RuntimeError("Falha simulada na recuperação")
    recuperacao_iniciada = Event()

    async def executor_com_falha(**kwargs):
        recuperacao_iniciada.set()
        raise falha_recuperacao

    async def aguardar_termino_recuperacao():
        tarefa = app.state.tarefa_recuperacao

        # Aguarda sem propagar nem consultar a exceção da tarefa.
        concluidas, _ = await asyncio.wait({tarefa}, timeout=2)
        assert tarefa in concluidas

        # Permite executar eventuais callbacks de supervisão.
        await asyncio.sleep(0)
        return tarefa

    artefato = {
        "pipeline": Mock(),
        "limiar": 0.50,
        "versao_artefato": "1.0",
    }

    with (
        caplog.at_level(logging.ERROR),
        patch(
            "src.stroke_prediction.api.main.carregar_modelo",
            return_value=artefato,
        ),
        patch(
            "src.stroke_prediction.api.main.monitorar_alertas",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Falha do Monitor A"),
        ) as monitor_mock,
        patch(
            "src.stroke_prediction.api.restart_recovery.executar_reinicio",
            new_callable=AsyncMock,
            side_effect=executor_com_falha,
        ) as executor_mock,
    ):
        with TestClient(app) as client:
            assert recuperacao_iniciada.wait(timeout=4)

            recuperacao = client.portal.call(
                aguardar_termino_recuperacao
            )
            monitor = app.state.tarefa_monitor

            # Guarda os registros antes de iniciar o shutdown.
            erros_antes_shutdown = [
                registro
                for registro in caplog.records
                if registro.exc_info
                and registro.exc_info[1] is falha_recuperacao
            ]

            assert monitor.done()
            assert recuperacao.done()
            assert monitor_mock.await_count == 1
            executor_mock.assert_awaited_once()

            # Falha da recuperação não é falha de avaliação das métricas.
            assert (
                app.state.estado_monitor.consultar()["falhas_consecutivas"]
                == 0
            )

            assert client.get("/health").status_code == 200

        # O encerramento deve terminar mesmo com a recuperação falhada.
        assert monitor.done()
        assert recuperacao.done()
        assert app.state.artefato is None
        assert monitor_mock.call_count == 1

    assert erros_antes_shutdown, (
        "A falha da recuperação não foi registrada antes do shutdown."
    )


def test_recuperacao_completa_apos_estabilidade(monkeypatch):
    monitor_b_iniciado = Event()
    execucoes = []
    fila = None
    avaliacao_concluida = None

    controle = ControleReinicializacoes(maximo_tentativas=3)

    async def sleep_controlado(intervalo):
        await fila.get()

    async def monitor_simulado(**kwargs):
        nonlocal fila, avaliacao_concluida

        assert kwargs["controle_reinicios"] is controle
        execucoes.append(asyncio.current_task())

        if len(execucoes) == 1:
            raise RuntimeError("Falha do Monitor A")

        fila = asyncio.Queue()
        avaliacao_concluida = asyncio.Event()
        monitor_b_iniciado.set()

        await monitorar_alertas_real(**kwargs)

    async def executor_sem_espera(
        atraso, estado_monitor, criar_tarefa
    ):
        assert atraso == 1.0
        estado_monitor.iniciar_nova_execucao()
        return criar_tarefa()

    def confirmar_observada(estado_monitor, controle):
        resultado = confirmar_recuperacao_real(
            estado_monitor=estado_monitor,
            controle=controle,
        )
        avaliacao_concluida.set()
        return resultado

    async def executar_avaliacao(instante):
        relogio.return_value = instante
        avaliacao_concluida.clear()
        fila.put_nowait(None)

        await asyncio.wait_for(
            avaliacao_concluida.wait(),
            timeout=2,
        )

    artefato = {
        "pipeline": Mock(),
        "limiar": 0.50,
        "versao_artefato": "1.0",
    }

    # Substitui apenas o acesso a asyncio no módulo do monitor.
    # O sleep do executor e o event loop permanecem independentes.
    monkeypatch.setattr(
        "src.stroke_prediction.api.alert_monitor.asyncio",
        SimpleNamespace(sleep=sleep_controlado),
    )

    with (
        patch(
            "src.stroke_prediction.api.monitor_health.monotonic",
            return_value=1000.0,
        ) as relogio,
        patch(
            "src.stroke_prediction.api.main.carregar_modelo",
            return_value=artefato,
        ),
        patch(
            "src.stroke_prediction.api.main.ControleReinicializacoes",
            return_value=controle,
        ),
        patch(
            "src.stroke_prediction.api.main.monitorar_alertas",
            new_callable=AsyncMock,
            side_effect=monitor_simulado,
        ) as monitor_mock,
        patch(
            "src.stroke_prediction.api.restart_recovery.executar_reinicio",
            new_callable=AsyncMock,
            side_effect=executor_sem_espera,
        ),
        patch(
            "src.stroke_prediction.api.alert_monitor.confirmar_recuperacao",
            side_effect=confirmar_observada,
        ),
        patch.object(
            controle,
            "registrar_reinicio",
            wraps=controle.registrar_reinicio,
        ) as registrar_mock,
        patch.object(
            controle,
            "reiniciar_contagem",
            wraps=controle.reiniciar_contagem,
        ) as zerar_mock,
    ):
        with TestClient(app) as client:
            assert monitor_b_iniciado.wait(timeout=4)

            assert app.state.controle_reinicios is controle
            registrar_mock.assert_called_once_with()

            # Usa a avaliação real com contadores conhecidos.
            with patch.object(
                app.state.metricas_janela,
                "consultar",
                return_value={"total": 100, "5xx": 0},
            ):
                client.portal.call(executar_avaliacao, 1000.0)
                zerar_mock.assert_not_called()

                client.portal.call(executar_avaliacao, 1060.0)
                zerar_mock.assert_not_called()

                client.portal.call(executar_avaliacao, 1120.0)
                zerar_mock.assert_called_once_with()

            assert monitor_mock.await_count == 2
            assert app.state.tarefa_monitor is execucoes[1]

            tarefa_b = app.state.tarefa_monitor
            assert not tarefa_b.done()

            # Após a recuperação, a numeração começa novamente em 1.
            assert controle.registrar_reinicio() == 1

        assert tarefa_b.done()
        assert tarefa_b.cancelled()
        assert app.state.artefato is None
