import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import ResponseValidationError

from src.stroke_prediction.models.persistence import carregar_modelo
from src.stroke_prediction.api.schemas import StrokeInput, StrokeOutput
from src.stroke_prediction.api.metrics import MetricasAPI
from src.stroke_prediction.api.window_metrics import MetricasJanela
from src.stroke_prediction.models.predict import prever
from src.stroke_prediction.api.alert_monitor import monitorar_alertas
from src.stroke_prediction.api.alert_state import ControleAlertasHisterese
from src.stroke_prediction.api.monitor_health import EstadoMonitor
from src.stroke_prediction.api.security import validar_chave_monitor
from src.stroke_prediction.api.alert_supervision import supervisionar_termino
from src.stroke_prediction.api.retry_state import ControleReinicializacoes
from src.stroke_prediction.api.restart_recovery import recuperar_tarefa_encerrada


logger = logging.getLogger(__name__)

RAIZ_PROJETO = Path(__file__).resolve().parents[3]
CAMINHO_MODELO = RAIZ_PROJETO / "artifacts" / "stroke_model.joblib"

def criar_tarefa_monitor(app: FastAPI) -> asyncio.Task:
    tarefa = asyncio.create_task(
        monitorar_alertas(
            metricas_janela=app.state.metricas_janela,
            controle=app.state.controle_alertas,
            estado_monitor=app.state.estado_monitor,
            intervalo_segundos=30,
            controle_reinicios=app.state.controle_reinicios,
        )
    )

    tarefa.add_done_callback(
        lambda tarefa_encerrada: tratar_termino_monitor(
            app,
            tarefa_encerrada,
        )
    )
    app.state.tarefa_monitor = tarefa

    return tarefa

def supervisionar_recuperacao(tarefa: asyncio.Task) -> None:
    if tarefa.cancelled():
        return

    erro = tarefa.exception()

    if erro is not None:
        logger.error(
            "Falha inesperada na tarefa de recuperação do monitor.",
            exc_info=(type(erro), erro, erro.__traceback__),
        )
        
def agendar_recuperacao(
    app: FastAPI,
    tarefa: asyncio.Task,
) -> None:
    if app.state.encerrando:
        return

    recuperacao_atual = app.state.tarefa_recuperacao

    if recuperacao_atual is not None and not recuperacao_atual.done():
        return

    app.state.tarefa_recuperacao = asyncio.create_task(
        recuperar_tarefa_encerrada(
            tarefa=tarefa,
            controle=app.state.controle_reinicios,
            estado_monitor=app.state.estado_monitor,
            criar_tarefa=lambda: criar_tarefa_monitor(app),
        )
    )

    app.state.tarefa_recuperacao.add_done_callback(
        supervisionar_recuperacao
    )


def tratar_termino_monitor(
        app: FastAPI,
        tarefa: asyncio.Task,
) -> None:
    supervisionar_termino(tarefa)

    if tarefa.cancelled():
        return

    if app.state.encerrando:
        return

    agendar_recuperacao(app, tarefa)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.encerrando = False
    app.state.metricas = MetricasAPI()
    app.state.metricas_janela = MetricasJanela(janela_segundos=300)
    app.state.estado_monitor = EstadoMonitor()
    app.state.controle_alertas = ControleAlertasHisterese(
        limiar_ativacao=10.0,
        limiar_recuperacao=5.0,
    )
    app.state.controle_reinicios = ControleReinicializacoes(
        maximo_tentativas=3,
    )
    app.state.tarefa_recuperacao = None
    app.state.artefato = None

    try:
        app.state.artefato = carregar_modelo(CAMINHO_MODELO)
    except FileNotFoundError:
        logger.warning(
            "Arquivo do modelo não encontrado: %s",
            CAMINHO_MODELO,
        )

    app.state.estado_monitor.iniciar_nova_execucao()
    criar_tarefa_monitor(app)

    try:
        yield
    finally:
        app.state.encerrando = True
        try:
            tarefa_recuperacao = app.state.tarefa_recuperacao

            if tarefa_recuperacao is not None:
                tarefa_recuperacao.cancel()

                try:
                    await tarefa_recuperacao
                except asyncio.CancelledError:
                    pass
                except Exception:
                    # o callback de supervisão é responsável pelo log
                    pass

        finally:
            tarefa_monitor = app.state.tarefa_monitor
            tarefa_monitor.cancel()

            try:
                await tarefa_monitor
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
            finally:
                app.state.artefato = None


app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def registrar_latencia(request: Request, call_next):
    inicio = perf_counter()

    resposta = await call_next(request)

    duracao_ms = (perf_counter() - inicio) * 1000

    request.app.state.metricas.registrar(resposta.status_code)
    request.app.state.metricas_janela.registrar(resposta.status_code)

    rota = request.scope.get("route")
    caminho_rota = getattr(rota, "path", "unmatched")

    logger.info(
        "HTTP %s %s status=%s duration_ms=%.2f",
        request.method,
        caminho_rota,
        resposta.status_code,
        duracao_ms,
    )

    return resposta


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request):
    if request.app.state.artefato is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready"},
        )

    return {"status": "ready"}


@app.get(
    "/internal/monitor-health",
    dependencies=[Depends(validar_chave_monitor)],
    include_in_schema=False,
)
def monitor_health(request: Request):
    estado = request.app.state.estado_monitor
    tarefa = request.app.state.tarefa_monitor

    resposta = estado.consultar()
    resposta["inativo"] = estado.esta_inativo()
    resposta["tarefa_ativa"] = not tarefa.done()
    
    return resposta


@app.post("/predict", response_model=StrokeOutput)
def predict(entrada: StrokeInput, request: Request):
    artefato = request.app.state.artefato

    if artefato is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo indisponível.",
        )

    dados = pd.DataFrame([entrada.model_dump()])

    resultados = prever(artefato, dados)
    resultado = resultados.iloc[0]

    return {
        "score": float(resultado["probabilidade"]),
        "classificacao": int(resultado["classificacao"]),
        "versao_artefato": artefato["versao_artefato"]
    }


@app.exception_handler(ResponseValidationError)
async def tratar_resposta_invalida(
    request: Request,
    exc: ResponseValidationError,
):
    logger.error(
        "Resposta inválida gerada pelo serviço de inferência."
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Erro interno na resposta do modelo."
        },
    )
