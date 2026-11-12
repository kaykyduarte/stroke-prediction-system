import logging
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import ResponseValidationError

from src.stroke_prediction.models.persistence import carregar_modelo
from src.stroke_prediction.api.schemas import StrokeInput, StrokeOutput
from src.stroke_prediction.models.predict import prever


logger = logging.getLogger(__name__)

RAIZ_PROJETO = Path(__file__).resolve().parents[3]
CAMINHO_MODELO = RAIZ_PROJETO / "artifacts" / "stroke_model.joblib"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artefato = None
    try:
        app.state.artefato = carregar_modelo(CAMINHO_MODELO)
    except FileNotFoundError:
        logger.warning(
            "Arquivo do modelo não encontrado: %s",
            CAMINHO_MODELO,
        )

    try:
        yield
    finally:
        app.state.artefato = None


app = FastAPI(lifespan=lifespan)


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