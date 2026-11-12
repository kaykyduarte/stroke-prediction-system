FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./

RUN python -m pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
COPY artifacts/stroke_model.joblib ./artifacts/stroke_model.joblib

RUN groupadd --system app \
    && useradd --system --gid app --no-create-home \
      --shell /usr/sbin/nologin app \
    && chmod -R a+rX /app/src /app/artifacts

EXPOSE 8000

USER app

CMD ["python", "-m", "uvicorn", "src.stroke_prediction.api.main:app", "--host", "0.0.0.0", "--port", "8000"]