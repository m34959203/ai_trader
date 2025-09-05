# syntax=docker/dockerfile:1

# ─────────────────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=UTC \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8001

# Минимальные системные пакеты (часовой пояс, curl для HEALTHCHECK)
RUN apt-get update \
 && apt-get install -y --no-install-recommends tzdata curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────
WORKDIR /app

# Сначала зависимости (лучше кэшируется)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Затем код приложения
COPY . /app

# Нерутовый пользователь
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8001

# ─────────────────────────────────────────────────────────────────────────────
# Healthcheck (ожидается эндпоинт /_livez в FastAPI)
# ─────────────────────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${UVICORN_PORT}/_livez" || exit 1

# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────
# Приложение объявлено как FastAPI app в src/main.py -> "src.main:app"
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001", "--proxy-headers", "--forwarded-allow-ips", "*"]
