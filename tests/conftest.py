# tests/conftest.py
from __future__ import annotations

import asyncio
import os
import sys
import pytest

import httpx
from httpx import AsyncClient
try:
    from httpx import ASGITransport  # httpx>=0.28
except Exception:  # pragma: no cover
    ASGITransport = None  # type: ignore

from asgi_lifespan import LifespanManager

# ──────────────────────────────────────────────────────────────────────────────
# Глобальный shim: поддержка устаревшего `AsyncClient(app=...)` в тестах
# httpx≥0.28 убрал shortcut `app=...`, теперь нужно явно указывать
# `transport=ASGITransport(app=...)`. Делаем автопатч на время тестов.
# Документация: https://www.python-httpx.org/advanced/transports/
# История слома: https://github.com/encode/httpx/issues/3111
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def _httpx_app_kwarg_shim():
    if ASGITransport is None:
        # Старый httpx — ничего не делаем
        yield
        return

    orig_init = httpx.AsyncClient.__init__

    def _shim_asyncclient_init(self, *args, **kwargs):
        app = kwargs.pop("app", None)
        if app is not None and "transport" not in kwargs:
            # Прозрачно превращаем в новый стиль
            kwargs["transport"] = ASGITransport(app=app)
            kwargs.setdefault("base_url", "http://test")
        return orig_init(self, *args, **kwargs)

    httpx.AsyncClient.__init__ = _shim_asyncclient_init  # type: ignore[assignment]
    try:
        yield
    finally:
        httpx.AsyncClient.__init__ = orig_init  # type: ignore[assignment]


# ──────────────────────────────────────────────────────────────────────────────
# Фикстура клиента для тестов, использующих её напрямую
# ВАЖНО: LifespanManager обеспечивает вызовы startup/shutdown FastAPI-приложения.
# ──────────────────────────────────────────────────────────────────────────────
@pytest.fixture
async def client() -> AsyncClient:
    """
    Лёгкий httpx-клиент поверх ASGI транспортного слоя.
    Корректно выполняем startup/shutdown хуки FastAPI (lifespan).
    """
    from src.main import app  # импорт здесь, чтобы не тащить app в глобал

    if ASGITransport is None:
        # На старом httpx fallback всё равно отработает, т.к. shim выше не нужен
        transport = None
        kwargs = {"app": app, "base_url": "http://test"}  # перехватится Starlette TestClient-подобной логикой
    else:
        transport = ASGITransport(app=app)
        kwargs = {"transport": transport, "base_url": "http://test"}

    # Явно прогоняем lifespan, как рекомендует asgi-lifespan:
    # https://github.com/florimondmanca/asgi-lifespan
    async with LifespanManager(app):
        async with AsyncClient(**kwargs) as ac:
            yield ac
