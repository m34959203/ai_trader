# tests/conftest.py
from __future__ import annotations

import asyncio
import os
import pytest
from httpx import AsyncClient
from httpx import ASGITransport
from asgi_lifespan import LifespanManager

# ... остальное как было ...

@pytest.fixture
async def client() -> AsyncClient:
    """
    Лёгкий httpx-клиент поверх ASGI транспортного слоя.
    Корректно выполняем startup/shutdown хуки FastAPI (lifespan) для httpx<0.27
    """
    from src.main import app
    transport = ASGITransport(app=app)  # без параметра lifespan
    async with LifespanManager(app):    # запускаем startup/shutdown
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac
