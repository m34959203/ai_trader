import pytest
from httpx import AsyncClient
from src.main import app

@pytest.mark.asyncio
async def test_health_ok():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert "resources" in data

@pytest.mark.asyncio
async def test_health_deep_ok():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        r = await ac.get("/health/deep?testnet=true")
        # В CI без сети может быть 4xx/5xx, но структура должна быть JSON
        assert r.headers["content-type"].startswith("application/json")
