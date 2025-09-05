import pytest
from httpx import AsyncClient, ASGITransport
from src.main import app
import src.main as main  # ВАЖНО: патчим тут, где функция используется

def _client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

@pytest.mark.asyncio
async def test_prices_json(monkeypatch):
    import pandas as pd
    # 2 свечи — ожидаем rows==2
    df = pd.DataFrame([
        {"timestamp": "2025-08-25T12:00:00Z", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100.0},
        {"timestamp": "2025-08-25T13:00:00Z", "open": 1.5, "high": 2.5, "low": 1.0, "close": 2.0, "volume": 200.0},
    ])
    monkeypatch.setattr(main, "get_prices", lambda **kwargs: df)

    async with _client() as ac:
        r = await ac.get("/prices", params={
            "source": "alphavantage", "ticker": "AAPL",
            "interval": "1h", "period": "7d", "fmt": "json"
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["rows"] == 2
        assert len(body["data"]) == 2
        assert isinstance(body["data"][0]["timestamp"], str)

@pytest.mark.asyncio
async def test_prices_csv(monkeypatch):
    import pandas as pd
    df = pd.DataFrame([
        {"timestamp": "2025-08-25T12:00:00Z", "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 100.0}
    ])
    monkeypatch.setattr(main, "get_prices", lambda **kwargs: df)

    async with _client() as ac:
        r = await ac.get("/prices", params={
            "source": "alphavantage", "ticker": "AAPL",
            "interval": "1h", "period": "7d", "fmt": "csv"
        })
        assert r.status_code == 200
        assert "text/csv" in r.headers.get("content-type", "")
        assert "attachment; filename=\"prices.csv\"" in r.headers.get("content-disposition", "")
        assert b"timestamp,open,high,low,close,volume" in r.content
