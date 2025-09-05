from __future__ import annotations
import time
import uuid
import pytest
from httpx import AsyncClient, ASGITransport
from src.main import app


def _client():
    # ASGITransport запускает app в памяти (без реального сервера)
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.mark.asyncio
async def test_next_offset(monkeypatch):
    """
    Проверяем пагинацию /ohlcv c next_offset. Используем уникальный символ на каждый
    прогон (чтобы не пересекаться с предыдущими данными в БД).
    """
    from sources import binance as binance_module

    sym = f"BTCUSDT_PAGING_{uuid.uuid4().hex[:8]}"

    def make_rows_for_symbol(symbol: str, n: int = 12):
        # 12 часов назад + равные шаги по 1 час
        t0 = int(time.time()) - 3600 * (n + 5)
        out = []
        price = 1.0
        for i in range(n):
            ts = t0 + i * 3600
            o = price
            h = o * 1.02
            l = o * 0.98
            c = (h + l) / 2
            v = i + 1
            out.append({
                "source": "binance",
                "asset": symbol,
                "tf": "1h",
                "ts": ts,
                "open": o, "high": h, "low": l, "close": c, "volume": v
            })
            price = c
        return out

    # Мокаем fetch так, чтобы он генерировал 12 свечей под запрошенный symbol
    def fake_fetch(symbol, timeframe, limit, ts_from, ts_to):
        return make_rows_for_symbol(symbol, n=12)

    monkeypatch.setattr(binance_module, "fetch", fake_fetch)

    async with _client() as ac:
        # Записываем 12 свечей для уникального символа
        r_store = await ac.post(
            "/prices/store",
            json={"source": "binance", "symbol": sym, "timeframe": "1h", "limit": 999},
        )
        assert r_store.status_code == 200, r_store.text

        # Страница 1: 0..4
        r1 = await ac.get("/ohlcv", params={
            "source": "binance", "ticker": sym, "timeframe": "1h", "limit": 5, "offset": 0
        })
        assert r1.status_code == 200, r1.text
        body1 = r1.json()
        assert body1["next_offset"] == 5
        assert len(body1["candles"]) == 5

        # Страница 2: 5..9
        r2 = await ac.get("/ohlcv", params={
            "source": "binance", "ticker": sym, "timeframe": "1h", "limit": 5, "offset": 5
        })
        assert r2.status_code == 200, r2.text
        body2 = r2.json()
        assert body2["next_offset"] == 10
        assert len(body2["candles"]) == 5

        # Страница 3: 10..11 (последняя)
        r3 = await ac.get("/ohlcv", params={
            "source": "binance", "ticker": sym, "timeframe": "1h", "limit": 5, "offset": 10
        })
        assert r3.status_code == 200, r3.text
        body3 = r3.json()
        assert body3["next_offset"] is None
        assert len(body3["candles"]) == 2
