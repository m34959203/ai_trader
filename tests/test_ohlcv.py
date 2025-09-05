# tests/test_ohlcv_api.py
from __future__ import annotations

import io
import csv
import time
import typing as t
import pytest
from httpx import AsyncClient, ASGITransport
from asgi_lifespan import LifespanManager

# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные генераторы данных
# ──────────────────────────────────────────────────────────────────────────────

def _tf_to_sec(tf: str | None) -> int:
    """
    Простая конверсия таймфрейма в секунды:
    '1m','3m','5m','15m','30m' → минуты
    '1h','4h'                   → часы
    '1d'                        → сутки
    Если не распознали — дефолт 3600.
    """
    if not tf:
        return 3600
    tf = tf.strip().lower()
    try:
        # отделяем число и суффикс
        num = ""
        unit = ""
        for ch in tf:
            if ch.isdigit():
                num += ch
            else:
                unit += ch
        n = int(num) if num else 1
        if unit == "m":
            return n * 60
        if unit == "h":
            return n * 3600
        if unit == "d":
            return n * 86400
    except Exception:
        pass
    return 3600


def _fake_series(
    n: int = 5,
    *,
    asset: str = "BTCUSDT",
    tf: str = "1h",
    source: str = "binance",
    start_ts: int | None = None,
    start_price: float = 100_000.0,
    vol0: float = 1000.0,
    step_sec: int | None = None,
) -> list[dict[str, t.Any]]:
    """
    Универсальный генератор OHLCV, создаёт n баров с уникальными ts.

    - start_ts по умолчанию = «далёкое прошлое» (чтобы не ловить коллизии)
    - step_sec: если не задан, вычисляется из tf (m/h/d)
    """
    if start_ts is None:
        # на пару суток назад, плюс небольшой буфер
        start_ts = int(time.time()) - 2 * 86400 - n * 3600

    if step_sec is None:
        step_sec = _tf_to_sec(tf)

    rows: list[dict[str, t.Any]] = []
    price = float(start_price)
    for i in range(n):
        ts = int(start_ts + i * step_sec)
        o = price
        h = o * 1.01
        l = o * 0.99
        c = (h + l) / 2.0
        v = vol0 + i
        rows.append(
            {
                "source": source,
                "asset": asset,
                "tf": tf,
                "ts": ts,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            }
        )
        price = c
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Фикстуры
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
async def client() -> AsyncClient:
    """
    Лёгкий httpx-клиент поверх ASGI транспорта, с явным запуском lifespan.
    Совместимо с httpx 0.28 — параметр lifespan у ASGITransport не используем.
    """
    from src.main import app
    transport = ASGITransport(app=app)
    async with LifespanManager(app):
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# ──────────────────────────────────────────────────────────────────────────────
# Базовый «живой» тест сервиса
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ping(client: AsyncClient):
    r = await client.get("/ping")
    assert r.status_code == 200
    j = r.json()
    assert j.get("status") == "ok"
    assert "message" in j


# ──────────────────────────────────────────────────────────────────────────────
# Запись из источника (мокаем сеть) и базовое чтение
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_store_and_read_binance(client: AsyncClient, monkeypatch: pytest.MonkeyPatch):
    # Мокаем источник binance.fetch, чтобы не ходить в сеть
    from sources import binance as binance_module

    rows = _fake_series(n=5, source="binance", asset="BTCUSDT", tf="1h")
    monkeypatch.setattr(binance_module, "fetch", lambda *a, **k: rows)

    # Сохраняем 5 свечей
    r = await client.post(
        "/prices/store",
        json={"source": "binance", "symbol": "BTCUSDT", "timeframe": "1h", "limit": 5},
    )
    assert r.status_code == 200, r.text
    assert r.json()["stored"] == 5

    # Читаем 3 самые ранние в asc
    r2 = await client.get(
        "/ohlcv",
        params={"source": "binance", "ticker": "BTCUSDT", "timeframe": "1h", "limit": 3, "order": "asc"},
    )
    assert r2.status_code == 200, r2.text
    data = r2.json()
    assert "candles" in data
    candles = data["candles"]
    assert len(candles) == 3
    assert candles[0]["asset"] == "BTCUSDT"
    # проверим что возврат идёт по возрастанию времени
    ts_list = [c["ts"] for c in candles]
    assert ts_list == sorted(ts_list)


# ──────────────────────────────────────────────────────────────────────────────
# Подсчёты и сводная статистика (count + stats)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_count_and_stats(client: AsyncClient, monkeypatch: pytest.MonkeyPatch):
    from sources import binance as binance_module

    rows = _fake_series(n=7, source="binance", asset="BTCUSDT", tf="1h")
    monkeypatch.setattr(binance_module, "fetch", lambda *args, **kwargs: rows)

    # сначала запишем
    r = await client.post(
        "/prices/store",
        json={"source": "binance", "symbol": "BTCUSDT", "timeframe": "1h", "limit": 7},
    )
    assert r.status_code == 200, r.text

    c = await client.get("/ohlcv/count", params={"source": "binance", "ticker": "BTCUSDT", "timeframe": "1h"})
    assert c.status_code == 200, c.text
    assert isinstance(c.json().get("count"), int)
    assert c.json()["count"] > 0

    s = await client.get("/ohlcv/stats", params={"source": "binance", "ticker": "BTCUSDT", "timeframe": "1h"})
    assert s.status_code == 200, s.text
    body = s.json()
    assert set(body.keys()) == {"min_ts", "max_ts", "count"}
    assert body["count"] >= 1
    assert body["min_ts"] <= body["max_ts"]


# ──────────────────────────────────────────────────────────────────────────────
# CSV-экспорт
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_csv_export(client: AsyncClient, monkeypatch: pytest.MonkeyPatch):
    # Загружаем немного новых данных, чтобы CSV точно не был пустой
    from sources import binance as binance_module

    rows = _fake_series(n=4, source="binance", asset="ETHUSDT", tf="1h")
    monkeypatch.setattr(binance_module, "fetch", lambda *a, **k: rows)

    r = await client.post(
        "/prices/store",
        json={"source": "binance", "symbol": "ETHUSDT", "timeframe": "1h", "limit": 4},
    )
    assert r.status_code == 200, r.text

    csv_resp = await client.get(
        "/ohlcv.csv",
        params={"source": "binance", "ticker": "ETHUSDT", "timeframe": "1h", "order": "asc", "limit": 10},
    )
    assert csv_resp.status_code == 200, csv_resp.text
    assert csv_resp.headers["content-type"].startswith("text/csv")

    buf = io.StringIO(csv_resp.text)
    reader = csv.reader(buf)
    header = next(reader)
    assert header == ["source", "asset", "tf", "ts", "open", "high", "low", "close", "volume"]
    rows_read = list(reader)
    assert len(rows_read) >= 1  # должны быть строки


# ──────────────────────────────────────────────────────────────────────────────
# Пагинация через next_offset
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pagination_next_offset(client: AsyncClient, monkeypatch: pytest.MonkeyPatch):
    from sources import binance as binance_module
    rows = _fake_series(n=7, source="binance", asset="XRPUSDT", tf="1h")
    monkeypatch.setattr(binance_module, "fetch", lambda *a, **k: rows)

    # загрузим 7 строк
    r = await client.post(
        "/prices/store",
        json={"source": "binance", "symbol": "XRPUSDT", "timeframe": "1h", "limit": 7},
    )
    assert r.status_code == 200

    # страница 1: limit=3 → next_offset=3
    p1 = await client.get(
        "/ohlcv",
        params={"source": "binance", "ticker": "XRPUSDT", "timeframe": "1h", "limit": 3, "offset": 0, "order": "asc"},
    )
    assert p1.status_code == 200
    d1 = p1.json()
    assert len(d1["candles"]) == 3
    assert d1["next_offset"] == 3

    # страница 2: offset=3 limit=3 → next_offset=6
    p2 = await client.get(
        "/ohlcv",
        params={"source": "binance", "ticker": "XRPUSDT", "timeframe": "1h", "limit": 3, "offset": 3, "order": "asc"},
    )
    assert p2.status_code == 200
    d2 = p2.json()
    assert len(d2["candles"]) == 3
    assert d2["next_offset"] == 6

    # страница 3: offset=6 limit=3 → влезет 1 строка, next_offset=None
    p3 = await client.get(
        "/ohlcv",
        params={"source": "binance", "ticker": "XRPUSDT", "timeframe": "1h", "limit": 3, "offset": 6, "order": "asc"},
    )
    assert p3.status_code == 200
    d3 = p3.json()
    assert len(d3["candles"]) == 1
    assert d3["next_offset"] is None


# ──────────────────────────────────────────────────────────────────────────────
# Порядок сортировки: desc
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_order_desc(client: AsyncClient, monkeypatch: pytest.MonkeyPatch):
    from sources import binance as binance_module
    rows = _fake_series(n=5, source="binance", asset="SOLUSDT", tf="1h")
    monkeypatch.setattr(binance_module, "fetch", lambda *a, **k: rows)

    r = await client.post(
        "/prices/store",
        json={"source": "binance", "symbol": "SOLUSDT", "timeframe": "1h", "limit": 5},
    )
    assert r.status_code == 200

    resp = await client.get(
        "/ohlcv",
        params={"source": "binance", "ticker": "SOLUSDT", "timeframe": "1h", "limit": 5, "order": "desc"},
    )
    assert resp.status_code == 200
    candles = resp.json()["candles"]
    ts_list = [c["ts"] for c in candles]
    assert ts_list == sorted(ts_list, reverse=True)


# ──────────────────────────────────────────────────────────────────────────────
# Ошибочная ситуация: неизвестный источник в /prices/store
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_store_unknown_source_returns_400(client: AsyncClient):
    r = await client.post(
        "/prices/store",
        json={"source": "unknown_src", "symbol": "ABC", "timeframe": "1h", "limit": 5},
    )
    assert r.status_code == 400
    body = r.json()
    # общий обработчик HTTPException возвращает поле "error"
    assert body.get("ok") is False or "error" in body or "detail" in body
