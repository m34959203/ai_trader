# tests/test_ui_smoke.py
from __future__ import annotations

import os
import pytest
from httpx import AsyncClient, ASGITransport
from fastapi.responses import JSONResponse

# ВАЖНО: включаем UI ДО импорта приложения
os.environ.setdefault("FEATURE_UI", "1")
os.environ.setdefault("DISABLE_DOCS", "1")  # чтобы не мешали /docs в CI
# По умолчанию пусть UI рендерит SIM-данные, но мы всё равно замокаем вызовы
os.environ.setdefault("UI_EXEC_MODE", "sim")
os.environ.setdefault("UI_EXEC_TESTNET", "1")

from src.main import app  # noqa: E402


def _client():
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


# ----------------------------
# Хелперы для моков UI-партиалов
# ----------------------------
async def _fake_balance(*_, **__):
    return {
        "exchange": "sim",
        "testnet": True,
        "equity_usdt": 10_000.0,
        "free": {"USDT": 10_000.0},
        "locked": {},
        "risk": {
            "enabled": True,
            "daily_start_equity": 10_000.0,
            "daily_max_loss_pct": 0.02,
            "max_trades_per_day": 10,
            "trades_today": 0,
        },
    }


async def _fake_positions(*_, **__):
    return [
        {"symbol": "BTCUSDT", "qty": 0.25, "side": "buy", "sl": None, "tp": None},
        {"symbol": "ETHUSDT", "qty": -1.0, "side": "sell", "sl": None, "tp": None},
    ]


async def _fake_last_orders(*_, **__):
    # минимальный набор полей, который обычно рисуется в шаблоне
    return [
        {
            "created_at": 1710000000000,
            "exchange": "sim",
            "testnet": True,
            "symbol": "BTCUSDT",
            "side": "buy",
            "type": "market",
            "qty": 0.25,
            "price": 65000.0,
            "status": "filled",
            "order_id": "1",
            "client_order_id": "cli-1",
        },
        {
            "created_at": 1710001000000,
            "exchange": "sim",
            "testnet": True,
            "symbol": "ETHUSDT",
            "side": "sell",
            "type": "limit",
            "qty": 1.0,
            "price": 3200.0,
            "status": "new",
            "order_id": "2",
            "client_order_id": "cli-2",
        },
    ]


async def _restricted_error_response(*_, **__):
    return JSONResponse(
        status_code=451,
        content={
            "error": "binance_restricted_location",
            "code": 0,
            "details": {
                "message": "Binance API недоступен из вашего региона (HTTP 451). Переключитесь в режим симуляции или используйте VPN/другой доступ.",
                "original": "Binance API error 451 (code=0): Service unavailable from a restricted location",
            },
        },
    )


# ----------------------------
# Тесты
# ----------------------------
@pytest.mark.asyncio
async def test_ui_root_page_ok():
    async with _client() as ac:
        r = await ac.get("/ui/")
        assert r.status_code == 200, r.text
        body = r.text.lower()
        # Ключевые блоки-заглушки с id (htmx потом подменит содержимое)
        assert 'id="balance"' in body
        assert 'id="positions"' in body
        assert 'id="orders"' in body
        # Заголовок страницы
        assert "ai-trader dashboard".lower() in body


@pytest.mark.asyncio
async def test_ui_partial_balance_ok(monkeypatch):
    # Патчим сервисный вызов на фейковые данные
    import routers.ui as ui
    monkeypatch.setattr(ui, "_get_balance", _fake_balance)

    async with _client() as ac:
        r = await ac.get("/ui/partials/balance", params={"mode": "sim", "testnet": "true"})
        assert r.status_code == 200, r.text
        body = r.text
        # Либо секция с id, либо явный заголовок "Баланс"
        assert ('id="balance"' in body) or ("Баланс" in body)


@pytest.mark.asyncio
async def test_ui_partial_balance_restricted(monkeypatch):
    import routers.ui as ui

    monkeypatch.setattr(ui, "_get_balance", _restricted_error_response)

    async with _client() as ac:
        r = await ac.get("/ui/partials/balance", params={"mode": "binance", "testnet": "false"})
        assert r.status_code == 200, r.text
        body = r.text
        assert "Binance API недоступен из вашего региона" in body
        assert "restricted location" in body


@pytest.mark.asyncio
async def test_ui_partial_positions_ok(monkeypatch):
    import routers.ui as ui
    monkeypatch.setattr(ui, "_list_positions", _fake_positions)

    async with _client() as ac:
        r = await ac.get("/ui/partials/positions", params={"mode": "sim", "testnet": "true"})
        assert r.status_code == 200, r.text
        body = r.text
        # Должны увидеть контейнер или один из тикеров
        assert ('id="positions"' in body) or ("BTCUSDT" in body) or ("ETHUSDT" in body)


@pytest.mark.asyncio
async def test_ui_partial_positions_restricted(monkeypatch):
    import routers.ui as ui

    monkeypatch.setattr(ui, "_list_positions", _restricted_error_response)

    async with _client() as ac:
        r = await ac.get("/ui/partials/positions", params={"mode": "binance", "testnet": "false"})
        assert r.status_code == 200, r.text
        body = r.text
        assert "Binance API недоступен из вашего региона" in body
        assert "restricted location" in body


@pytest.mark.asyncio
async def test_ui_partial_orders_ok(monkeypatch):
    # Если в проекте есть crud_orders — замокаем, иначе endpoint сам вернёт пустой список.
    import routers.ui as ui

    if getattr(ui, "crud_orders", None) is not None:
        # Переопределяем асинхронную функцию репозитория
        async def _fake_get_last_orders(session, limit=20):
            return await _fake_last_orders()

        monkeypatch.setattr(ui.crud_orders, "get_last_orders", _fake_get_last_orders)

        # А зависимость сессии может быть не нужна — роутер её пробрасывает как Depends
        # Ничего не трогаем: ui.partial_orders сам вызовет crud_orders.get_last_orders(...)
    # Иначе оставляем как есть — фрагмент отрисует пустой список «без падений»

    async with _client() as ac:
        r = await ac.get("/ui/partials/orders")
        assert r.status_code == 200, r.text
        body = r.text
        # Либо секция с id, либо заголовок/таблица
        ok_markers = [
            'id="orders"',
            "Последние ордера",
            "<table",
            "order",  # английский маркер на случай другой локализации шаблона
        ]
        assert any(m in body for m in ok_markers)
