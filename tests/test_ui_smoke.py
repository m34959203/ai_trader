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


class _DummyLiveCoordinator:
    def __init__(self) -> None:
        self._pnl = {
            "ts": "2024-04-01T00:00:00Z",
            "start_equity": 10_000.0,
            "current_equity": 10_750.5,
            "realized_pnl": 320.25,
            "drawdown_pct": 0.0123,
            "trades_count": 7,
        }
        self._broker = {
            "updated_at": "2024-04-01T00:05:00Z",
            "connected": True,
            "gateway": "binance",
            "open_orders": [
                {
                    "request_id": "ord-1",
                    "status": "new",
                    "filled_quantity": 0.0,
                    "raw": {"request": {"symbol": "BTCUSDT"}},
                }
            ],
            "positions": {"BTCUSDT": 0.25},
            "balances": {"USDT": {"total": 12000.0, "free": 9500.0, "locked": 2500.0}},
            "last_error": None,
        }
        self._trades = [
            {
                "ts": "2024-04-01T00:03:00Z",
                "symbol": "BTCUSDT",
                "strategy": "alpha",
                "side": "buy",
                "quantity": 0.1,
                "price": 65_000.0,
                "notional": 6_500.0,
                "status": "filled",
                "executed": True,
                "request_id": "req-123",
            }
        ]
        self._limits = {
            "risk_config": {
                "per_trade_cap": 0.02,
                "daily_max_loss_pct": 0.05,
                "max_trades_per_day": 25,
            },
            "daily": {
                "start_equity": 10_000.0,
                "current_equity": 10_750.5,
                "realized_pnl": 320.25,
            },
            "strategies": [
                {
                    "name": "alpha",
                    "enabled": True,
                    "max_risk_fraction": 0.02,
                    "max_daily_trades": 15,
                    "trades_today": 3,
                    "updated_at": "2024-04-01T00:04:00Z",
                }
            ],
        }

    def pnl_snapshot(self):
        return dict(self._pnl)

    def broker_status(self):
        return dict(self._broker)

    def list_trades(self, limit: int = 20):
        return list(self._trades[:limit])

    def limits_snapshot(self):
        return dict(self._limits)


@pytest.mark.asyncio
async def test_live_widgets_require_setup(monkeypatch):
    import routers.ui as ui
    from src.main import app

    ui._LAST_DIGEST.clear()
    monkeypatch.setattr(app.state, "live_trading", None, raising=False)

    async with _client() as ac:
        for path in [
            "/ui/partials/live_pnl",
            "/ui/partials/broker_status",
            "/ui/partials/live_trades",
            "/ui/partials/limits",
        ]:
            resp = await ac.get(path)
            assert resp.status_code == 200, resp.text
            body = resp.text
            assert "Live trading не настроен" in body
            assert "doc/live_trading_requirements.md" in body


@pytest.mark.asyncio
async def test_live_widgets_render_with_coordinator(monkeypatch):
    import routers.ui as ui
    from src.main import app

    coordinator = _DummyLiveCoordinator()
    ui._LAST_DIGEST.clear()
    monkeypatch.setattr(app.state, "live_trading", coordinator, raising=False)
    monkeypatch.setattr(
        "tasks.reports.get_reports_summary",
        lambda: {
            "generated_at": "2024-04-01T00:06:00Z",
            "csv": {"size": 2048},
            "pdf": {"size": 4096},
        },
    )

    async with _client() as ac:
        pnl_resp = await ac.get("/ui/partials/live_pnl")
        assert pnl_resp.status_code == 200, pnl_resp.text
        assert "Live PnL" in pnl_resp.text
        assert "10 750.5" in pnl_resp.text or "10750.5" in pnl_resp.text

        broker_resp = await ac.get("/ui/partials/broker_status")
        assert broker_resp.status_code == 200, broker_resp.text
        assert "Статус брокера" in broker_resp.text
        assert "binance" in broker_resp.text

        trades_resp = await ac.get("/ui/partials/live_trades", params={"limit": 10})
        assert trades_resp.status_code == 200, trades_resp.text
        assert "Live сделки" in trades_resp.text
        assert "alpha" in trades_resp.text

        limits_resp = await ac.get("/ui/partials/limits")
        assert limits_resp.status_code == 200, limits_resp.text
        assert "Риск-лимиты" in limits_resp.text
        assert "alpha" in limits_resp.text
        assert "CSV" in limits_resp.text


@pytest.mark.asyncio
async def test_live_pnl_htmx_returns_204_when_snapshot_not_changed(monkeypatch):
    import routers.ui as ui
    from src.main import app

    coordinator = _DummyLiveCoordinator()
    ui._LAST_DIGEST.clear()
    monkeypatch.setattr(app.state, "live_trading", coordinator, raising=False)

    async with _client() as ac:
        first = await ac.get("/ui/partials/live_pnl", headers={"HX-Request": "true"})
        assert first.status_code == 200, first.text

        second = await ac.get("/ui/partials/live_pnl", headers={"HX-Request": "true"})
        assert second.status_code == 204, second.text
