from __future__ import annotations

from copy import deepcopy

import pytest

from services.broker_gateway import SimulatedBrokerGateway
from services.live_trading import LiveTradingCoordinator
from services.model_router import DEFAULT_MODEL_CONFIG, router_singleton


@pytest.fixture(scope="module")
def live_router():
    cfg = deepcopy(DEFAULT_MODEL_CONFIG)
    cfg["models"]["sentiment"] = {
        "name": "sentiment:finbert",
        "params": {"use_pipeline": False},
    }
    router_singleton.configure(cfg)
    return router_singleton


def test_live_trading_executes_order_when_risk_allows(live_router):
    gateway = SimulatedBrokerGateway()
    coordinator = LiveTradingCoordinator(live_router, gateway)

    result = coordinator.route_and_execute(
        "ETHUSDT",
        {
            "price": 2000.0,
            "trend_ma_fast": 2100.0,
            "trend_ma_slow": 1900.0,
            "rsi": 25.0,
            "macd": 1.2,
            "macd_signal": 0.5,
            "atr": 10.0,
            "equity": 10000.0,
            "day_pnl": 500.0,
        },
        news_text="strong earnings",
    )

    assert result["executed"] is True
    assert result["order"] is not None
    assert result["order"]["status"] == "filled"
    assert result["request_id"]
    assert result["retries"] >= 0
    assert gateway.positions()["ETHUSDT"] > 0


def test_live_trading_respects_daily_loss_block(live_router):
    gateway = SimulatedBrokerGateway()
    coordinator = LiveTradingCoordinator(live_router, gateway)

    result = coordinator.route_and_execute(
        "BTCUSDT",
        {
            "price": 50000.0,
            "trend_ma_fast": 48000.0,
            "trend_ma_slow": 51000.0,
            "rsi": 75.0,
            "macd": -1.2,
            "macd_signal": -0.2,
            "atr": 800.0,
            "equity": 20000.0,
            "day_pnl": -2000.0,
        },
        news_text="regulatory crackdown",
    )

    assert result["executed"] is False
    assert result["order"] is None
    assert result["error"] is None
    assert result["request_id"] is None

