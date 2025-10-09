from __future__ import annotations

from copy import deepcopy

import pytest

from services.model_router import DEFAULT_MODEL_CONFIG, router_singleton
from services.trading_service import decide_and_execute


@pytest.fixture(scope="module")
def configured_router():
    cfg = deepcopy(DEFAULT_MODEL_CONFIG)
    cfg["models"]["sentiment"] = {
        "name": "sentiment:finbert",
        "params": {"use_pipeline": False},
    }
    router_singleton.configure(cfg)
    return router_singleton


def test_decide_and_execute_blocks_on_daily_loss(configured_router):
    result = decide_and_execute(
        "BTCUSDT",
        {
            "price": 50000,
            "trend_ma_fast": 51000,
            "trend_ma_slow": 49500,
            "rsi": 30,
            "macd": 2.0,
            "macd_signal": 1.0,
            "atr": 200,
            "equity": 10000,
            "day_pnl": -700,
        },
        news_text="fraud probe",
    )
    assert result["trading_blocked"] is True
    assert result["risk_fraction"] == 0.0
    assert result["signal"]["signal"] in {"buy", "hold", "sell"}


def test_decide_and_execute_risk_adjusts(configured_router):
    result = decide_and_execute(
        "ETHUSDT",
        {
            "price": 3000,
            "trend_ma_fast": 3100,
            "trend_ma_slow": 2800,
            "rsi": 25,
            "macd": 1.5,
            "macd_signal": 0.5,
            "atr": 15,
            "equity": 20000,
            "day_pnl": 100,
        },
        news_text="record profits",
    )
    assert result["trading_blocked"] is False
    assert 0.0 <= result["risk_fraction"] <= configured_router.risk_config["per_trade_cap"]
