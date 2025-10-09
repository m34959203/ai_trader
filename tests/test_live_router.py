from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers.live_trading import router as live_router
from services.broker_gateway import SimulatedBrokerGateway
from services.live_trading import LiveTradingCoordinator
from services.model_router import DEFAULT_MODEL_CONFIG, ModelRouter


def _build_app() -> TestClient:
    model_router = ModelRouter(DEFAULT_MODEL_CONFIG)
    model_router.configure(DEFAULT_MODEL_CONFIG)
    gateway = SimulatedBrokerGateway()
    coordinator = LiveTradingCoordinator(model_router, gateway)

    app = FastAPI()
    app.state.live_trading = coordinator
    app.include_router(live_router)
    return TestClient(app)


def test_live_status_endpoint_returns_configuration():
    client = _build_app()
    response = client.get("/live/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["configured"] is True
    assert payload["gateway"] == "SimulatedBrokerGateway"


def test_live_trade_endpoint_executes_and_returns_order():
    client = _build_app()
    request_body = {
        "symbol": "BTCUSDT",
        "features": {
            "symbol": "BTCUSDT",
            "price": 20000.0,
            "rsi": 55.0,
            "atr": 150.0,
            "equity": 50000.0,
            "day_pnl": 0.0,
        },
        "news_text": "BTC consolidates near resistance",
    }
    response = client.post("/live/trade", json=request_body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["decision"]["signal"]["signal"] in {"buy", "sell", "hold"}
    if payload["executed"]:
        assert payload["order"] is not None
    else:
        assert payload["order"] is None
    assert payload["executed"] in {True, False}
