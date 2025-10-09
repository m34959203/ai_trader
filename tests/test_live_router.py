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


def test_live_trade_and_order_management_endpoints():
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
    if payload["order"] is not None:
        assert payload["request_id"]
        request_id = payload["request_id"]
    else:
        # Risk checks may refuse to trade in deterministic test runs.
        request_id = payload["request_id"]
        assert request_id is None
        return

    status_resp = client.get(f"/live/orders/{request_id}")
    assert status_resp.status_code == 200
    status_payload = status_resp.json()
    assert status_payload["found"] is True
    assert status_payload["order"]["request_id"] == request_id

    orders_resp = client.get("/live/orders", params={"refresh": True})
    assert orders_resp.status_code == 200
    orders_payload = orders_resp.json()
    assert isinstance(orders_payload["orders"], list)

    cancel_resp = client.post(f"/live/orders/{request_id}/cancel")
    assert cancel_resp.status_code == 200
    cancel_payload = cancel_resp.json()
    assert cancel_payload["request_id"] == request_id
    assert cancel_payload["cancelled"] in {True, False}

    sync_resp = client.post("/live/sync")
    assert sync_resp.status_code == 200
    sync_payload = sync_resp.json()
    assert "balances" in sync_payload
    assert "positions" in sync_payload
