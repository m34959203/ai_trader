from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import httpx
import pytest

from services.broker_gateway import (
    BinanceBrokerGateway,
    BrokerGatewayError,
    OrderRequest,
    OrderResponse,
    SimulatedBrokerGateway,
)
from services.live_trading import LiveTradingCoordinator
import services.live_trading as live_module


def _make_gateway(
    responders: Dict[Tuple[str, str], Callable[[httpx.Request], httpx.Response]],
    *,
    time_value: float = 1_600_000_000.0,
) -> Tuple[BinanceBrokerGateway, httpx.Client]:
    def handler(request: httpx.Request) -> httpx.Response:
        key = (request.method, request.url.path)
        responder = responders.get(key)
        if responder is None:
            raise AssertionError(f"No responder registered for {key}")
        return responder(request)

    transport = httpx.MockTransport(handler)
    client = httpx.Client(
        transport=transport,
        base_url="https://testnet.binance.vision",
        headers={"X-MBX-APIKEY": "test-key"},
    )
    gateway = BinanceBrokerGateway(
        api_key="test-key",
        api_secret="super-secret",
        testnet=True,
        client=client,
        recv_window=5000,
        time_provider=lambda: time_value,
    )
    gateway._owns_client = True  # type: ignore[attr-defined]
    return gateway, client


def test_binance_gateway_status_and_balances_round_trip():
    responders: Dict[Tuple[str, str], Callable[[httpx.Request], httpx.Response]] = {}

    def order_responder(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        assert params["origClientOrderId"] == "order-1"
        assert params["symbol"] == "BTCUSDT"
        return httpx.Response(
            200,
            json={
                "symbol": "BTCUSDT",
                "status": "PARTIALLY_FILLED",
                "executedQty": "0.500",
                "cummulativeQuoteQty": "100.0",
                "clientOrderId": "order-1",
                "time": 1_600_000_000_000,
            },
        )

    def account_responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "balances": [
                    {"asset": "USDT", "free": "100.0", "locked": "50.0"},
                    {"asset": "BTC", "free": "0.25", "locked": "0.05"},
                ]
            },
        )

    responders[("GET", "/api/v3/order")] = order_responder
    responders[("GET", "/api/v3/account")] = account_responder
    gateway, client = _make_gateway(responders)
    try:
        status = gateway.get_order_status("order-1", symbol="btcusdt")
        assert status.status == "partially_filled"
        assert pytest.approx(status.filled_quantity, rel=1e-6) == 0.5
        assert pytest.approx(status.average_price, rel=1e-6) == 200.0

        balances = gateway.balances()
        assert pytest.approx(balances["USDT"]["total"], rel=1e-6) == 150.0
        assert pytest.approx(balances["BTC"]["total"], rel=1e-6) == 0.3

        positions = gateway.positions()
        assert pytest.approx(positions["USDT"], rel=1e-6) == 150.0
    finally:
        gateway.close()
        client.close()


class PartialFillGateway(SimulatedBrokerGateway):
    def __init__(self) -> None:
        super().__init__(immediate_fill=False)
        self.attempts = 0
        self._status_emitted = False

    def submit_order(self, request: OrderRequest) -> OrderResponse:
        if self.attempts == 0:
            self.attempts += 1
            raise BrokerGatewayError("temporary outage")
        response = super().submit_order(request)
        pending = OrderResponse(
            request_id=response.request_id,
            status="new",
            filled_quantity=0.0,
            average_price=None,
            submitted_at=response.submitted_at,
            raw={"request": request.to_dict()},
        )
        self._orders[-1] = pending
        return pending

    def get_order_status(
        self,
        request_id: str,
        *,
        symbol: Optional[str] = None,
    ) -> Optional[OrderResponse]:
        for idx, order in enumerate(self._orders):
            if order.request_id == request_id and not self._status_emitted:
                req = order.raw["request"]
                filled_qty = req["quantity"] / 2
                updated = OrderResponse(
                    request_id=order.request_id,
                    status="partially_filled",
                    filled_quantity=filled_qty,
                    average_price=req["metadata"].get("reference_price"),
                    submitted_at=order.submitted_at,
                    raw=order.raw,
                )
                self._orders[idx] = updated
                self._status_emitted = True
                self._update_position(req["symbol"], req["side"], filled_qty)
                return updated
            if order.request_id == request_id:
                return order
        return None


def test_coordinator_handles_retries_and_status_poll(monkeypatch):
    fake_gateway = PartialFillGateway()
    coordinator = LiveTradingCoordinator(
        router=None,
        gateway=fake_gateway,
        status_poll_attempts=2,
        order_retry_attempts=1,
    )

    def fake_decide(symbol, features, news_text=None, router=None):
        return {
            "trading_blocked": False,
            "risk_fraction": 0.1,
            "limits": {"equity": features.get("equity", 0.0)},
            "signal": {"signal": "buy"},
        }

    monkeypatch.setattr(live_module, "decide_and_execute", fake_decide)

    summary = coordinator.route_and_execute(
        "ETHUSDT",
        {"price": 100.0, "equity": 1000.0},
    )

    assert summary["retries"] == 1
    assert summary["executed"] is True
    assert summary["order"]["status"] == "partially_filled"
    assert summary["order"]["filled_quantity"] > 0

    cached = coordinator.get_order_status(summary["request_id"], refresh=False)
    assert cached is not None
    assert cached["status"] == "partially_filled"

    snapshot = coordinator.sync_account()
    assert "ETHUSDT" in snapshot["positions"]
    assert snapshot["balances"]["ETHUSDT"]["total"] > 0
