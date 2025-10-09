from __future__ import annotations

import hashlib
import hmac
from typing import Callable, Dict, Tuple
from urllib.parse import parse_qsl, urlencode

import httpx
import pytest

from services.broker_gateway import BinanceBrokerGateway, BrokerGatewayError, OrderRequest


def _make_gateway(
    responders: Dict[Tuple[str, str], Callable[[httpx.Request], httpx.Response]],
    *,
    time_value: float = 1_600_000_000.0,
) -> Tuple[BinanceBrokerGateway, httpx.Client]:
    """Utility to create a BinanceBrokerGateway with a mocked transport."""

    def handler(request: httpx.Request) -> httpx.Response:
        key: Tuple[str, str] = (request.method, request.url.path)
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
    gateway._owns_client = True  # type: ignore[attr-defined]  # ensure close() shuts down mock client
    return gateway, client


def test_submit_order_includes_signature_and_parses_response():
    def responder(request: httpx.Request) -> httpx.Response:
        assert request.headers["X-MBX-APIKEY"] == "test-key"
        body = dict(parse_qsl(request.content.decode()))
        # Signature is computed over the body without the signature key.
        signature_payload = {k: v for k, v in body.items() if k != "signature"}
        query = urlencode(signature_payload)
        expected_signature = hmac.new(
            b"super-secret",
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        assert body["signature"] == expected_signature
        return httpx.Response(
            200,
            json={
                "symbol": "BTCUSDT",
                "status": "FILLED",
                "executedQty": "0.010",
                "cummulativeQuoteQty": "200.0",
                "price": "20000.0",
                "clientOrderId": "client-123",
            },
        )

    gateway, client = _make_gateway({("POST", "/api/v3/order"): responder})
    try:
        response = gateway.submit_order(
            OrderRequest(
                symbol="BTCUSDT",
                side="buy",
                quantity=0.01,
                metadata={"client_order_id": "client-123"},
            )
        )
    finally:
        gateway.close()
        client.close()

    assert response.status == "filled"
    assert response.request_id == "client-123"
    assert pytest.approx(response.filled_quantity, rel=1e-6) == 0.01
    assert pytest.approx(response.average_price, rel=1e-6) == 20000.0


def test_cancel_order_returns_true_on_cancelled_status():
    responders: Dict[Tuple[str, str], Callable[[httpx.Request], httpx.Response]] = {}

    def cancel_responder(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        assert params["origClientOrderId"] == "abc"
        return httpx.Response(200, json={"status": "CANCELED"})

    responders[("DELETE", "/api/v3/order")] = cancel_responder
    gateway, client = _make_gateway(responders)
    try:
        assert gateway.cancel_order("abc") is True
    finally:
        gateway.close()
        client.close()


def test_list_open_orders_maps_payloads():
    responders: Dict[Tuple[str, str], Callable[[httpx.Request], httpx.Response]] = {}

    def list_responder(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        assert params.get("symbol") == "ETHUSDT"
        return httpx.Response(
            200,
            json=[
                {
                    "symbol": "ETHUSDT",
                    "status": "NEW",
                    "executedQty": "0.0",
                    "price": "1500.0",
                    "clientOrderId": "order-1",
                    "time": 1_600_000_000_000,
                },
                {
                    "symbol": "ETHUSDT",
                    "status": "PARTIALLY_FILLED",
                    "executedQty": "1.0",
                    "cummulativeQuoteQty": "1500.0",
                    "clientOrderId": "order-2",
                    "time": 1_600_000_001_000,
                },
            ],
        )

    responders[("GET", "/api/v3/openOrders")] = list_responder
    gateway, client = _make_gateway(responders)
    try:
        orders = list(gateway.list_open_orders("ethusdt"))
    finally:
        gateway.close()
        client.close()

    assert len(orders) == 2
    assert orders[0].request_id == "order-1"
    assert orders[0].status == "new"
    assert orders[0].average_price is None
    assert orders[1].request_id == "order-2"
    assert orders[1].status == "partially_filled"
    assert pytest.approx(orders[1].average_price, rel=1e-6) == 1500.0


def test_positions_returns_non_zero_balances():
    responders: Dict[Tuple[str, str], Callable[[httpx.Request], httpx.Response]] = {}

    def account_responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "balances": [
                    {"asset": "USDT", "free": "100.0", "locked": "0.0"},
                    {"asset": "BTC", "free": "0.0", "locked": "0.0"},
                    {"asset": "ETH", "free": "0.5", "locked": "0.2"},
                ]
            },
        )

    responders[("GET", "/api/v3/account")] = account_responder
    gateway, client = _make_gateway(responders)
    try:
        positions = gateway.positions()
    finally:
        gateway.close()
        client.close()

    assert positions == {"USDT": 100.0, "ETH": 0.7}


def test_request_raises_on_error_status():
    responders: Dict[Tuple[str, str], Callable[[httpx.Request], httpx.Response]] = {}

    def error_responder(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"code": -1013, "msg": "Invalid quantity"})

    responders[("POST", "/api/v3/order")] = error_responder
    gateway, client = _make_gateway(responders)
    try:
        with pytest.raises(BrokerGatewayError):
            gateway.submit_order(OrderRequest(symbol="BTCUSDT", side="buy", quantity=0))
    finally:
        gateway.close()
        client.close()
