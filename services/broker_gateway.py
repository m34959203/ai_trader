"""Broker gateway abstractions for live order execution.

This module defines broker-agnostic contracts alongside concrete gateway
implementations used by the trading coordinator.  A lightweight
``SimulatedBrokerGateway`` remains available for tests and dry-runs, while
``BinanceBrokerGateway`` provides a production-ready adapter for Binance
Spot/Testnet REST APIs without introducing asynchronous dependencies.
"""

from __future__ import annotations

import hashlib
import hmac
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, MutableMapping, Optional, Protocol
from urllib.parse import urlencode

import httpx

from monitoring.observability import OBSERVABILITY

OrderSide = str
OrderType = str
OrderStatus = str


@dataclass(slots=True)
class OrderRequest:
    """Standardised order submission payload."""

    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = "market"
    limit_price: Optional[float] = None
    time_in_force: str = "GTC"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "quantity": float(self.quantity),
            "order_type": self.order_type,
            "limit_price": None if self.limit_price is None else float(self.limit_price),
            "time_in_force": self.time_in_force,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class OrderResponse:
    """Uniform response returned by broker gateways."""

    request_id: str
    status: OrderStatus
    filled_quantity: float
    average_price: Optional[float] = None
    submitted_at: float = field(default_factory=lambda: time.time())
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "status": self.status,
            "filled_quantity": float(self.filled_quantity),
            "average_price": None if self.average_price is None else float(self.average_price),
            "submitted_at": float(self.submitted_at),
            "raw": dict(self.raw),
        }


class BrokerGateway(Protocol):
    """Protocol describing the minimum behaviour for live broker adapters."""

    def submit_order(self, request: OrderRequest) -> OrderResponse:
        ...

    def cancel_order(self, request_id: str) -> bool:
        ...

    def list_open_orders(self, symbol: Optional[str] = None) -> Iterable[OrderResponse]:
        ...

    def get_order_status(self, request_id: str, *, symbol: Optional[str] = None) -> Optional[OrderResponse]:
        ...

    def positions(self) -> MutableMapping[str, float]:
        ...

    def balances(self) -> MutableMapping[str, Dict[str, float]]:
        ...


class BrokerGatewayError(RuntimeError):
    """Raised when a broker operation fails."""


class SimulatedBrokerGateway:
    """In-memory broker used for tests and dry-runs.

    The simulator immediately fills market orders and keeps track of net
    position sizes per symbol.  It is intentionally lightweight yet it
    mirrors the behaviour of a real gateway closely enough for unit tests
    and early-stage integrations.
    """

    def __init__(self, *, immediate_fill: bool = True) -> None:
        self._immediate_fill = immediate_fill
        self._orders: List[OrderResponse] = []
        self._positions: Dict[str, float] = {}

    def submit_order(self, request: OrderRequest) -> OrderResponse:
        if request.quantity <= 0:
            raise BrokerGatewayError("Quantity must be positive")
        order_id = request.metadata.get("client_order_id") or uuid.uuid4().hex
        status: OrderStatus
        filled_qty = 0.0
        avg_price: Optional[float] = None

        if self._immediate_fill or request.order_type == "market":
            status = "filled"
            filled_qty = request.quantity
            avg_price = self._resolve_fill_price(request)
            self._update_position(request.symbol, request.side, filled_qty)
        else:  # pragma: no cover - exercised via integration tests
            status = "submitted"

        response = OrderResponse(
            request_id=order_id,
            status=status,
            filled_quantity=filled_qty,
            average_price=avg_price,
            raw={"request": request.to_dict()},
        )
        self._orders.append(response)
        return response

    def cancel_order(self, request_id: str) -> bool:  # pragma: no cover - unused in current unit tests
        for idx, order in enumerate(self._orders):
            if order.request_id == request_id and order.status not in {"cancelled", "canceled", "filled"}:
                updated = OrderResponse(
                    request_id=order.request_id,
                    status="cancelled",
                    filled_quantity=order.filled_quantity,
                    average_price=order.average_price,
                    submitted_at=order.submitted_at,
                    raw=order.raw,
                )
                self._orders[idx] = updated
                return True
        return False

    def list_open_orders(self, symbol: Optional[str] = None) -> Iterable[OrderResponse]:
        return [
            order
            for order in self._orders
            if order.status not in {"filled", "cancelled", "canceled"}
            and (symbol is None or order.raw.get("request", {}).get("symbol") == symbol)
        ]

    def get_order_status(self, request_id: str, *, symbol: Optional[str] = None) -> Optional[OrderResponse]:
        for order in self._orders:
            if order.request_id == request_id:
                return order
        return None

    def positions(self) -> MutableMapping[str, float]:
        return dict(self._positions)

    def balances(self) -> MutableMapping[str, Dict[str, float]]:
        balances: Dict[str, Dict[str, float]] = {}
        for symbol, qty in self._positions.items():
            balances[symbol] = {
                "free": float(qty),
                "locked": 0.0,
                "total": abs(float(qty)),
            }
        return balances

    def _update_position(self, symbol: str, side: OrderSide, quantity: float) -> None:
        delta = quantity if side.lower() == "buy" else -quantity
        self._positions[symbol] = self._positions.get(symbol, 0.0) + delta

    @staticmethod
    def _resolve_fill_price(request: OrderRequest) -> Optional[float]:
        if request.order_type == "market":
            return request.metadata.get("reference_price")
        return request.limit_price


class BinanceBrokerGateway:
    """Synchronous Binance Spot/Testnet gateway implementing :class:`BrokerGateway`."""

    _MAIN_BASE_URL = "https://api.binance.com"
    _TESTNET_BASE_URL = "https://testnet.binance.vision"

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str,
        testnet: bool = True,
        base_url: Optional[str] = None,
        recv_window: int = 5000,
        timeout: float = 15.0,
        client: Optional[httpx.Client] = None,
        time_provider: Optional[Callable[[], float]] = None,
    ) -> None:
        if not api_key or not api_secret:
            raise ValueError("api_key and api_secret must be provided")

        self._api_key = api_key
        self._api_secret = api_secret.encode()
        self._recv_window = max(1, int(recv_window))
        self._time = time_provider or time.time
        base = (base_url or (self._TESTNET_BASE_URL if testnet else self._MAIN_BASE_URL)).rstrip("/")

        headers = {"X-MBX-APIKEY": api_key}
        self._client = client or httpx.Client(base_url=base, timeout=float(timeout), headers=headers)
        self._owns_client = client is None

    def close(self) -> None:
        if self._owns_client:
            try:
                self._client.close()
            except Exception:  # pragma: no cover - defensive
                pass

    def submit_order(self, request: OrderRequest) -> OrderResponse:
        payload = {
            "symbol": request.symbol.upper(),
            "side": request.side.upper(),
            "type": request.order_type.upper(),
            "quantity": request.quantity,
            "timeInForce": request.time_in_force,
        }
        if request.limit_price is not None:
            payload["price"] = request.limit_price
        if client_id := request.metadata.get("client_order_id"):
            payload["newClientOrderId"] = client_id
        if quote_qty := request.metadata.get("quote_quantity"):
            payload["quoteOrderQty"] = quote_qty

        data = self._request("POST", "/api/v3/order", payload, signed=True)
        status = str(data.get("status", "UNKNOWN")).lower()
        filled = float(data.get("executedQty", 0.0))
        avg_price = self._extract_average_price(data)
        request_id = str(data.get("clientOrderId") or data.get("orderId") or uuid.uuid4().hex)
        return OrderResponse(
            request_id=request_id,
            status=status,
            filled_quantity=filled,
            average_price=avg_price,
            raw=data,
        )

    def cancel_order(self, request_id: str) -> bool:
        payload = {"origClientOrderId": request_id}
        data = self._request("DELETE", "/api/v3/order", payload, signed=True)
        status = str(data.get("status", "")).lower()
        return status == "canceled" or status == "cancelled"

    def list_open_orders(self, symbol: Optional[str] = None) -> Iterable[OrderResponse]:
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        data = self._request("GET", "/api/v3/openOrders", params, signed=True)
        responses: List[OrderResponse] = []
        for item in data:
            status = str(item.get("status", "UNKNOWN")).lower()
            filled = float(item.get("executedQty", 0.0))
            avg_price = self._extract_average_price(item)
            request_id = str(item.get("clientOrderId") or item.get("orderId") or uuid.uuid4().hex)
            responses.append(
                OrderResponse(
                    request_id=request_id,
                    status=status,
                    filled_quantity=filled,
                    average_price=avg_price,
                    submitted_at=float(item.get("time", self._time())),
                    raw=item,
                )
            )
        return responses

    def get_order_status(self, request_id: str, *, symbol: Optional[str] = None) -> Optional[OrderResponse]:
        params: Dict[str, Any] = {"origClientOrderId": request_id}
        if symbol:
            params["symbol"] = symbol.upper()
        else:
            raise ValueError("symbol is required when querying Binance order status")
        data = self._request("GET", "/api/v3/order", params, signed=True)
        status = str(data.get("status", "UNKNOWN")).lower()
        filled = float(data.get("executedQty", 0.0))
        avg_price = self._extract_average_price(data)
        request_id = str(data.get("clientOrderId") or data.get("orderId") or request_id)
        return OrderResponse(
            request_id=request_id,
            status=status,
            filled_quantity=filled,
            average_price=avg_price,
            submitted_at=float(data.get("time", self._time())),
            raw=data,
        )

    def positions(self) -> MutableMapping[str, float]:
        balances = self.balances()
        return {asset: data["total"] for asset, data in balances.items() if data["total"] > 0}

    def balances(self) -> MutableMapping[str, Dict[str, float]]:
        account = self._request("GET", "/api/v3/account", {}, signed=True)
        balances = account.get("balances", [])
        out: Dict[str, Dict[str, float]] = {}
        for balance in balances:
            asset = balance.get("asset")
            if not asset:
                continue
            free = float(balance.get("free", 0.0))
            locked = float(balance.get("locked", 0.0))
            total = free + locked
            out[str(asset)] = {"free": free, "locked": locked, "total": total}
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _request(self, method: str, path: str, params: Dict[str, Any], *, signed: bool) -> Any:
        payload = dict(params or {})
        if signed:
            payload = self._signed_params(payload)

        operation = f"{method.upper()} {path}"
        start = self._time()
        try:
            if method.upper() in {"GET", "DELETE"}:
                response = self._client.request(method, path, params=payload)
            else:
                response = self._client.request(method, path, data=payload)
        except httpx.HTTPError as exc:  # pragma: no cover - network failures mocked in tests
            OBSERVABILITY.record_broker_latency(
                broker="binance",
                operation=operation,
                latency=self._time() - start,
                status="http_error",
            )
            raise BrokerGatewayError(f"HTTP error calling Binance: {exc}") from exc

        latency = self._time() - start
        if response.status_code >= 400:
            try:
                detail = response.json()
            except Exception:
                detail = response.text
            OBSERVABILITY.record_broker_latency(
                broker="binance",
                operation=operation,
                latency=latency,
                status=f"error_{response.status_code}",
            )
            raise BrokerGatewayError(f"Binance error {response.status_code}: {detail}")
        try:
            data = response.json()
        except ValueError as exc:  # pragma: no cover - defensive
            OBSERVABILITY.record_broker_latency(
                broker="binance",
                operation=operation,
                latency=latency,
                status="decode_error",
            )
            raise BrokerGatewayError("Failed to decode Binance response") from exc
        OBSERVABILITY.record_broker_latency(
            broker="binance",
            operation=operation,
            latency=latency,
            status="success",
        )
        return data

    def _signed_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        payload = {key: self._normalise_param(value) for key, value in params.items()}
        payload["timestamp"] = str(int(self._time() * 1000))
        payload["recvWindow"] = str(self._recv_window)
        query = urlencode(payload, doseq=True)
        signature = hmac.new(self._api_secret, query.encode("utf-8"), hashlib.sha256).hexdigest()
        payload["signature"] = signature
        return payload

    @staticmethod
    def _normalise_param(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.16f}".rstrip("0").rstrip(".") if value % 1 else str(int(value))
        return str(value)

    @staticmethod
    def _extract_average_price(data: Dict[str, Any]) -> Optional[float]:
        executed = float(data.get("executedQty", 0.0))
        if executed <= 0:
            return None
        cumulative = data.get("cummulativeQuoteQty") or data.get("cumulativeQuoteQty")
        if cumulative is not None:
            try:
                cumulative_val = float(cumulative)
                if cumulative_val > 0:
                    return cumulative_val / executed
            except (TypeError, ValueError):
                pass
        price = data.get("price")
        if price is not None:
            try:
                price_val = float(price)
                if price_val > 0:
                    return price_val
            except (TypeError, ValueError):
                pass
        fills = data.get("fills") or []
        total_quote = 0.0
        total_qty = 0.0
        for fill in fills:
            try:
                qty = float(fill.get("qty", 0.0))
                price_val = float(fill.get("price", 0.0))
            except (TypeError, ValueError):
                continue
            total_qty += qty
            total_quote += qty * price_val
        if total_qty > 0 and total_quote > 0:
            return total_quote / total_qty
        return None


__all__ = [
    "OrderRequest",
    "OrderResponse",
    "BrokerGateway",
    "BrokerGatewayError",
    "SimulatedBrokerGateway",
    "BinanceBrokerGateway",
]
