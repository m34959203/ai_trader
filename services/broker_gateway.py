"""Broker gateway abstractions for live order execution.

This module intentionally keeps runtime dependencies minimal so that the
production integration layer can be implemented without blocking the
existing paper-trading stack.  The concrete implementation introduced
here is an in-memory simulator that emulates a broker connection; real
gateways can subclass :class:`BaseBrokerGateway` and reuse the request
/ response dataclasses defined below.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Protocol

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

    def positions(self) -> MutableMapping[str, float]:
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
            if order.request_id == request_id and order.status not in {"cancelled", "filled"}:
                self._orders[idx] = OrderResponse(
                    request_id=order.request_id,
                    status="cancelled",
                    filled_quantity=order.filled_quantity,
                    average_price=order.average_price,
                    submitted_at=order.submitted_at,
                    raw=order.raw,
                )
                return True
        return False

    def list_open_orders(self, symbol: Optional[str] = None) -> Iterable[OrderResponse]:
        return [
            order
            for order in self._orders
            if order.status not in {"filled", "cancelled"}
            and (symbol is None or order.raw.get("request", {}).get("symbol") == symbol)
        ]

    def positions(self) -> MutableMapping[str, float]:
        return dict(self._positions)

    def _update_position(self, symbol: str, side: OrderSide, quantity: float) -> None:
        delta = quantity if side.lower() == "buy" else -quantity
        self._positions[symbol] = self._positions.get(symbol, 0.0) + delta

    @staticmethod
    def _resolve_fill_price(request: OrderRequest) -> Optional[float]:
        if request.order_type == "market":
            return request.metadata.get("reference_price")
        return request.limit_price

