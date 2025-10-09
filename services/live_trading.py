"""High level orchestration for live trade execution.

The coordinator introduced here bridges the analytical stack
(`decide_and_execute`) with a concrete :mod:`services.broker_gateway`
implementation.  It is intentionally conservative: orders are only sent
when risk constraints allow it, and the returned payload mirrors the
decision together with the broker acknowledgement.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, Iterable, Optional

from services.broker_gateway import BrokerGateway, BrokerGatewayError, OrderRequest, OrderResponse
from services.trading_service import decide_and_execute, evaluate_pre_trade_controls

LOG = logging.getLogger("ai_trader.live_trading")


class LiveTradingCoordinator:
    """Routes signals into a broker gateway once risk checks pass."""

    def __init__(
        self,
        router,
        gateway: BrokerGateway,
        *,
        min_quantity: float = 1e-6,
        quantity_rounding: Optional[int] = 6,
        status_poll_attempts: int = 1,
        order_retry_attempts: int = 1,
    ) -> None:
        self._router = router
        self._gateway = gateway
        self._min_quantity = float(min_quantity)
        self._quantity_rounding = quantity_rounding
        self._status_poll_attempts = max(0, int(status_poll_attempts))
        self._order_retry_attempts = max(0, int(order_retry_attempts))
        self._orders: Dict[str, OrderResponse] = {}
        self._order_symbols: Dict[str, str] = {}
        self._terminal_statuses = {"filled", "canceled", "cancelled", "rejected", "expired"}

    def route_and_execute(
        self,
        symbol: str,
        features: Dict[str, Any],
        *,
        news_text: Optional[str] = None,
        order_type: str = "market",
    ) -> Dict[str, Any]:
        decision = decide_and_execute(symbol, features, news_text, router=self._router)

        summary: Dict[str, Any] = {
            "decision": decision,
            "order": None,
            "executed": False,
            "error": None,
            "request_id": None,
            "retries": 0,
            "precheck": None,
        }

        if decision["trading_blocked"]:
            LOG.warning("Trading blocked by risk limits for %s", symbol)
            return summary

        risk_fraction = float(decision["risk_fraction"])
        if risk_fraction <= 0:
            LOG.info("No trade for %s (risk fraction=%.6f)", symbol, risk_fraction)
            return summary

        price = float(features.get("price") or 0.0)
        equity = float(decision.get("limits", {}).get("equity") or features.get("equity") or 0.0)
        if price <= 0 or equity <= 0:
            summary["error"] = "missing_price_or_equity"
            LOG.error("Cannot compute order size for %s (price=%s, equity=%s)", symbol, price, equity)
            return summary

        notional = equity * risk_fraction
        quantity = notional / price
        if self._quantity_rounding is not None:
            quantity = round(quantity, self._quantity_rounding)
        if quantity < self._min_quantity:
            summary["error"] = "quantity_below_minimum"
            LOG.warning("Computed quantity %.8f below minimum for %s", quantity, symbol)
            return summary

        side = decision["signal"].get("signal")
        if side not in {"buy", "sell"}:
            LOG.info("Signal %s does not translate into executable side", side)
            return summary

        precheck = evaluate_pre_trade_controls(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            decision=decision,
            features=features,
            order_type=order_type,
        )
        summary["precheck"] = precheck
        if not precheck["ok"]:
            summary["error"] = "pre_trade_checks_failed"
            summary["error_details"] = precheck["violations"]
            LOG.warning(
                "Pre-trade controls blocked order for %s: %s",
                symbol,
                ", ".join(v["code"] for v in precheck["violations"]),
            )
            return summary

        client_order_id = uuid.uuid4().hex
        request = OrderRequest(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            metadata={
                "reference_price": price,
                "risk_fraction": risk_fraction,
                "client_order_id": client_order_id,
            },
        )

        response: Optional[OrderResponse] = None
        attempts = 0
        while attempts <= self._order_retry_attempts:
            try:
                response = self._gateway.submit_order(request)
                break
            except BrokerGatewayError as exc:
                if attempts >= self._order_retry_attempts:
                    summary["error"] = str(exc)
                    LOG.exception("Broker rejected order for %s after retries: %s", symbol, exc)
                    return summary
                attempts += 1
                summary["retries"] = attempts
                LOG.warning(
                    "Retrying order for %s due to broker error (%s/%s): %s",
                    symbol,
                    attempts,
                    self._order_retry_attempts,
                    exc,
                )
                continue
            except Exception as exc:  # pragma: no cover - defensive guardrail
                summary["error"] = str(exc)
                LOG.exception("Unexpected broker error for %s", symbol)
                return summary

        if response is None:  # pragma: no cover - defensive guardrail
            summary["error"] = "unknown_broker_response"
            LOG.error("Gateway returned no response for %s", symbol)
            return summary

        summary["retries"] = attempts
        summary["request_id"] = response.request_id
        self._store_order_response(response, symbol)

        refreshed_response = self._maybe_poll_status(response, symbol)
        final_response = refreshed_response or response
        summary["order"] = final_response.to_dict()
        summary["executed"] = final_response.filled_quantity > 0 or final_response.status in {
            "filled",
            "partially_filled",
            "partial",
        }
        if summary["executed"]:
            LOG.info(
                "Executed %s order for %s (qty=%.6f, status=%s, price=%s)",
                side,
                symbol,
                final_response.filled_quantity,
                final_response.status,
                final_response.average_price,
            )
        else:
            LOG.info("Order for %s acknowledged with status %s", symbol, final_response.status)

        return summary

    def describe(self) -> Dict[str, Any]:
        """Export coordinator configuration for diagnostics endpoints."""

        try:
            risk_cfg = dict(self._router.risk_config)  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover - router without risk_config
            risk_cfg = {}
        return {
            "gateway": type(self._gateway).__name__,
            "min_quantity": self._min_quantity,
            "quantity_rounding": self._quantity_rounding,
            "risk": risk_cfg,
        }

    # ------------------------------------------------------------------
    # Order/state management helpers exposed to routers
    # ------------------------------------------------------------------
    def list_orders(self, *, refresh: bool = False, symbol: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        if refresh:
            self.refresh_open_orders(symbol=symbol)
        return [
            order.to_dict()
            for order in self._orders.values()
            if symbol is None or self._order_symbols.get(order.request_id) == symbol.upper()
        ]

    def get_order_status(
        self,
        request_id: str,
        *,
        symbol: Optional[str] = None,
        refresh: bool = True,
    ) -> Optional[Dict[str, Any]]:
        if refresh:
            symbol = symbol or self._order_symbols.get(request_id)
            try:
                response = self._gateway.get_order_status(request_id, symbol=symbol)
            except NotImplementedError:  # pragma: no cover - simulator falls back to cache
                response = None
            except ValueError as exc:
                LOG.warning("Cannot refresh order %s without symbol: %s", request_id, exc)
                response = None
            except BrokerGatewayError as exc:
                LOG.exception("Failed to refresh order %s", request_id)
                raise
            else:
                if response is not None:
                    self._store_order_response(response, symbol)
        cached = self._orders.get(request_id)
        return cached.to_dict() if cached else None

    def refresh_open_orders(self, *, symbol: Optional[str] = None) -> Iterable[Dict[str, Any]]:
        try:
            open_orders = list(self._gateway.list_open_orders(symbol))
        except BrokerGatewayError as exc:
            LOG.exception("Failed to list open orders")
            raise
        for order in open_orders:
            self._store_order_response(order, symbol or order.raw.get("symbol"))
        return [order.to_dict() for order in open_orders]

    def cancel_order(self, request_id: str, *, symbol: Optional[str] = None) -> Dict[str, Any]:
        try:
            cancelled = self._gateway.cancel_order(request_id)
        except BrokerGatewayError as exc:
            LOG.exception("Failed to cancel order %s", request_id)
            raise
        status: Optional[Dict[str, Any]] = None
        if cancelled:
            status_dict = self.get_order_status(request_id, symbol=symbol, refresh=True)
            if status_dict is None and request_id in self._orders:
                existing = self._orders[request_id]
                updated = OrderResponse(
                    request_id=existing.request_id,
                    status="cancelled",
                    filled_quantity=existing.filled_quantity,
                    average_price=existing.average_price,
                    submitted_at=existing.submitted_at,
                    raw=existing.raw,
                )
                self._store_order_response(updated, symbol)
                status_dict = updated.to_dict()
            status = status_dict
        return {"request_id": request_id, "cancelled": bool(cancelled), "order": status}

    def sync_account(self) -> Dict[str, Any]:
        try:
            balances = self._gateway.balances()
            positions = self._gateway.positions()
        except BrokerGatewayError as exc:
            LOG.exception("Failed to sync account state")
            raise
        return {"balances": balances, "positions": positions}

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------
    def _store_order_response(self, response: OrderResponse, symbol: Optional[str]) -> None:
        self._orders[response.request_id] = response
        if symbol:
            self._order_symbols[response.request_id] = symbol.upper()
        elif isinstance(response.raw, dict) and response.raw.get("symbol"):
            self._order_symbols[response.request_id] = str(response.raw["symbol"]).upper()

    def _maybe_poll_status(self, response: OrderResponse, symbol: str) -> Optional[OrderResponse]:
        if self._status_poll_attempts <= 0:
            return None
        current = response
        for attempt in range(self._status_poll_attempts):
            if current.status in self._terminal_statuses and current.filled_quantity >= response.filled_quantity:
                break
            try:
                refreshed = self._gateway.get_order_status(response.request_id, symbol=symbol)
            except NotImplementedError:  # pragma: no cover - simulator
                break
            except BrokerGatewayError as exc:
                LOG.warning("Status poll %s/%s failed for %s: %s", attempt + 1, self._status_poll_attempts, symbol, exc)
                break
            if refreshed is None:
                break
            self._store_order_response(refreshed, symbol)
            current = refreshed
            if current.status in self._terminal_statuses:
                break
        return current if current is not response else None


__all__ = ["LiveTradingCoordinator"]

