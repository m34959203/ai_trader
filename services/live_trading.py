"""High level orchestration for live trade execution.

The coordinator introduced here bridges the analytical stack
(`decide_and_execute`) with a concrete :mod:`services.broker_gateway`
implementation.  It is intentionally conservative: orders are only sent
when risk constraints allow it, and the returned payload mirrors the
decision together with the broker acknowledgement.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from services.broker_gateway import BrokerGateway, BrokerGatewayError, OrderRequest
from services.trading_service import decide_and_execute

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
    ) -> None:
        self._router = router
        self._gateway = gateway
        self._min_quantity = float(min_quantity)
        self._quantity_rounding = quantity_rounding

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

        request = OrderRequest(
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            metadata={
                "reference_price": price,
                "risk_fraction": risk_fraction,
            },
        )

        try:
            response = self._gateway.submit_order(request)
        except BrokerGatewayError as exc:
            summary["error"] = str(exc)
            LOG.exception("Broker rejected order for %s: %s", symbol, exc)
            return summary
        except Exception as exc:  # pragma: no cover - defensive guardrail
            summary["error"] = str(exc)
            LOG.exception("Unexpected broker error for %s", symbol)
            return summary

        summary["order"] = response.to_dict()
        summary["executed"] = response.status in {"filled", "partial"}
        if summary["executed"]:
            LOG.info(
                "Executed %s order for %s (qty=%.6f, price=%s)",
                side,
                symbol,
                response.filled_quantity,
                response.average_price,
            )
        else:
            LOG.info("Order for %s acknowledged with status %s", symbol, response.status)

        return summary


__all__ = ["LiveTradingCoordinator"]

