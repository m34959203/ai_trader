"""High level orchestration for live trade execution.

The coordinator introduced here bridges the analytical stack
(`decide_and_execute`) with a concrete :mod:`services.broker_gateway`
implementation.  It is intentionally conservative: orders are only sent
when risk constraints allow it, and the returned payload mirrors the
decision together with the broker acknowledgement.
"""

from __future__ import annotations

import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from services.broker_gateway import BrokerGateway, BrokerGatewayError, OrderRequest, OrderResponse
from services.trading_service import decide_and_execute, evaluate_pre_trade_controls
from state.daily_limits import ensure_day, load_state
from utils.risk_config import load_risk_config
from utils.structured_logging import get_logger


LOG = get_logger("ai_trader.live_trading")


@dataclass
class StrategyControlState:
    enabled: bool = True
    max_risk_fraction: Optional[float] = None
    max_daily_trades: Optional[int] = None
    notes: Optional[str] = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self, name: str, trades_today: int) -> Dict[str, Any]:
        return {
            "name": name,
            "enabled": bool(self.enabled),
            "max_risk_fraction": None if self.max_risk_fraction is None else float(self.max_risk_fraction),
            "max_daily_trades": None if self.max_daily_trades is None else int(self.max_daily_trades),
            "notes": self.notes,
            "updated_at": self.updated_at,
            "trades_today": max(0, int(trades_today)),
        }


@dataclass
class StrategyUsage:
    day: date
    count: int = 0


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
        self._strategy_controls: Dict[str, StrategyControlState] = {}
        self._strategy_usage: Dict[str, StrategyUsage] = {}
        self._trade_log: Deque[Dict[str, Any]] = deque(maxlen=200)
        self._last_equity: Optional[float] = None
        self._last_broker_error: Optional[str] = None

    # ------------------------------------------------------------------
    # Internal helpers — strategy state & clocks
    # ------------------------------------------------------------------
    @staticmethod
    def _strategy_key(name: Optional[str]) -> str:
        if name is None:
            return "default"
        key = str(name).strip()
        return key or "default"

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    def _current_local_date(self) -> date:
        cfg = load_risk_config()
        tz = cfg.tz
        if tz is not None:
            return datetime.now(tz).date()
        return datetime.utcnow().date()

    def _ensure_strategy(self, name: Optional[str]) -> Tuple[str, StrategyControlState, StrategyUsage]:
        key = self._strategy_key(name)
        state = self._strategy_controls.get(key)
        if state is None:
            state = StrategyControlState()
            self._strategy_controls[key] = state
        usage = self._strategy_usage.get(key)
        today = self._current_local_date()
        if usage is None or usage.day != today:
            usage = StrategyUsage(day=today, count=0)
            self._strategy_usage[key] = usage
        return key, state, usage

    def update_strategy_control(
        self,
        name: str,
        *,
        enabled: Optional[bool] = None,
        max_risk_fraction: Optional[float] = None,
        max_daily_trades: Optional[int] = None,
        notes: Optional[str] = None,
        provided: Optional[Iterable[str]] = None,
    ) -> Dict[str, Any]:
        provided_set = {p for p in (provided or [])}
        key, state, usage = self._ensure_strategy(name)
        changed = False

        if "enabled" in provided_set and enabled is not None:
            state.enabled = bool(enabled)
            changed = True

        if "max_risk_fraction" in provided_set:
            state.max_risk_fraction = None if max_risk_fraction is None else max(0.0, float(max_risk_fraction))
            changed = True

        if "max_daily_trades" in provided_set:
            state.max_daily_trades = None if max_daily_trades is None else max(0, int(max_daily_trades))
            changed = True

        if "notes" in provided_set:
            state.notes = notes.strip() if isinstance(notes, str) else notes
            changed = True

        if changed:
            state.updated_at = self._utc_now()

        return state.to_dict(key, usage.count)

    def list_strategy_controls(self) -> List[Dict[str, Any]]:
        today = self._current_local_date()
        if not self._strategy_controls:
            self._ensure_strategy("default")
        out: List[Dict[str, Any]] = []
        for name, state in sorted(self._strategy_controls.items(), key=lambda item: item[0]):
            usage = self._strategy_usage.get(name)
            trades_today = 0 if usage is None or usage.day != today else usage.count
            out.append(state.to_dict(name, trades_today))
        return out

    def _check_strategy_limits(
        self,
        name: Optional[str],
    ) -> Tuple[str, StrategyControlState, StrategyUsage, Optional[str]]:
        key, state, usage = self._ensure_strategy(name)
        if not state.enabled:
            return key, state, usage, "strategy_disabled"
        if state.max_daily_trades is not None and usage.count >= state.max_daily_trades:
            return key, state, usage, "strategy_daily_limit"
        return key, state, usage, None

    def _increment_strategy_usage(self, key: str) -> None:
        usage = self._strategy_usage.get(key)
        today = self._current_local_date()
        if usage is None or usage.day != today:
            usage = StrategyUsage(day=today, count=0)
            self._strategy_usage[key] = usage
        usage.count += 1

    # ------------------------------------------------------------------
    # Public diagnostics helpers
    # ------------------------------------------------------------------
    def pnl_snapshot(self) -> Dict[str, Any]:
        cfg = load_risk_config()
        current_equity = self._last_equity
        state = None
        try:
            state = load_state(cfg.tz_name)
            if state is not None:
                state = ensure_day(state, cfg.tz_name, current_equity=current_equity or 0.0)
        except Exception:
            state = None

        start_equity = float(state.start_equity) if state else None  # type: ignore[attr-defined]
        realized_pnl = float(state.realized_pnl) if state else None  # type: ignore[attr-defined]
        trades_count = int(state.trades_count) if state else 0  # type: ignore[attr-defined]

        drawdown_pct = None
        if start_equity and current_equity is not None and start_equity > 0:
            drawdown_pct = max(0.0, (start_equity - current_equity) / start_equity)

        return {
            "ts": self._utc_now(),
            "start_equity": start_equity,
            "current_equity": current_equity,
            "realized_pnl": realized_pnl,
            "drawdown_pct": drawdown_pct,
            "trades_count": trades_count,
        }

    def limits_snapshot(self) -> Dict[str, Any]:
        cfg = load_risk_config().validate()
        pnl = self.pnl_snapshot()
        daily = {
            "trades_count": pnl["trades_count"],
            "start_equity": pnl["start_equity"],
            "current_equity": pnl["current_equity"],
            "realized_pnl": pnl["realized_pnl"],
            "drawdown_pct": pnl["drawdown_pct"],
        }
        risk_config = {
            "per_trade_cap": cfg.risk_pct_per_trade,
            "portfolio_max_risk_pct": cfg.portfolio_max_risk_pct,
            "max_open_positions": cfg.max_open_positions,
            "daily_max_loss_pct": cfg.daily_max_loss_pct,
            "max_trades_per_day": cfg.max_trades_per_day,
            "tz_name": cfg.tz_name,
        }
        return {
            "risk_config": risk_config,
            "daily": daily,
            "strategies": self.list_strategy_controls(),
        }

    def broker_status(self, *, include_orders: bool = True) -> Dict[str, Any]:
        open_orders: List[Dict[str, Any]] = []
        positions: Dict[str, float] = {}
        balances: Dict[str, Dict[str, float]] = {}
        connected = True
        last_error = self._last_broker_error

        if include_orders:
            try:
                for order in self._gateway.list_open_orders():
                    open_orders.append(order.to_dict())
            except Exception as exc:  # pragma: no cover - defensive guard
                connected = False
                last_error = str(exc)

        try:
            raw_positions = self._gateway.positions()
            if isinstance(raw_positions, dict):
                positions = {str(sym): float(qty) for sym, qty in raw_positions.items()}
        except Exception as exc:  # pragma: no cover - defensive guard
            connected = False
            last_error = str(exc)

        try:
            raw_balances = self._gateway.balances()
            if isinstance(raw_balances, dict):
                balances = {
                    str(asset): {k: float(v) for k, v in (info or {}).items()}
                    for asset, info in raw_balances.items()
                }
        except Exception as exc:  # pragma: no cover - defensive guard
            connected = False
            last_error = str(exc)

        return {
            "updated_at": self._utc_now(),
            "connected": connected,
            "gateway": type(self._gateway).__name__,
            "open_orders": open_orders,
            "positions": positions,
            "balances": balances,
            "last_error": last_error,
        }

    def list_trades(self, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), len(self._trade_log)))
        return list(list(self._trade_log)[:limit])

    def _record_trade(
        self,
        *,
        symbol: str,
        strategy: str,
        side: Optional[str],
        quantity: Optional[float],
        price: Optional[float],
        executed: bool,
        error: Optional[str],
        risk_fraction: Optional[float],
        notional: Optional[float],
        request_id: Optional[str],
        decision: Dict[str, Any],
        order: Optional[Dict[str, Any]],
    ) -> None:
        record = {
            "ts": self._utc_now(),
            "symbol": symbol,
            "strategy": strategy,
            "side": side,
            "quantity": None if quantity is None else float(quantity),
            "price": None if price is None else float(price),
            "executed": bool(executed),
            "error": error,
            "risk_fraction": None if risk_fraction is None else float(risk_fraction),
            "notional": None if notional is None else float(notional),
            "request_id": request_id,
            "status": order.get("status") if order else None,
            "filled_quantity": order.get("filled_quantity") if order else None,
            "average_price": order.get("average_price") if order else None,
            "raw": order.get("raw") if order else None,
        }
        signal = (decision or {}).get("signal") or {}
        try:
            record["confidence"] = float(signal.get("confidence")) if signal.get("confidence") is not None else None
        except Exception:
            record["confidence"] = None
        limits = (decision or {}).get("limits") or {}
        record["day_pnl"] = limits.get("day_pnl")
        record["equity"] = limits.get("equity")
        self._trade_log.appendleft(record)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def route_and_execute(
        self,
        symbol: str,
        features: Dict[str, Any],
        *,
        news_text: Optional[str] = None,
        order_type: str = "market",
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        strategy_name = strategy or features.get("strategy") or features.get("strategy_name")
        decision = decide_and_execute(symbol, features, news_text, router=self._router)

        strategy_key, control, usage, block_reason = self._check_strategy_limits(strategy_name)

        summary: Dict[str, Any] = {
            "decision": decision,
            "order": None,
            "executed": False,
            "error": None,
            "request_id": None,
            "retries": 0,
            "precheck": None,
            "strategy": strategy_key,
        }

        if decision["trading_blocked"]:
            summary["error"] = "risk_blocked"
            self._record_trade(
                symbol=symbol,
                strategy=strategy_key,
                side=(decision.get("signal") or {}).get("signal"),
                quantity=None,
                price=None,
                executed=False,
                error=summary["error"],
                risk_fraction=decision.get("risk_fraction"),
                notional=None,
                request_id=None,
                decision=decision,
                order=None,
            )
            LOG.warning("Trading blocked by risk limits for %s", symbol)
            return summary

        if block_reason is not None:
            summary["error"] = block_reason
            self._record_trade(
                symbol=symbol,
                strategy=strategy_key,
                side=(decision.get("signal") or {}).get("signal"),
                quantity=None,
                price=None,
                executed=False,
                error=block_reason,
                risk_fraction=decision.get("risk_fraction"),
                notional=None,
                request_id=None,
                decision=decision,
                order=None,
            )
            LOG.info("Strategy %s blocked trade for %s due to %s", strategy_key, symbol, block_reason)
            return summary

        risk_fraction = float(decision.get("risk_fraction", 0.0))
        if control.max_risk_fraction is not None:
            limited = min(risk_fraction, control.max_risk_fraction)
            decision.setdefault("limits", {})["strategy_risk_cap"] = control.max_risk_fraction
            decision["risk_fraction"] = limited
            risk_fraction = limited

        if risk_fraction <= 0:
            summary["error"] = "risk_fraction_zero"
            self._record_trade(
                symbol=symbol,
                strategy=strategy_key,
                side=(decision.get("signal") or {}).get("signal"),
                quantity=None,
                price=None,
                executed=False,
                error=summary["error"],
                risk_fraction=risk_fraction,
                notional=None,
                request_id=None,
                decision=decision,
                order=None,
            )
            LOG.info("No trade for %s (risk fraction limited to %.6f)", symbol, risk_fraction)
            return summary

        price = float(features.get("price") or 0.0)
        equity = float(decision.get("limits", {}).get("equity") or features.get("equity") or 0.0)
        if equity > 0:
            self._last_equity = equity
        if price <= 0 or equity <= 0:
            summary["error"] = "missing_price_or_equity"
            self._record_trade(
                symbol=symbol,
                strategy=strategy_key,
                side=(decision.get("signal") or {}).get("signal"),
                quantity=None,
                price=None,
                executed=False,
                error=summary["error"],
                risk_fraction=risk_fraction,
                notional=None,
                request_id=None,
                decision=decision,
                order=None,
            )
            LOG.error("Cannot compute order size for %s (price=%s, equity=%s)", symbol, price, equity)
            return summary

        notional = equity * risk_fraction
        quantity = notional / price if price > 0 else 0.0
        if self._quantity_rounding is not None:
            quantity = round(quantity, self._quantity_rounding)
        if quantity < self._min_quantity:
            summary["error"] = "quantity_below_minimum"
            self._record_trade(
                symbol=symbol,
                strategy=strategy_key,
                side=(decision.get("signal") or {}).get("signal"),
                quantity=quantity,
                price=price,
                executed=False,
                error=summary["error"],
                risk_fraction=risk_fraction,
                notional=notional,
                request_id=None,
                decision=decision,
                order=None,
            )
            LOG.warning("Computed quantity %.8f below minimum for %s", quantity, symbol)
            return summary

        side = decision.get("signal", {}).get("signal")
        if side not in {"buy", "sell"}:
            summary["error"] = "signal_not_executable"
            self._record_trade(
                symbol=symbol,
                strategy=strategy_key,
                side=side,
                quantity=quantity,
                price=price,
                executed=False,
                error=summary["error"],
                risk_fraction=risk_fraction,
                notional=notional,
                request_id=None,
                decision=decision,
                order=None,
            )
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
            self._record_trade(
                symbol=symbol,
                strategy=strategy_key,
                side=side,
                quantity=quantity,
                price=price,
                executed=False,
                error=summary["error"],
                risk_fraction=risk_fraction,
                notional=notional,
                request_id=None,
                decision=decision,
                order=None,
            )
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
                "strategy": strategy_key,
            },
        )

        response: Optional[OrderResponse] = None
        attempts = 0
        while attempts <= self._order_retry_attempts:
            try:
                response = self._gateway.submit_order(request)
                self._last_broker_error = None
                break
            except BrokerGatewayError as exc:
                self._last_broker_error = str(exc)
                if attempts >= self._order_retry_attempts:
                    summary["error"] = str(exc)
                    self._record_trade(
                        symbol=symbol,
                        strategy=strategy_key,
                        side=side,
                        quantity=quantity,
                        price=price,
                        executed=False,
                        error=summary["error"],
                        risk_fraction=risk_fraction,
                        notional=notional,
                        request_id=None,
                        decision=decision,
                        order=None,
                    )
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
                self._last_broker_error = str(exc)
                summary["error"] = str(exc)
                self._record_trade(
                    symbol=symbol,
                    strategy=strategy_key,
                    side=side,
                    quantity=quantity,
                    price=price,
                    executed=False,
                    error=summary["error"],
                    risk_fraction=risk_fraction,
                    notional=notional,
                    request_id=None,
                    decision=decision,
                    order=None,
                )
                LOG.exception("Unexpected broker error for %s", symbol)
                return summary

        if response is None:  # pragma: no cover - defensive guardrail
            summary["error"] = "unknown_broker_response"
            self._record_trade(
                symbol=symbol,
                strategy=strategy_key,
                side=side,
                quantity=quantity,
                price=price,
                executed=False,
                error=summary["error"],
                risk_fraction=risk_fraction,
                notional=notional,
                request_id=None,
                decision=decision,
                order=None,
            )
            LOG.error("Gateway returned no response for %s", symbol)
            return summary

        summary["retries"] = attempts
        summary["request_id"] = response.request_id
        self._store_order_response(response, symbol)

        refreshed_response = self._maybe_poll_status(response, symbol)
        final_response = refreshed_response or response
        order_dict = final_response.to_dict()
        summary["order"] = order_dict
        executed = final_response.filled_quantity > 0 or final_response.status in {
            "filled",
            "partially_filled",
            "partial",
        }
        summary["executed"] = executed

        self._record_trade(
            symbol=symbol,
            strategy=strategy_key,
            side=side,
            quantity=quantity,
            price=order_dict.get("average_price") or price,
            executed=executed,
            error=summary.get("error"),
            risk_fraction=risk_fraction,
            notional=notional,
            request_id=order_dict.get("request_id"),
            decision=decision,
            order=order_dict,
        )

        if executed:
            self._increment_strategy_usage(strategy_key)
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

