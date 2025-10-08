from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import Executor, OrderResult, Position

LOG = logging.getLogger("ai_trader.executors.ui")


class UIAutomationError(RuntimeError):
    pass


@dataclass(slots=True)
class UIAutomationBackend:
    """Stateful backend shared between DOM and OCR flows."""

    dom_available: bool = True
    ocr_available: bool = True
    latency_ms: int = 150
    positions: Dict[str, float] = field(default_factory=dict)
    _order_seq: int = 0

    def _next_order_id(self, source: str) -> str:
        self._order_seq += 1
        return f"ui-{source}-{self._order_seq}"

    def place_order(self, symbol: str, side: str, amount: float, source: str) -> OrderResult:
        qty = float(amount)
        sym = symbol.upper()
        side_l = side.lower()
        if side_l not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        pos = self.positions.get(sym, 0.0)
        pos = pos + qty if side_l == "buy" else pos - qty
        self.positions[sym] = pos
        order_id = self._next_order_id(source)
        status = "submitted" if source == "dom" else "manual"
        LOG.info("UI backend recorded order %s %s %.8f via %s", sym, side_l, qty, source)
        return OrderResult(
            id=order_id,
            symbol=sym,
            side=side_l,
            amount=qty,
            price=None,
            status=status,
            raw={"source": source, "timestamp": int(time.time() * 1000)},
        )

    def close_symbol(self, symbol: str) -> OrderResult:
        sym = symbol.upper()
        qty = self.positions.get(sym, 0.0)
        if abs(qty) < 1e-9:
            return OrderResult(
                id=self._next_order_id("close"),
                symbol=sym,
                side="flat",
                amount=0.0,
                price=None,
                status="noop",
                raw={"info": "position already flat"},
            )
        side = "sell" if qty > 0 else "buy"
        res = self.place_order(sym, side, abs(qty), "close")
        self.positions[sym] = 0.0
        res.raw["closed_qty"] = qty
        return res

    def snapshot_positions(self) -> List[Position]:
        out: List[Position] = []
        for sym, qty in self.positions.items():
            base = sym[:-4] if sym.endswith("USDT") else sym
            out.append(
                Position(
                    symbol=sym,
                    base=base,
                    quote=sym[len(base):] if len(sym) > len(base) else "USDT",
                    free_base=qty,
                    free_quote=0.0,
                    total_base=qty,
                    total_quote_value=None,
                )
            )
        return out


class DOMAutomationClient:
    def __init__(self, backend: UIAutomationBackend):
        self._backend = backend

    async def place_market_order(self, symbol: str, side: str, amount: float) -> OrderResult:
        await asyncio.sleep(self._backend.latency_ms / 1000)
        if not self._backend.dom_available:
            raise UIAutomationError("DOM not reachable")
        return self._backend.place_order(symbol, side, amount, "dom")


class OCRAutomationClient:
    def __init__(self, backend: UIAutomationBackend):
        self._backend = backend

    async def place_market_order(self, symbol: str, side: str, amount: float) -> OrderResult:
        await asyncio.sleep(self._backend.latency_ms / 500)
        if not self._backend.ocr_available:
            raise UIAutomationError("OCR pipeline offline")
        return self._backend.place_order(symbol, side, amount, "ocr")

    async def fetch_positions(self) -> List[Position]:
        return self._backend.snapshot_positions()


class UIExecutorAgent(Executor):
    """UI automation executor with DOM and OCR fallbacks."""

    name = "ui"

    def __init__(self, *, testnet: bool = False, config: Optional[Dict[str, Any]] = None):
        super().__init__(testnet=testnet, config=config)
        backend_cfg = config or {}
        self._backend = UIAutomationBackend(
            dom_available=backend_cfg.get("dom_available", True),
            ocr_available=backend_cfg.get("ocr_available", True),
            latency_ms=int(backend_cfg.get("latency_ms", 150)),
        )
        self._dom = DOMAutomationClient(self._backend)
        self._ocr = OCRAutomationClient(self._backend)
        self._failovers = 0
        self._lock = asyncio.Lock()

    async def open_market(
        self,
        *,
        symbol: str,
        side: str,
        amount: float,
        sl_pct: Optional[float] = None,
        tp_pct: Optional[float] = None,
        client_tag: Optional[str] = None,
    ) -> OrderResult:
        async with self._lock:
            try:
                result = await self._dom.place_market_order(symbol, side, amount)
                result.raw.update({"failovers": self._failovers, "client_tag": client_tag})
                return result
            except UIAutomationError as dom_exc:
                self._failovers += 1
                LOG.warning("DOM automation failed (%s), fallback to OCR", dom_exc)
                fallback = await self._ocr.place_market_order(symbol, side, amount)
                fallback.raw.update({"failover_reason": str(dom_exc), "client_tag": client_tag})
                return fallback

    async def close_all(self, *, symbol: str) -> List[OrderResult]:
        async with self._lock:
            result = self._backend.close_symbol(symbol)
            return [result]

    async def get_positions(self, *, symbols: Optional[List[str]] = None) -> List[Position]:
        pos = await self._ocr.fetch_positions()
        if symbols is None:
            return pos
        wanted = {s.upper() for s in symbols}
        return [p for p in pos if p.symbol.upper() in wanted]

    async def close(self) -> None:
        LOG.info("UI executor agent closed (failovers=%d)", self._failovers)


class UIExecutorStub(UIExecutorAgent):
    """Backward compatible alias for legacy code paths."""

    pass

