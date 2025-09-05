from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class OrderResult:
    id: str
    symbol: str
    side: str          # "buy" / "sell"
    amount: float
    price: Optional[float]
    status: str        # "open" | "closed" | "canceled"
    raw: Dict[str, Any]


@dataclass
class Position:
    symbol: str
    base: str
    quote: str
    free_base: float
    free_quote: float
    total_base: float
    total_quote_value: Optional[float] = None  # по рынку (если известно)


class Executor(abc.ABC):
    """Интерфейс исполнителя сделок (реальный, симулятор, UI-бот)."""

    name: str = "base"

    def __init__(self, *, testnet: bool = False, config: Optional[Dict[str, Any]] = None):
        self.testnet = bool(testnet)
        self.config = config or {}

    # ---- ордера ----
    @abc.abstractmethod
    async def open_market(
        self,
        *,
        symbol: str,
        side: str,          # "buy"/"sell"
        amount: float,      # количество в базовой валюте (BTC в BTCUSDT)
        sl_pct: Optional[float] = None,
        tp_pct: Optional[float] = None,
        client_tag: Optional[str] = None,
    ) -> OrderResult:
        ...

    @abc.abstractmethod
    async def close_all(self, *, symbol: str) -> List[OrderResult]:
        """Закрыть позицию (для spot — продать базу)."""
        ...

    # ---- контекст / состояние ----
    @abc.abstractmethod
    async def get_positions(self, *, symbols: Optional[List[str]] = None) -> List[Position]:
        ...

    @abc.abstractmethod
    async def close(self) -> None:
        ...
