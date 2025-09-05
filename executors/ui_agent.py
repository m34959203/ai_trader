from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import Executor, OrderResult, Position


class UIExecutorStub(Executor):
    """
    Заглушка для экранного агента (кликер + OCR).
    Реальная реализация будет работать с оконным приложением/браузером.
    """

    name = "ui"

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
        # здесь будет: поиск поля ввода -> ввод количества -> клик по кнопке Buy/Sell -> валидация
        return OrderResult(
            id="ui-stub",
            symbol=symbol,
            side=side,
            amount=amount,
            price=None,
            status="unknown",
            raw={"info": "UI agent stub - no real trade executed"},
        )

    async def close_all(self, *, symbol: str) -> List[OrderResult]:
        return []

    async def get_positions(self, *, symbols: Optional[List[str]] = None) -> List[Position]:
        return []

    async def close(self) -> None:
        return None
