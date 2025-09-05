from __future__ import annotations

import os
import time
from typing import Optional, Dict, Any, List


class SimulatedExecutor:
    """
    Минималистичный спот-эмулятор под API, который ожидают routers/trading_exec.py и UI:
      • методы: fetch_balance, get_price, get_ticker_price, price (алиас),
                open_order, open_with_protection, close_order, close_all_positions,
                list_positions, round_qty, close
      • формат ответов совместим с реальным исполнителем
      • простейший учёт нетто-позиций и баланса (без комиссий)
      • поддержка *USDT символов (BTCUSDT и т.п.)
    """

    name = "sim"
    is_simulator = True

    def __init__(self, testnet: bool = True) -> None:
        self.testnet = bool(testnet)
        self.client = None  # для совместимости с проверками вида "if ex.client ..."
        # Базовая "эквити" в USDT можно поменять: set SIM_EQUITY_USDT
        equity_usdt = float(os.getenv("SIM_EQUITY_USDT", "100000"))
        px_btc = float(os.getenv("SIM_BTCUSDT_PRICE", "60000"))

        # Балансы (free/locked формат будет собран из этого словаря)
        self._balances: Dict[str, float] = {
            "USDT": equity_usdt,
            "BTC": 0.0,
        }

        # Нетто по символам (для UI/positions)
        self._positions: Dict[str, float] = {
            "BTCUSDT": 0.0,
        }

        # Котировки (можно расширять окружением позже)
        self._prices: Dict[str, float] = {
            "BTCUSDT": px_btc,
        }

    # ──────────────────────────────────────────────────────────────────────
    # Вспомогательные/совместимость
    # ──────────────────────────────────────────────────────────────────────
    async def close(self) -> None:
        """Совместимость с реальным исполнителем (ничего не делает)."""
        return None

    async def round_qty(self, symbol: str, qty: float) -> float:
        # На споте обычно 6-8 знаков; берём 6, чтобы не ловить отказы фильтров.
        return round(float(qty), 6)

    # Алиасы цен, чтобы код мог вызывать get_price / get_ticker_price / price
    async def get_price(self, symbol: str) -> float:
        symbol = str(symbol).upper().replace(" ", "")
        if symbol not in self._prices:
            # по умолчанию заведём 1.0 для детерминизма
            self._prices[symbol] = 1.0
        return float(self._prices[symbol])

    async def get_ticker_price(self, symbol: str) -> Dict[str, Any]:
        return {"symbol": symbol, "price": await self.get_price(symbol)}

    async def price(self, symbol: str) -> float:
        return await self.get_price(symbol)

    # ──────────────────────────────────────────────────────────────────────
    # Баланс / позиции
    # ──────────────────────────────────────────────────────────────────────
    async def fetch_balance(self) -> Dict[str, Any]:
        """Возвращаем баланс в структуре, с которой работает risk/exec/UI."""
        free = {k: float(v) for k, v in self._balances.items()}
        locked: Dict[str, float] = {}
        return {"exchange": "sim", "testnet": True, "free": free, "locked": locked}

    async def list_positions(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for sym, qty in self._positions.items():
            if abs(qty) > 1e-12:
                out.append({"symbol": sym, "qty": float(qty)})
        return out

    # ──────────────────────────────────────────────────────────────────────
    # Ордеры
    # ──────────────────────────────────────────────────────────────────────
    async def open_with_protection(
        self,
        *,
        symbol: str,
        side: str,
        qty: Optional[float],
        entry_type: str = "market",
        entry_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        quote_qty: Optional[float] = None,
        timeInForce: Optional[str] = None,
    ) -> Dict[str, Any]:
        res = await self.open_order(
            symbol=symbol,
            side=side,
            type=entry_type,
            qty=qty,
            price=entry_price,
            timeInForce=timeInForce,
            client_order_id=client_order_id,
            quote_qty=quote_qty,
        )
        res["protection"] = {"sl_price": sl_price, "tp_price": tp_price}
        res["protection_ids"] = None
        return res

    async def open_order(
        self,
        *,
        symbol: str,
        side: str,
        type: str = "market",
        qty: Optional[float] = None,
        price: Optional[float] = None,
        timeInForce: Optional[str] = None,
        client_order_id: Optional[str] = None,
        quote_qty: Optional[float] = None,
    ) -> Dict[str, Any]:
        symbol = str(symbol).upper().replace(" ", "")
        side = str(side).lower()
        typ = str(type).lower()

        # текущая/лимитная цена
        px = float(price) if (typ == "limit" and price is not None) else float(await self.get_price(symbol))

        # если передали quote_qty — сконвертим в базовую
        if (qty is None or float(qty) <= 0.0) and quote_qty is not None:
            qty = float(quote_qty) / px
        if qty is None or qty <= 0:
            raise ValueError("qty must be positive in simulator")

        qty = await self.round_qty(symbol, float(qty))

        # поддерживаем только *USDT кроссы (простая разметка)
        if not symbol.endswith("USDT"):
            raise ValueError("sim supports *USDT symbols only in this minimal build")

        base, quote = symbol[:-4], symbol[-4:]  # BTC / USDT

        # Спот-логика без комиссий
        if side == "buy":
            cost = qty * px
            if self._balances.get(quote, 0.0) < cost:
                raise ValueError(f"Insufficient {quote} in simulator")
            self._balances[quote] -= cost
            self._balances[base] = self._balances.get(base, 0.0) + qty
            self._positions[symbol] = self._positions.get(symbol, 0.0) + qty
        elif side == "sell":
            if self._balances.get(base, 0.0) < qty:
                raise ValueError(f"Insufficient {base} in simulator")
            proceeds = qty * px
            self._balances[base] -= qty
            self._balances[quote] = self._balances.get(quote, 0.0) + proceeds
            self._positions[symbol] = self._positions.get(symbol, 0.0) - qty
        else:
            raise ValueError("side must be 'buy' or 'sell'")

        order_id = f"sim-{int(time.time() * 1000)}"
        return {
            "exchange": "sim",
            "testnet": True,
            "order_id": order_id,
            "client_order_id": client_order_id,
            "symbol": symbol,
            "side": side,
            "type": typ,
            "price": px,
            "qty": qty,
            "status": "FILLED",
            "raw": {
                "mock": True,
                "orderId": order_id,
                "clientOrderId": client_order_id,
                "transactTime": int(time.time() * 1000),
            },
        }

    async def close_order(
        self,
        *,
        symbol: str,
        qty: Optional[float] = None,
        type: str = "market",
        price: Optional[float] = None,
        timeInForce: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        symbol = str(symbol).upper().replace(" ", "")
        typ = str(type).lower()
        px = float(price) if (typ == "limit" and price is not None) else float(await self.get_price(symbol))

        if not symbol.endswith("USDT"):
            raise ValueError("sim supports *USDT symbols only in this minimal build")

        base, quote = symbol[:-4], symbol[-4:]
        pos = float(self._positions.get(symbol, 0.0))

        if abs(pos) < 1e-12 and (qty is None or qty <= 0):
            return {"exchange": "sim", "testnet": True, "status": "NOTHING_TO_CLOSE", "symbol": symbol}

        if qty is None or qty <= 0:
            qty = abs(pos)

        qty = await self.round_qty(symbol, float(qty))

        if pos > 0:
            # закрываем лонг -> продаём base
            qty = min(qty, self._balances.get(base, 0.0))
            self._balances[base] -= qty
            self._balances[quote] = self._balances.get(quote, 0.0) + qty * px
            self._positions[symbol] = self._positions.get(symbol, 0.0) - qty
        elif pos < 0:
            # шорты в минимальном эмуляторе не поддерживаем
            pass

        order_id = f"sim-close-{int(time.time() * 1000)}"
        return {
            "exchange": "sim",
            "testnet": True,
            "order_id": order_id,
            "client_order_id": client_order_id,
            "symbol": symbol,
            "side": "sell",
            "type": typ,
            "price": px,
            "qty": qty,
            "status": "FILLED",
            "raw": {"mock": True, "orderId": order_id, "clientOrderId": client_order_id},
        }

    async def close_all_positions(self) -> None:
        """Используется deadman-switch: закрыть всё по рынку."""
        for symbol, qty in list(self._positions.items()):
            if abs(qty) > 1e-12:
                await self.close_order(symbol=symbol, qty=abs(qty), type="market")

    # Доп. утилита для тестов
    async def set_price(self, symbol: str, price: float) -> None:
        symbol = str(symbol).upper().replace(" ", "")
        self._prices[symbol] = float(price)
