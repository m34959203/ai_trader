# risk/trailing.py
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from decimal import Decimal, getcontext
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple

import pandas as pd

# высокая точность при вычислениях шагов и цен
getcontext().prec = 40

from executors.api_binance import BinanceExecutor  # тип исполнителя
from utils.risk_config import load_risk_config

LOG = logging.getLogger("ai_trader.trailing")

Side = Literal["buy", "sell"]


# ──────────────────────────────────────────────────────────────────────────────
# Конфиг
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrailingRuleConfig:
    """
    Настройки трейлинга.

    mode:
      • "percent" — стоп = price * (1 -/+ trail_pct)
      • "atr"     — стоп = price -/+ (atr_mult * ATR(period))  (требует ohlcv_fn)
    """
    mode: Literal["percent", "atr"] = "percent"

    # percent mode
    trail_pct: float = 0.01          # 1% подтяжка

    # atr mode
    atr_period: int = 14
    atr_mult: float = 2.0

    # общие
    poll_interval_sec: float = 8.0   # период опроса
    min_abs_move: float = 0.0        # минимальное абсолютное смещение для перестановки
    min_rel_move: float = 0.001      # 0.1% относительно предыдущего стопа — фильтр шума
    time_in_force: str = "GTC"       # для STOP_LOSS_LIMIT

    # Работа с OCO: по умолчанию НЕ трогаем и не отменяем OCO, чтобы не нарушать связки
    allow_cancel_oco: bool = False   # True → при наличии OCO попробуем отменить весь список (см. код)

    # защита от дерганья
    cooldown_after_update_sec: float = 2.0  # короткий кулдаун после успешной перестановки


# ──────────────────────────────────────────────────────────────────────────────
# Утилиты: индикаторы и округления
# ──────────────────────────────────────────────────────────────────────────────

def _atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    ATR по классике (True Range c EMA).
    Требуются колонки: high, low, close (регистронезависимо).
    """
    if df is None or df.empty or period <= 1:
        return 0.0

    # приведём имена
    cols = {c.lower(): c for c in df.columns}
    req = ["high", "low", "close"]
    if not all(k in cols for k in req):
        return 0.0

    h = pd.to_numeric(df[cols["high"]], errors="coerce")
    l = pd.to_numeric(df[cols["low"]], errors="coerce")
    c = pd.to_numeric(df[cols["close"]], errors="coerce")
    pc = c.shift(1)

    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    # EMA с min_periods для устойчивости
    atr = tr.ewm(alpha=1 / float(period), adjust=False, min_periods=period).mean().iloc[-1]
    try:
        return float(atr)
    except Exception:
        return 0.0


def _to_dec(x: Any) -> Decimal:
    return Decimal(str(x))


def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value // step) * step


# ──────────────────────────────────────────────────────────────────────────────
# Адаптер под Binance spot (используем низкоуровневый клиент из executor)
# ──────────────────────────────────────────────────────────────────────────────

class _BinanceAdapter:
    """
    Мини-адаптер для операций со стоп-ордерами на споте Binance.

    Мы используем внутренний client из BinanceExecutor (_BinanceSpotClient):
      • GET    /api/v3/openOrders         → список активных ордеров
      • DELETE /api/v3/order              → отмена ордера
      • DELETE /api/v3/orderList          → отмена OCO списка (если allow_cancel_oco)
      • POST   /api/v3/order (STOP_LOSS_LIMIT) → постановка стопа
      • GET    /api/v3/exchangeInfo       → фильтры (tick/step/minNotional)
      • GET    /api/v3/ticker/price       → цена
    """
    def __init__(self, ex: BinanceExecutor):
        self.ex = ex
        self.client = ex.client  # _BinanceSpotClient

    # ---------- market data ----------
    async def get_price(self, symbol: str) -> Optional[float]:
        p = await self.client.get_ticker_price(symbol)
        return float(p) if p is not None else None

    async def get_filters(self, symbol: str) -> Dict[str, Decimal]:
        info = await self.client.get_symbol_info(symbol)
        tick = step = min_qty = min_notional = None
        if info:
            for f in info.get("filters", []):
                t = f.get("filterType")
                if t == "PRICE_FILTER":
                    tick = _to_dec(f.get("tickSize", "0.00000001"))
                elif t == "LOT_SIZE":
                    step = _to_dec(f.get("stepSize", "0.00000001"))
                    min_qty = _to_dec(f.get("minQty", "0.0"))
                elif t in ("MIN_NOTIONAL", "NOTIONAL"):
                    min_notional = _to_dec(f.get("minNotional", "0"))
        return {
            "tick": tick or _to_dec("0.00000001"),
            "step": step or _to_dec("0.00000001"),
            "min_qty": min_qty or _to_dec("0.0"),
            "min_notional": min_notional or _to_dec("0.0"),
        }

    # ---------- order mgmt ----------
    async def list_open_orders(self, symbol: str) -> List[Dict[str, Any]]:
        # используем устойчивый метод клиента с ретраями
        data = await self.client._request_with_retries(  # type: ignore[attr-defined]
            "GET", "/api/v3/openOrders", params={"symbol": symbol.upper()}, sign=True
        )
        return list(data) if isinstance(data, list) else []

    async def cancel_order(
        self,
        *,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self.client.delete_order(symbol=symbol, orderId=order_id, origClientOrderId=client_order_id)

    async def cancel_oco(self, *, order_list_id: Optional[int] = None, list_client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Отмена OCO по списку. Работает только если у нас есть orderListId или listClientOrderId.
        """
        params: Dict[str, Any] = {}
        if order_list_id is not None:
            params["orderListId"] = int(order_list_id)
        if list_client_order_id:
            params["listClientOrderId"] = list_client_order_id
        if not params:
            return {"status": "NOOP"}
        # спецификация Binance: DELETE /api/v3/orderList
        return await self.client._request_with_retries(  # type: ignore[attr-defined]
            "DELETE", "/api/v3/orderList", params=params, sign=True
        )

    async def place_stop_loss_limit(
        self,
        *,
        symbol: str,
        side: Side,            # сторона закрытия: для long → SELL, для short → BUY
        qty: float,
        stop_price: float,     # trigger
        price: Optional[float] = None,  # limit price; если не передан — используем stop_price
        tif: str = "GTC",
    ) -> Dict[str, Any]:
        side_u = "SELL" if side == "buy" else "BUY"  # противоположная стороне входа
        price_u = float(price if price is not None else stop_price)
        return await self.client.post_order(
            symbol=symbol,
            side=side_u,
            type="STOP_LOSS_LIMIT",
            quantity=qty,
            price=price_u,
            stopPrice=float(stop_price),
            timeInForce=tif,
            newOrderRespType="RESULT",
        )

    # --- helpers for rounding/notional ---
    async def coerce_qty_price(self, symbol: str, qty: float, price: float) -> Tuple[float, float]:
        """
        Приведение qty/price к сетке фильтров; соблюдение minQty и minNotional.
        """
        f = await self.get_filters(symbol)
        d_qty = _to_dec(qty)
        d_price = _to_dec(price)
        # округление к сетке
        d_price = _floor_to_step(d_price, f["tick"])
        d_qty = _floor_to_step(d_qty, f["step"])
        # min qty
        if d_qty < f["min_qty"]:
            d_qty = f["min_qty"]
        # min notional
        notion = d_qty * d_price
        if f["min_notional"] > 0 and notion < f["min_notional"] and d_price > 0:
            need_qty = _floor_to_step(f["min_notional"] / d_price, f["step"])
            if need_qty > d_qty:
                d_qty = need_qty
        return float(d_qty), float(d_price)


# ──────────────────────────────────────────────────────────────────────────────
# Trailing Manager
# ──────────────────────────────────────────────────────────────────────────────

FetchOHLCV = Callable[[str, int], Awaitable[pd.DataFrame]]

class TrailingManager:
    """
    Фоновый менеджер трейлинга стопа для ОДНОЙ позиции (символ/сторона/кол-во).
    Поддерживаются режимы: percent (по умолчанию) и atr (через ohlcv_fn).
    """

    def __init__(
        self,
        executor: BinanceExecutor,
        *,
        symbol: str,
        side: Side,             # входная сторона (long=buy, short=sell)
        qty: float,
        config: Optional[TrailingRuleConfig] = None,
        ohlcv_fn: Optional[FetchOHLCV] = None,  # async fn(symbol, limit) -> DataFrame[open,high,low,close,volume], UTC index
    ):
        self.ex = executor
        self.adapter = _BinanceAdapter(executor)
        self.symbol = symbol.upper().replace(" ", "")
        self.side = side
        self.qty = float(qty)
        self.cfg = config or TrailingRuleConfig()
        self.ohlcv_fn = ohlcv_fn

        # состояние
        self._last_set_sl: Optional[float] = None
        self._last_action_ts: float = 0.0
        self._running_lock = asyncio.Lock()

    # ---------- вычисление желаемого SL ----------
    async def _desired_sl(self, price: float) -> Optional[float]:
        if price is None or price <= 0:
            return None

        if self.cfg.mode == "percent":
            return price * (1.0 - self.cfg.trail_pct) if self.side == "buy" else price * (1.0 + self.cfg.trail_pct)

        if self.cfg.mode == "atr":
            if not self.ohlcv_fn:
                LOG.warning("ATR mode requested but ohlcv_fn is not provided — skipping")
                return None
            try:
                # берём чуть больше истории, чем period
                df = await self.ohlcv_fn(self.symbol, max(100, self.cfg.atr_period * 4))
                atr = _atr(df, self.cfg.atr_period)
            except Exception as e:
                LOG.warning("ATR fetch/compute failed: %r", e)
                return None

            if atr <= 0:
                return None

            return price - self.cfg.atr_mult * atr if self.side == "buy" else price + self.cfg.atr_mult * atr

        return None

    # ---------- поиск текущего стопа ----------
    async def _find_existing_stop(self) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        """
        Ищем активный STOP_LOSS_LIMIT, противоположной стороне входа.
        Возвращаем (структура ордера Binance | None, orderListId | None).
        Если стоп — нога OCO, вернётся также orderListId.
        """
        try:
            orders = await self.adapter.list_open_orders(self.symbol)
        except Exception as e:
            LOG.warning("openOrders fetch failed: %r", e)
            return None, None

        close_side = "SELL" if self.side == "buy" else "BUY"

        # Ищем именно STOP_LOSS_LIMIT (как одиночный, так и OCO-стоп-нога)
        candidates: List[Dict[str, Any]] = [
            o for o in orders
            if (o.get("type") == "STOP_LOSS_LIMIT" and str(o.get("side")).upper() == close_side)
        ]
        if not candidates:
            return None, None

        # берём самый «свежий» по времени создания
        candidates.sort(key=lambda o: (o.get("time") or o.get("transactTime") or 0), reverse=True)
        chosen = candidates[0]
        order_list_id = chosen.get("orderListId")  # если это нога OCO — поле будет присутствовать
        try:
            order_list_id = int(order_list_id) if order_list_id is not None else None
        except Exception:
            order_list_id = None
        return chosen, order_list_id

    # ---------- создание/обновление стопа ----------
    async def _place_or_update_stop(self, desired_sl: float) -> Dict[str, Any]:
        """
        Если стопа нет — ставим. Если есть — решаем, стоит ли переставлять.
        Возвращаем подробную сводку.
        """
        # кулдаун
        now = time.time()
        if (now - self._last_action_ts) < max(0.0, self.cfg.cooldown_after_update_sec):
            return {"action": "skipped", "reason": "cooldown", "cooldown": self.cfg.cooldown_after_update_sec}

        if self.qty <= 0:
            return {"action": "skipped", "reason": "non_positive_qty"}

        existing, order_list_id = await self._find_existing_stop()
        prev_stop: Optional[float] = None
        if existing:
            try:
                prev_stop = float(existing.get("stopPrice") or existing.get("price") or 0.0)
            except Exception:
                prev_stop = None

        # Для long: стоп только вверх; для short: только вниз
        def _is_improvement(new_sl: float, old_sl: Optional[float]) -> bool:
            if old_sl is None or old_sl <= 0:
                return True
            return (new_sl > old_sl) if self.side == "buy" else (new_sl < old_sl)

        if not _is_improvement(desired_sl, prev_stop):
            return {"action": "skipped", "reason": "not_improvement", "prev_stop": prev_stop, "candidate": desired_sl}

        # Фильтры от мелких движений
        if prev_stop is not None and prev_stop > 0:
            abs_move = abs(desired_sl - prev_stop)
            rel_move = abs_move / max(prev_stop, 1e-12)
            if abs_move < max(0.0, self.cfg.min_abs_move) and rel_move < max(0.0, self.cfg.min_rel_move):
                return {"action": "skipped", "reason": "too_small_move", "prev_stop": prev_stop, "candidate": desired_sl}

        # Приведём qty/price к фильтрам биржи
        qty_adj, price_adj = await self.adapter.coerce_qty_price(self.symbol, self.qty, desired_sl)
        if qty_adj <= 0 or price_adj <= 0:
            return {"action": "skipped", "reason": "invalid_adjusted_values", "qty_adj": qty_adj, "price_adj": price_adj}

        # Если есть OCO и он разрешён к отмене — отменим весь список.
        if existing and order_list_id is not None and self.cfg.allow_cancel_oco:
            try:
                oco_res = await self.adapter.cancel_oco(order_list_id=order_list_id)
                LOG.info("Canceled OCO for %s: %s", self.symbol, oco_res)
            except Exception as e:
                LOG.warning("Cancel OCO failed (will proceed placing new stop anyway): %r", e)
        elif existing:
            # иначе отменим только сам стоп-ордер (best-effort)
            try:
                await self.adapter.cancel_order(
                    symbol=self.symbol,
                    order_id=int(existing.get("orderId")) if existing.get("orderId") is not None else None,
                    client_order_id=existing.get("clientOrderId"),
                )
            except Exception as e:
                LOG.warning("Cancel existing stop failed (will try place anyway): %r", e)

        # Поставим новый STOP_LOSS_LIMIT (лимит-цена = стоп-цена)
        try:
            resp = await self.adapter.place_stop_loss_limit(
                symbol=self.symbol,
                side=self.side,
                qty=qty_adj,
                stop_price=price_adj,
                price=price_adj,
                tif=self.cfg.time_in_force,
            )
            self._last_set_sl = float(price_adj)
            self._last_action_ts = time.time()
            return {
                "action": "placed" if not existing else "replaced",
                "symbol": self.symbol,
                "side": self.side,
                "qty": float(qty_adj),
                "prev_stop": prev_stop,
                "new_stop": float(price_adj),
                "status": resp.get("status"),
                "order_id": resp.get("orderId"),
                "client_order_id": resp.get("clientOrderId"),
            }
        except Exception as e:
            LOG.error("Place STOP_LOSS_LIMIT failed: %r", e)
            return {"action": "error", "reason": "place_failed", "error": repr(e)}

    # ---------- публичные методы ----------
    async def run_once(self) -> Optional[Dict[str, Any]]:
        """
        Один цикл: читаем цену → считаем желаемый SL → ставим/переставляем при необходимости.
        Возвращает краткую сводку об изменении или None, если ничего не делали.
        """
        if self._running_lock.locked():
            return None  # защищаемся от параллельных вызовов из вне

        async with self._running_lock:
            price = await self.adapter.get_price(self.symbol)
            if price is None or price <= 0:
                LOG.warning("No price for %s — skip trailing iteration", self.symbol)
                return {"action": "skipped", "reason": "no_price"}

            desired = await self._desired_sl(price)
            if desired is None or desired <= 0:
                return {"action": "skipped", "reason": "no_desired_sl"}

            # Для long стоп ниже цены, для short — выше; подстрахуемся
            if self.side == "buy" and desired >= price:
                desired = price * (1.0 - abs(self.cfg.trail_pct))
            elif self.side == "sell" and desired <= price:
                desired = price * (1.0 + abs(self.cfg.trail_pct))

            return await self._place_or_update_stop(float(desired))

    async def run_forever(self, *, stop_event: Optional[asyncio.Event] = None) -> None:
        """
        Бесконечный цикл трейлинга. Останавливается по stop_event.set().
        """
        interval = max(1.0, float(self.cfg.poll_interval_sec))
        while True:
            try:
                await self.run_once()
            except Exception as e:
                LOG.exception("Trailing iteration failed: %r", e)
            # остановка
            if stop_event is not None and stop_event.is_set():
                return
            await asyncio.sleep(interval)
