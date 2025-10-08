# routers/autopilot.py
from __future__ import annotations

import asyncio
import logging
from typing import Optional, Literal, Dict, Any

from fastapi import APIRouter, HTTPException, Query, Body
from pydantic import BaseModel, Field

from executors.api_binance import BinanceExecutor, BinanceAPIError

LOG = logging.getLogger("ai_trader.autopilot")
router = APIRouter(prefix="/autopilot", tags=["autopilot"])

# ──────────────────────────────────────────────────────────────────────────────
# Глобальное состояние пилота (один воркер для простоты)
# ──────────────────────────────────────────────────────────────────────────────
_TASK: Optional[asyncio.Task] = None
_STOP = asyncio.Event()
_EXECUTOR: Optional[BinanceExecutor] = None

_STATE: Dict[str, Any] = {
    "running": False,
    "symbol": None,
    "side_mode": None,
    "force": None,
    "step_sec": None,
    "testnet": True,
    "started_at": None,
    "last_error": None,
    "budget_usdt": None,
}

# ──────────────────────────────────────────────────────────────────────────────
# Модель для опционального override ключей через JSON-тело
# ──────────────────────────────────────────────────────────────────────────────
class ApiCreds(BaseModel):
    api_key: Optional[str] = Field(default=None, description="Binance API key (override ENV)")
    api_secret: Optional[str] = Field(default=None, description="Binance API secret (override ENV)")

# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные утилиты
# ──────────────────────────────────────────────────────────────────────────────
async def _get_symbol_assets(executor: BinanceExecutor, symbol: str) -> tuple[str, str]:
    """Вернуть (baseAsset, quoteAsset) по символу или кинуть 400."""
    s_info = await executor.client.get_symbol_info(symbol.upper())
    if not s_info:
        raise HTTPException(status_code=400, detail=f"Unknown symbol: {symbol.upper()}")
    base = s_info.get("baseAsset")
    quote = s_info.get("quoteAsset")
    if not base or not quote:
        raise HTTPException(status_code=400, detail=f"No base/quote assets for symbol {symbol.upper()}")
    return base, quote

async def _free_balances(executor: BinanceExecutor) -> dict:
    """Карта свободных балансов вида {'USDT': 123.45, 'BTC': 0.01, ...}."""
    acc = await executor.fetch_balance()
    return {b["asset"]: float(b["free"]) for b in (acc.get("balances") or [])}

async def _place_market_buy_by_quote(
    executor: BinanceExecutor,
    symbol: str,
    quote_amount: float,
) -> Dict[str, Any]:
    """
    MARKET BUY по quoteOrderQty (биржа сама валидирует NOTIONAL).
    ВАЖНО: ваш BinanceExecutor.open_order НЕ поддерживает параметр `test`,
    поэтому не передаём его сюда.
    """
    return await executor.open_order(
        symbol=symbol,
        side="buy",
        type="market",
        quote_qty=quote_amount,
        client_order_id=None,   # сгенерируется автоматически внутри executora
    )

async def _place_market_sell_by_qty(
    executor: BinanceExecutor,
    symbol: str,
    qty: float,
) -> Dict[str, Any]:
    """MARKET SELL по количеству (qty) с авто-округлением по фильтрам."""
    adj = await executor.round_qty(symbol, qty)
    if adj <= 0:
        raise ValueError("Adjusted quantity <= 0")
    return await executor.open_order(
        symbol=symbol,
        side="sell",
        type="market",
        qty=adj,
        client_order_id=None,   # сгенерируется автоматически
    )

# ──────────────────────────────────────────────────────────────────────────────
# Основной цикл автопилота
# ──────────────────────────────────────────────────────────────────────────────
async def _loop_autotrade(
    executor: BinanceExecutor,
    symbol: str,
    budget_usdt: float,
    step_sec: int,
    side_mode: Literal["both", "buy_only", "sell_only"],
    force: bool,
):
    LOG.info(
        "Autopilot started: symbol=%s budget=%.6f step=%ss side=%s force=%s",
        symbol, budget_usdt, step_sec, side_mode, force,
    )
    buy_turn = True

    try:
        base, quote = await _get_symbol_assets(executor, symbol)

        while not _STOP.is_set():
            try:
                # Балансы
                free = await _free_balances(executor)
                free_base = float(free.get(base, 0.0))
                free_quote = float(free.get(quote, 0.0))

                # Выбор стороны
                if side_mode == "buy_only":
                    side = "BUY"
                elif side_mode == "sell_only":
                    side = "SELL"
                else:
                    side = "BUY" if buy_turn else "SELL"

                # Стратегия объёма:
                # - BUY: quote_qty = min(budget_usdt, free_quote) (если !force),
                #         если force — используем budget_usdt (биржа валидирует NOTIONAL)
                # - SELL: продаём min(rounded(free_base), qty_from_budget) если !force,
                #         иначе — берём qty_from_budget (или микролот как fallback)
                if side == "BUY":
                    if force:
                        quote_amt = max(0.5, budget_usdt)  # >= ~0.5 для прохождения MIN_NOTIONAL многих пар
                    else:
                        if free_quote <= 0:
                            LOG.info("BUY skip: no free %s (free_quote=%.6f)", quote, free_quote)
                            await asyncio.sleep(step_sec)
                            buy_turn = not buy_turn
                            continue
                        quote_amt = min(budget_usdt, free_quote)

                    try:
                        r = await _place_market_buy_by_quote(executor, symbol, quote_amt)
                        LOG.info("BUY placed: symbol=%s quote=%.6f status=%s", symbol, quote_amt, r.get("status"))
                    except BinanceAPIError as e:
                        LOG.error("BUY BinanceAPIError: %s", e)
                        if not force:
                            raise
                    except Exception as e:
                        LOG.error("BUY error: %r", e)
                        if not force:
                            raise

                else:  # SELL
                    if free_base <= 0 and not force:
                        LOG.info("SELL skip: no free %s (free_base=%.8f)", base, free_base)
                        await asyncio.sleep(step_sec)
                        buy_turn = not buy_turn
                        continue

                    # Подберём qty исходя из бюджета и/или всего free_base
                    qty_target = free_base

                    # Попробуем оценить цену, чтобы связать бюджет с qty (best-effort)
                    try:
                        last_px = await executor.get_last_price(symbol)
                    except Exception:
                        last_px = None

                    if last_px and last_px > 0:
                        qty_from_budget = budget_usdt / float(last_px)
                        qty_target = min(free_base, qty_from_budget) if not force else max(qty_from_budget, 0.0)

                    # Fallback: если совсем пусто, но force — поставим минимальный пилотный лот
                    if (qty_target is None or qty_target <= 0.0) and force:
                        qty_target = 0.0005  # безопасный микролот; executor.round_qty его приведёт

                    try:
                        r = await _place_market_sell_by_qty(executor, symbol, qty=qty_target)
                        LOG.info("SELL placed: symbol=%s qty=%.8f status=%s", symbol, qty_target, r.get("status"))
                    except BinanceAPIError as e:
                        LOG.error("SELL BinanceAPIError: %s", e)
                        if not force:
                            raise
                    except Exception as e:
                        LOG.error("SELL error: %r", e)
                        if not force:
                            raise

                buy_turn = not buy_turn
                await asyncio.sleep(step_sec)

            except asyncio.CancelledError:
                LOG.info("Autopilot task cancelled.")
                break
            except Exception as e:
                LOG.exception("Autopilot iteration error: %r", e)
                _STATE["last_error"] = repr(e)
                if not force:
                    # Без форса — выходим при ошибке
                    break
                await asyncio.sleep(step_sec)

    finally:
        LOG.info("Autopilot stopped.")

# ──────────────────────────────────────────────────────────────────────────────
# Роуты управления
# ──────────────────────────────────────────────────────────────────────────────
@router.post("/start")
async def start_autopilot(
    symbol: str = Query("BTCUSDT"),
    budget_usdt: float = Query(50.0, ge=1.0),
    step_sec: int = Query(15, ge=5, le=600),
    side_mode: Literal["both", "buy_only", "sell_only"] = "both",
    force: bool = Query(False),
    testnet: bool = Query(True),
    # OPTIONAL override ключей: query...
    api_key_q: str | None = Query(None, alias="api_key"),
    api_secret_q: str | None = Query(None, alias="api_secret"),
    # ...или JSON-тело
    creds: ApiCreds = Body(default=ApiCreds()),
):
    """
    Запустить фоновый цикл автопилота.
    Ключи берутся из ENV (.env) через utils.secrets.get_binance_keys() внутри BinanceExecutor,
    если явно не переданы в запросе (query/body).
    """
    global _TASK, _STOP, _EXECUTOR, _STATE

    if _TASK and not _TASK.done():
        raise HTTPException(status_code=409, detail="Autopilot already running")

    # Если пользователь передал ключи — используем; иначе позволяем Executor достать из ENV
    api_key = api_key_q or creds.api_key
    api_secret = api_secret_q or creds.api_secret

    try:
        _EXECUTOR = BinanceExecutor(api_key=api_key, api_secret=api_secret, testnet=testnet)

        # sanity check connectivity + symbol
        await _EXECUTOR.client.get_time()
        await _get_symbol_assets(_EXECUTOR, symbol)

    except BinanceAPIError as e:
        if _EXECUTOR:
            await _EXECUTOR.close()
        _EXECUTOR = None
        raise HTTPException(status_code=400, detail=f"Binance connectivity/symbol check failed: {e.msg}") from e

    except RuntimeError as e:
        # Типичный случай: ключи не найдены в ENV и не переданы в запросе
        if _EXECUTOR:
            await _EXECUTOR.close()
        _EXECUTOR = None
        raise HTTPException(
            status_code=400,
            detail=(
                "API keys not found in environment. "
                "Заполните .env или передайте api_key/api_secret в запросе."
            ),
        ) from e

    except Exception as e:
        if _EXECUTOR:
            await _EXECUTOR.close()
        _EXECUTOR = None
        raise HTTPException(status_code=400, detail=f"Connectivity check error: {repr(e)}") from e

    _STOP = asyncio.Event()

    _STATE.update({
        "running": True,
        "symbol": symbol.upper(),
        "side_mode": side_mode,
        "force": force,
        "step_sec": step_sec,
        "testnet": testnet,
        "started_at": asyncio.get_running_loop().time(),
        "last_error": None,
        "budget_usdt": float(budget_usdt),
    })

    _TASK = asyncio.create_task(_loop_autotrade(
        _EXECUTOR, symbol, budget_usdt, step_sec, side_mode, force
    ))

    return {
        "status": "started",
        "symbol": symbol.upper(),
        "step_sec": step_sec,
        "force": force,
        "testnet": testnet,
        "side_mode": side_mode,
        "budget_usdt": float(budget_usdt),
    }

@router.post("/stop")
async def stop_autopilot():
    global _TASK, _STOP, _EXECUTOR, _STATE
    if not _TASK:
        return {"status": "idle"}

    _STOP.set()
    try:
        await asyncio.wait_for(_TASK, timeout=7.0)
    except asyncio.TimeoutError:
        _TASK.cancel()

    _TASK = None
    _STATE["running"] = False

    # Аккуратно закрыть HTTP-клиента
    if _EXECUTOR:
        try:
            await _EXECUTOR.close()
        finally:
            _EXECUTOR = None

    return {"status": "stopped"}

@router.get("/status")
async def status_autopilot():
    running = _TASK is not None and not _TASK.done()
    d = dict(_STATE)
    d["running"] = running
    return d
