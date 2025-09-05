# routers/trading_exec.py
from __future__ import annotations

import os
import time
import logging
import inspect
from typing import Optional, Literal, Dict, Any, List, Tuple

from fastapi import APIRouter, Query, Body, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from executors.api_binance import BinanceExecutor

# Опциональная специфичная ошибка Binance
try:  # type: ignore
    from executors.api_binance import BinanceAPIError  # noqa: F401
    _HAS_BINANCE_ERROR = True
except Exception:  # pragma: no cover
    BinanceAPIError = RuntimeError  # type: ignore
    _HAS_BINANCE_ERROR = False

# Ключи/конфиг
try:
    from utils.secrets import get_binance_keys, load_exec_config  # type: ignore
except Exception:
    def load_exec_config() -> Dict[str, Any]:
        return {}

    def get_binance_keys(*, testnet: bool) -> Tuple[str, str]:
        if testnet:
            k = os.getenv("BINANCE_TESTNET_API_KEY") or os.getenv("BINANCE_API_KEY_TESTNET")
            s = os.getenv("BINANCE_TESTNET_API_SECRET") or os.getenv("BINANCE_API_SECRET_TESTNET")
        else:
            k = os.getenv("BINANCE_API_KEY")
            s = os.getenv("BINANCE_API_SECRET")
        if not k or not s:
            raise RuntimeError(
                "Binance API keys not found. "
                "Set BINANCE_TESTNET_API_KEY/SECRET (testnet) or BINANCE_API_KEY/SECRET (mainnet)."
            )
        return k, s

# Исполнители: sim (опционально)
try:
    from executors.simulated import SimulatedExecutor  # type: ignore
except Exception:
    class SimulatedExecutor(BinanceExecutor):  # type: ignore
        """Простая сим-заглушка поверх интерфейса BinanceExecutor (минимально совместимая)."""
        name = "sim"

        async def open_order(self, **kwargs):
            symbol = (kwargs.get("symbol") or "BTCUSDT").upper()
            side = (kwargs.get("side") or "buy").lower()
            qty = float(kwargs.get("qty") or kwargs.get("quote_qty") or 0)
            order_id = f"sim-{os.urandom(3).hex()}"
            return {
                "exchange": "sim",
                "testnet": True,
                "order_id": order_id,
                "client_order_id": kwargs.get("client_order_id"),
                "symbol": symbol,
                "side": side,
                "type": (kwargs.get("type") or "market").lower(),
                "price": float(kwargs.get("price") or 60000.0),
                "qty": qty,
                "status": "FILLED",
                "protection": {"sl_price": kwargs.get("sl_price"), "tp_price": kwargs.get("tp_price")},
                "raw": {"mock": True, "orderId": order_id, "clientOrderId": kwargs.get("client_order_id")},
            }

        async def fetch_balance(self):
            return {
                "exchange": "sim",
                "testnet": True,
                "free": {"USDT": 1_000_000, "BTC": 100, "USDC": 0, "FDUSD": 0, "TUSD": 0, "BUSD": 0},
                "locked": {},
            }

        async def close_order(self, **kwargs):
            return {
                "exchange": "sim",
                "testnet": True,
                "status": "CANCELED",
                "order_id": kwargs.get("order_id") or f"sim-close-{os.urandom(2).hex()}",
                "raw": {"mock": True, "kwargs": kwargs},
            }

        async def list_positions(self):
            return []

        async def round_qty(self, symbol: str, qty: float) -> float:
            return float(qty)

# DB: сессия и логирование ордеров
try:
    from db.session import get_session  # type: ignore
    from db import crud_orders  # type: ignore
except Exception:
    get_session = None  # type: ignore
    crud_orders = None  # type: ignore

# RISK (опционально)
try:
    from utils.risk_config import load_risk_config  # type: ignore
    from state.daily_limits import (  # type: ignore
        ensure_day, load_state, can_open_more_trades, register_new_trade,
        register_realized_pnl, daily_loss_hit, start_of_day_equity
    )
    from risk.deadman import HEARTBEAT_FILE  # type: ignore
    _HAS_RISK = True
except Exception:
    _HAS_RISK = False

    class _RiskCfg:
        risk_pct_per_trade = float(os.getenv("RISK_PCT_PER_TRADE", "0.01"))
        daily_max_loss_pct = float(os.getenv("DAILY_MAX_LOSS_PCT", "0.02"))
        max_trades_per_day = int(os.getenv("MAX_TRADES_PER_DAY", "15"))
        deadman_max_stale_sec = int(os.getenv("DEADMAN_MAX_STALE_SEC", "90"))
        tz_name = os.getenv("TZ_NAME", "Asia/Almaty")

    def load_risk_config() -> _RiskCfg:  # type: ignore
        return _RiskCfg()

    def ensure_day(state, tz_name: str, current_equity: float):  # type: ignore
        class _S:
            def __init__(self, start_eq: float):
                self.day = "NA"
                self.start_equity = float(start_eq)
                self.trades_count = 0
                self.realized_pnl = 0.0
        return _S(current_equity)

    def load_state(tz_name: str):  # type: ignore
        return None

    def can_open_more_trades(state, max_trades_per_day: int) -> bool:  # type: ignore
        return True

    def register_new_trade(state) -> None:  # type: ignore
        pass

    def register_realized_pnl(state, pnl: float) -> None:  # type: ignore
        pass

    def daily_loss_hit(state, daily_max_loss_pct: float, current_equity: float) -> bool:  # type: ignore
        return False

    def start_of_day_equity(state) -> float:  # type: ignore
        return getattr(state, "start_equity", 0.0)

    HEARTBEAT_FILE = None  # type: ignore

router = APIRouter(prefix="/exec", tags=["exec"])
LOG = logging.getLogger("ai_trader.exec")

# ---------- helpers (ошибки) ----------
def _err(reason: str, *, http: int = 400, code: Optional[int] = None,
         error: Optional[Dict[str, Any]] = None, data: Any = None) -> JSONResponse:
    payload: Dict[str, Any] = {"error": reason}
    if code is not None:
        payload["code"] = code
    if error is not None:
        payload["details"] = error
    if data is not None:
        payload["data"] = data
    return JSONResponse(status_code=int(http), content=payload)

def _map_exc(e: Exception) -> JSONResponse:
    if _HAS_BINANCE_ERROR and isinstance(e, BinanceAPIError):  # type: ignore
        status = getattr(e, "status_code", 400) or 400
        code = getattr(e, "code", None)
        msg = getattr(e, "msg", str(e))
        return _err("binance_error", http=int(status), code=code, error={"message": msg})
    return _err("exec_error", http=400, error={"message": str(e)})

# ---------- executors ----------
def _get_binance_executor(testnet: bool) -> BinanceExecutor:
    try:
        api_key, api_secret = get_binance_keys(testnet=testnet)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"config_error: {e}")
    return BinanceExecutor(api_key=api_key, api_secret=api_secret, testnet=testnet)  # type: ignore[return-value]

def _select_executor(mode: Literal["binance", "sim"], testnet: bool):
    """
    ВАЖНО: SimulatedExecutor в разных реализациях может не принимать api_key/api_secret.
    """
    if mode == "sim":
        try:
            return SimulatedExecutor(testnet=testnet)  # type: ignore[call-arg]
        except TypeError:
            return SimulatedExecutor()  # type: ignore[call-arg]
    return _get_binance_executor(testnet=testnet)

def _order_id_int(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(str(x))
    except Exception:
        return None

def _merge_params(
    *,
    symbol: Optional[str],
    side: Optional[str],
    qty: Optional[float],
    price: Optional[float],
    type_default: str,
    timeInForce: Optional[str],
    body: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = dict(body or {})
    out: Dict[str, Any] = {
        "symbol": (symbol if symbol is not None else payload.get("symbol")),
        "side": (side if side is not None else payload.get("side")),
        "qty": (qty if qty is not None else payload.get("qty")),
        "price": (price if price is not None else payload.get("price")),
        "type": (payload.get("type") or type_default),
        "timeInForce": (timeInForce if timeInForce is not None else payload.get("timeInForce")),
        "client_order_id": payload.get("clientOrderId") or payload.get("client_order_id"),
        "quote_qty": payload.get("quote_qty") or payload.get("quoteOrderQty"),
        "sl_pct": payload.get("sl_pct"),
        "tp_pct": payload.get("tp_pct"),
        "sl_price": payload.get("sl_price"),
        "tp_price": payload.get("tp_price"),
        "reason": payload.get("reason"),
    }
    if out["symbol"]:
        out["symbol"] = str(out["symbol"]).upper().replace(" ", "")
    if out["side"]:
        out["side"] = str(out["side"]).lower()
    if out["type"]:
        out["type"] = str(out["type"]).lower()
    return out

# ──────────────────────────────────────────────────────────────────────────────
# SAFE logger wrapper — избегаем ошибки "unexpected keyword argument 'reason'"
# ──────────────────────────────────────────────────────────────────────────────
async def _safe_log_order(*, session: Any, **kwargs) -> None:
    """
    Передаём только те аргументы, которые реально поддерживает crud_orders.log_order
    (избавляемся от TypeError на Windows).
    """
    if crud_orders is None or session is None:
        return
    log_fn = getattr(crud_orders, "log_order", None)
    if not callable(log_fn):
        return
    sig = inspect.signature(log_fn)
    allowed = {k: v for k, v in kwargs.items() if (k in sig.parameters)}
    try:
        await log_fn(session=session, **allowed)
    except Exception as e:
        LOG.warning("order log failed: %r", e)

# ──────────────────────────────────────────────────────────────────────────────
# Risk helpers / heartbeat
# ──────────────────────────────────────────────────────────────────────────────
_STABLES = {"USDT", "USDC", "FDUSD", "TUSD", "BUSD", "USD"}

async def _estimate_equity_usdt(executor) -> float:
    """
    Суммарная оценка equity в USDT. Устойчива к разным форматам баланса.
    """
    try:
        bal = await executor.fetch_balance()
    except Exception:
        return 0.0

    free_map: Dict[str, float] = {}
    locked_map: Dict[str, float] = {}

    try:
        if isinstance(bal, dict) and isinstance(bal.get("balances"), list):
            for b in bal.get("balances", []):
                try:
                    asset = str(b.get("asset") or "").strip().upper()
                    if not asset:
                        continue
                    free = float(str(b.get("free") or "0") or 0)
                    locked = float(str(b.get("locked") or "0") or 0)
                    free_map[asset] = free_map.get(asset, 0.0) + free
                    locked_map[asset] = locked_map.get(asset, 0.0) + locked
                except Exception:
                    continue
        elif isinstance(bal, dict) and (isinstance(bal.get("free"), dict) or isinstance(bal.get("locked"), dict)):
            for k, v in dict(bal.get("free") or {}).items():
                try:
                    free_map[str(k).upper()] = free_map.get(str(k).upper(), 0.0) + float(str(v) or "0")
                except Exception:
                    continue
            for k, v in dict(bal.get("locked") or {}).items():
                try:
                    locked_map[str(k).upper()] = locked_map.get(str(k).upper(), 0.0) + float(str(v) or "0")
                except Exception:
                    continue
        elif isinstance(bal, dict):
            for k, v in bal.items():
                try:
                    asset = str(k).upper()
                    if asset in {"EXCHANGE", "TESTNET"}:
                        continue
                    free_map[asset] = free_map.get(asset, 0.0) + float(str(v) or "0")
                except Exception:
                    continue
        else:
            return 0.0
    except Exception:
        return 0.0

    total = 0.0
    for ccy in _STABLES:
        total += free_map.get(ccy, 0.0) + locked_map.get(ccy, 0.0)

    assets = (set(free_map.keys()) | set(locked_map.keys())) - _STABLES

    async def _get_price(sym: str) -> Optional[float]:
        for name in ("get_price", "get_ticker_price", "price"):
            m = getattr(executor, name, None)
            if callable(m):
                try:
                    v = await m(sym)
                    if isinstance(v, dict):
                        for key in ("price", "last", "close", "markPrice"):
                            if key in v:
                                v = v[key]
                                break
                    return float(str(v))
                except Exception:
                    continue
        return None

    for asset in assets:
        amount = free_map.get(asset, 0.0) + locked_map.get(asset, 0.0)
        if amount <= 0:
            continue
        symbol = f"{asset}USDT"
        px = await _get_price(symbol)
        if px is not None:
            total += amount * px

    return float(total)

async def _entry_price_for_order(executor, *, symbol: str, order_type: str, provided_price: Optional[float]) -> Optional[float]:
    t = (order_type or "market").lower()
    if t == "limit" and provided_price is not None:
        return float(provided_price)
    for name in ("get_price", "get_ticker_price", "price"):
        m = getattr(executor, name, None)
        if callable(m):
            try:
                v = await m(symbol)
                if isinstance(v, dict):
                    v = v.get("price") or v.get("last") or v.get("close")
                return float(v)
            except Exception:
                continue
    return None

def _sl_from_pct(side: str, entry_price: float, sl_pct: float) -> float:
    if sl_pct <= 0 or entry_price <= 0:
        return 0.0
    return entry_price * (1.0 - sl_pct) if side.lower() == "buy" else entry_price * (1.0 + sl_pct)

def _tp_from_pct(side: str, entry_price: float, tp_pct: float) -> float:
    if tp_pct <= 0 or entry_price <= 0:
        return 0.0
    return entry_price * (1.0 + tp_pct) if side.lower() == "buy" else entry_price * (1.0 - tp_pct)

def _stop_distance(entry_price: float, sl_price: float) -> float:
    d = abs(float(entry_price) - float(sl_price))
    return d if d > 0 else 0.0

def _max_loss_allowed(equity_usdt: float, risk_pct: float) -> float:
    return max(0.0, float(equity_usdt) * float(risk_pct))

def _risk_qty_from_distance(equity_usdt: float, risk_pct: float, stop_distance: float) -> float:
    if stop_distance <= 0:
        return 0.0
    return float(equity_usdt) * float(risk_pct) / float(stop_distance)

async def _deadman_check_and_close(executor, max_stale_sec: int) -> None:
    try:
        from risk.deadman import HEARTBEAT_FILE as _HB  # reimport для стабильности
    except Exception:
        _HB = None  # type: ignore
    try:
        if not _HB:
            return
        last = 0
        if _HB.exists():
            last = int(_HB.read_text(encoding="utf-8").strip() or "0")
        now = int(time.time())
        if last == 0 or (now - last) > int(max_stale_sec):
            LOG.error("Dead-man switch: heartbeat stale for %ss.", max_stale_sec)
            close_all = getattr(executor, "close_all_positions", None)
            if callable(close_all):
                try:
                    await close_all()
                except Exception as e:
                    LOG.warning("close_all_positions failed in deadman: %r", e)
            try:
                _HB.write_text(str(now), encoding="utf-8")
            except Exception:
                pass
    except Exception as e:
        LOG.warning("deadman check error: %r", e)

def _heartbeat_touch() -> None:
    try:
        from risk.deadman import HEARTBEAT_FILE as _HB  # локальный импорт
    except Exception:
        _HB = None  # type: ignore
    try:
        if _HB:
            _HB.parent.mkdir(parents=True, exist_ok=True)
            _HB.write_text(str(int(time.time())), encoding="utf-8")
    except Exception:
        pass

# ──────────────────────────────────────────────────────────────────────────────
# ВНУТРЕННИЕ ХЕЛПЕРЫ (для UI-частичных)
# ──────────────────────────────────────────────────────────────────────────────
async def list_positions_data(mode: Literal["binance", "sim"] = "binance", testnet: bool = True):
    ex = _select_executor(mode, testnet)
    try:
        return await ex.list_positions()
    finally:
        try:
            await ex.close()
        except Exception:
            pass

async def balance_data(mode: Literal["binance", "sim"] = "binance", testnet: bool = True):
    ex = _select_executor(mode, testnet)
    try:
        bal = await ex.fetch_balance()
        try:
            equity = await _estimate_equity_usdt(ex)
        except Exception:
            equity = None
        risk_cfg = load_risk_config()
        state = ensure_day(
            load_state(getattr(risk_cfg, "tz_name", "Asia/Almaty")),
            getattr(risk_cfg, "tz_name", "Asia/Almaty"),
            current_equity=float(equity or 0.0),
        )
        # Разворачиваем баланс на верхний уровень, чтобы был ключ "exchange"
        base = bal if isinstance(bal, dict) else {}
        return {
            **base,
            "equity_usdt": equity,
            "risk": {
                "daily_start_equity": start_of_day_equity(state),
                "daily_max_loss_pct": getattr(risk_cfg, "daily_max_loss_pct", 0.02),
                "max_trades_per_day": getattr(risk_cfg, "max_trades_per_day", 15),
            },
        }
    finally:
        try:
            await ex.close()
        except Exception:
            pass

# ---------- HEALTH ----------
@router.get("/healthz")
async def healthz(
    mode: Literal["binance", "sim"] = Query("binance"),
    testnet: bool = Query(True),
):
    ex = _select_executor(mode, testnet)
    try:
        price = None
        try:
            price = await ex.get_price("BTCUSDT")
        except Exception:
            pass
        info_ok = True
        try:
            client = getattr(ex, "client", None)
            if client and hasattr(client, "get_exchange_info"):
                await client.get_exchange_info()
        except Exception:
            info_ok = False
        # Плоский ответ
        return {"mode": mode, "testnet": testnet, "price_ok": price is not None, "exchange_info_ok": info_ok}
    except Exception as e:
        return _map_exc(e)
    finally:
        try:
            await ex.close()
        except Exception:
            pass

# ---------- CONFIG ----------
@router.get("/config")
async def exec_config(
    mode: Literal["binance", "sim"] = Query("binance"),
    testnet: bool = Query(True),
):
    try:
        cfg = load_exec_config() or {}
        has_keys = True
        try:
            get_binance_keys(testnet=testnet)
        except Exception:
            has_keys = False
        risk_cfg = load_risk_config()
        return {
            "mode": mode,
            "testnet": testnet,
            "config_loaded": bool(cfg),
            "has_api_keys": has_keys,
            "keys_source": "env",
            "risk": {
                "enabled": _HAS_RISK,
                "risk_pct_per_trade": getattr(risk_cfg, "risk_pct_per_trade", None),
                "daily_max_loss_pct": getattr(risk_cfg, "daily_max_loss_pct", None),
                "max_trades_per_day": getattr(risk_cfg, "max_trades_per_day", None),
                "deadman_max_stale_sec": getattr(risk_cfg, "deadman_max_stale_sec", None),
            },
        }
    except Exception as e:
        return _map_exc(e)

# ---------- OPEN ----------
@router.post("/open")
async def open_order(
    mode: Literal["binance", "sim"] = Query("binance"),
    testnet: bool = Query(True),

    # query/body
    symbol: Optional[str] = Query(None),
    side: Optional[Literal["buy", "sell"]] = Query(None),
    type: Literal["market", "limit"] = Query("market"),
    qty: Optional[float] = Query(None),
    price: Optional[float] = Query(None),
    quote_qty_q: Optional[float] = Query(None, alias="quote_qty", description="MARKET: сумма в квоте (например, 10 USDT)"),
    timeInForce: Optional[str] = Query(None, description="Напр. GTC для LIMIT"),

    # риск-поля:
    sl_price: Optional[float] = Query(None, description="Стоп-лосс ценой"),
    sl_pct: Optional[float] = Query(None, description="0.01 = 1%"),
    tp_price: Optional[float] = Query(None, description="Тейк-профит ценой"),
    tp_pct: Optional[float] = Query(None, description="0.02 = 2%"),

    body: Dict[str, Any] = Body(default=None),

    # DB
    session: AsyncSession = Depends(get_session) if get_session else None,  # type: ignore
):
    p = _merge_params(
        symbol=symbol, side=side, qty=qty, price=price, type_default=type, timeInForce=timeInForce, body=body
    )
    if quote_qty_q is not None:
        p["quote_qty"] = quote_qty_q
    if sl_price is not None: p["sl_price"] = sl_price
    if sl_pct is not None: p["sl_pct"] = sl_pct
    if tp_price is not None: p["tp_price"] = tp_price
    if tp_pct is not None: p["tp_pct"] = tp_pct

    # валидация
    if not p["symbol"] or not p["side"]:
        return _err("validation", http=422, error={"message": "symbol and side are required"})
    t = (p["type"] or "market").lower()
    if t == "limit":
        if p["qty"] is None or p["price"] is None:
            return _err("validation", http=422, error={"message": "LIMIT order requires both qty and price"})
    elif t == "market":
        if p["qty"] is None and p["quote_qty"] is None and p.get("sl_price") is None and p.get("sl_pct") is None:
            return _err("validation", http=422, error={"message": "MARKET order requires qty or quote_qty (or sl to autosize)"} )
    else:
        return _err("validation", http=422, error={"message": "Unsupported order type"})

    ex = _select_executor(mode, testnet)

    # HEARTBEAT / DEADMAN
    _heartbeat_touch()
    try:
        cfg = load_risk_config()
        await _deadman_check_and_close(ex, max_stale_sec=getattr(cfg, "deadman_max_stale_sec", 90))
    except Exception:
        pass

    # RISK-GATE
    try:
        equity = await _estimate_equity_usdt(ex)
    except Exception:
        equity = 0.0

    risk_cfg = load_risk_config()
    state = ensure_day(
        load_state(getattr(risk_cfg, "tz_name", "Asia/Almaty")),
        getattr(risk_cfg, "tz_name", "Asia/Almaty"),
        current_equity=equity,
    )

    if daily_loss_hit(state, getattr(risk_cfg, "daily_max_loss_pct", 0.02), current_equity=equity):
        reason = {
            "error": "risk_blocked",
            "why": f"daily_loss_hit:{getattr(risk_cfg, 'daily_max_loss_pct', 0.02)*100:.2f}%",
            "start_equity": start_of_day_equity(state),
            "equity_now": equity,
        }
        if session is not None and crud_orders is not None:
            await _safe_log_order(
                session=session,
                exchange=("binance" if mode == "binance" else "sim"),
                testnet=bool(testnet or mode == "sim"),
                symbol=p["symbol"],
                side=p["side"],
                type=p["type"],
                qty=p.get("qty") or p.get("quote_qty"),
                price=p.get("price"),
                status="BLOCKED",
                order_id=None,
                client_order_id=p.get("client_order_id"),
                raw={"risk": reason, "ts_ms": int(time.time() * 1000)},
            )
        try:
            await ex.close()
        except Exception:
            pass
        return _err("risk_blocked", http=409, data=reason)

    if not can_open_more_trades(state, getattr(risk_cfg, "max_trades_per_day", 15)):
        reason = {"error": "risk_blocked", "why": f"trades_limit_reached:{getattr(risk_cfg,'max_trades_per_day',15)}"}
        if session is not None and crud_orders is not None:
            await _safe_log_order(
                session=session,
                exchange=("binance" if mode == "binance" else "sim"),
                testnet=bool(testnet or mode == "sim"),
                symbol=p["symbol"],
                side=p["side"],
                type=p["type"],
                qty=p.get("qty") or p.get("quote_qty"),
                price=p.get("price"),
                status="BLOCKED",
                order_id=None,
                client_order_id=p.get("client_order_id"),
                raw={"risk": reason, "ts_ms": int(time.time() * 1000)},
            )
        try:
            await ex.close()
        except Exception:
            pass
        return _err("risk_blocked", http=409, data=reason)

    # autosizing / SL-TP
    qty_adjusted = False
    risk_note = None
    entry_price = await _entry_price_for_order(ex, symbol=p["symbol"], order_type=p["type"], provided_price=p.get("price"))
    final_sl_price = None
    final_tp_price = p.get("tp_price")

    try:
        if p.get("sl_price") is not None:
            final_sl_price = float(p["sl_price"])
        elif p.get("sl_pct") is not None and entry_price:
            final_sl_price = _sl_from_pct(p["side"], float(entry_price), float(p["sl_pct"]))
    except Exception:
        final_sl_price = None

    try:
        if final_tp_price is None and p.get("tp_pct") is not None and entry_price:
            final_tp_price = _tp_from_pct(p["side"], float(entry_price), float(p["tp_pct"]))
    except Exception:
        final_tp_price = None

    if final_sl_price and entry_price:
        stop_dist = _stop_distance(float(entry_price), float(final_sl_price))
        max_allowed_loss = _max_loss_allowed(equity, getattr(risk_cfg, "risk_pct_per_trade", 0.01))
        if p.get("qty") is None and p.get("quote_qty") is None:
            auto_qty = _risk_qty_from_distance(equity, getattr(risk_cfg, "risk_pct_per_trade", 0.01), stop_dist)
            if auto_qty > 0:
                p["qty"] = auto_qty
                risk_note = "qty_autosized_from_risk"
        elif p.get("qty") is not None:
            current_loss = float(stop_dist) * float(p["qty"])
            if current_loss > max_allowed_loss > 0:
                safe_qty = max_allowed_loss / float(stop_dist)
                p["qty"] = float(safe_qty)
                qty_adjusted = True
                risk_note = "qty_adjusted_to_risk"

    # округление qty по фильтрам
    try:
        round_qty = getattr(ex, "round_qty", None)
        if callable(round_qty) and p.get("qty") is not None:
            p["qty"] = float(await round_qty(p["symbol"], float(p["qty"])))
    except Exception:
        pass

    if p.get("qty") is not None and float(p["qty"]) <= 0:
        try:
            await ex.close()
        except Exception:
            pass
        return _err("validation", http=422, error={"message": "qty computed is non-positive (check sl and risk config)"})

    # исполнение
    try:
        open_with_protection = getattr(ex, "open_with_protection", None)
        if callable(open_with_protection):
            result = await open_with_protection(
                symbol=p["symbol"],
                side=p["side"],
                qty=p.get("qty"),
                entry_type=p["type"],
                entry_price=p.get("price"),
                sl_price=final_sl_price,
                tp_price=final_tp_price,
                client_order_id=p.get("client_order_id"),
                quote_qty=p.get("quote_qty"),
                timeInForce=p.get("timeInForce"),
            )
        else:
            result = await ex.open_order(
                symbol=p["symbol"],
                side=p["side"],
                type=p["type"],
                qty=p.get("qty"),
                price=p.get("price"),
                timeInForce=p.get("timeInForce"),
                client_order_id=p.get("client_order_id"),
                quote_qty=p.get("quote_qty"),
            )

        response = {
            "order_id": result.get("order_id"),
            "client_order_id": result.get("client_order_id") or p.get("client_order_id"),
            "status": result.get("status"),
            "symbol": result.get("symbol") or p.get("symbol"),
            "side": result.get("side") or p.get("side"),
            "type": result.get("type") or p.get("type"),
            "qty": result.get("qty") or p.get("qty") or p.get("quote_qty"),
            "price": result.get("price") or p.get("price") or entry_price,
            "risk_note": risk_note,
            "sl_price": final_sl_price,
            "sl_pct": p.get("sl_pct"),
            "tp_price": final_tp_price,
            "tp_pct": p.get("tp_pct"),
            "protection_ids": result.get("protection_ids"),
            "protection": result.get("protection"),
        }

        try:
            if str(response.get("status", "")).upper() in ("FILLED", "NEW", "PARTIALLY_FILLED", "ACCEPTED"):
                register_new_trade(state)
        except Exception:
            pass

        # журнал
        if session is not None and crud_orders is not None:
            log_from_resp = getattr(crud_orders, "log_from_exchange_response", None)
            if callable(log_from_resp) and isinstance(result.get("raw"), dict) and mode == "binance":
                try:
                    await log_from_resp(
                        session=session,
                        exchange="binance",
                        testnet=bool(testnet),
                        symbol=p["symbol"],
                        side=p["side"],
                        type=p["type"],
                        response=result["raw"],
                        time_in_force=p.get("timeInForce"),
                        commit=True,
                    )
                except Exception as e:
                    LOG.warning("order log (rich) failed: %r", e)
            else:
                await _safe_log_order(
                    session=session,
                    exchange=("binance" if mode == "binance" else "sim"),
                    testnet=bool(testnet or mode == "sim"),
                    symbol=p["symbol"],
                    side=p["side"],
                    type=p["type"],
                    qty=response.get("qty"),
                    price=response.get("price"),
                    status=response.get("status"),
                    order_id=response.get("order_id"),
                    client_order_id=response.get("client_order_id"),
                    raw={
                        "request": {
                            "symbol": p["symbol"],
                            "side": p["side"],
                            "type": p["type"],
                            "qty": p.get("qty"),
                            "price": p.get("price"),
                            "timeInForce": p.get("timeInForce"),
                            "client_order_id": p.get("client_order_id"),
                            "quote_qty": p.get("quote_qty"),
                            "sl_pct": p.get("sl_pct"),
                            "sl_price": p.get("sl_price"),
                            "tp_pct": p.get("tp_pct"),
                            "tp_price": p.get("tp_price"),
                        },
                        "response": result,
                        "equity_usdt": equity,
                        "start_equity": start_of_day_equity(state),
                        "qty_adjusted": qty_adjusted,
                        "ts_ms": int(time.time() * 1000),
                    },
                )

        # Плоский ответ (без "ok/data")
        return response

    except Exception as e:
        return _map_exc(e)
    finally:
        try:
            await ex.close()
        except Exception:
            pass

# ---------- CLOSE / CANCEL ----------
@router.post("/close")
async def close_order(
    mode: Literal["binance", "sim"] = Query("binance"),
    testnet: bool = Query(True),

    # режим 1 — отмена активного ордера
    symbol: Optional[str] = Query(None),
    order_id: Optional[str] = Query(None, description="ID ордера на отмену"),
    client_order_id: Optional[str] = Query(None, description="ClientOrderId на отмену"),

    # режим 2 — «закрыть руками»: встречный MARKET
    side: Optional[Literal["buy", "sell"]] = Query(None, description="При ручном закрытии — визуальная метка в логе"),
    qty: Optional[float] = Query(None, description="Количество для встречного MARKET"),

    realized_pnl: Optional[float] = Query(None),
    session: AsyncSession = Depends(get_session) if get_session else None,  # type: ignore
):
    if not symbol:
        return _err("validation", http=422, error={"message": "symbol is required"})

    ex = _select_executor(mode, testnet)
    _heartbeat_touch()

    try:
        # Режим 1: отмена ордера
        if (order_id or client_order_id) and not qty:
            cancel_res = None
            try:
                client = getattr(ex, "client", None)
                if client and hasattr(client, "delete_order"):
                    cancel_res = await client.delete_order(
                        symbol=symbol,
                        orderId=_order_id_int(order_id),
                        origClientOrderId=client_order_id,
                    )
                else:
                    cancel_res = {"status": "CANCELED", "orderId": order_id, "clientOrderId": client_order_id}
            except Exception as e:
                return _map_exc(e)

            if session is not None and crud_orders is not None:
                await _safe_log_order(
                    session=session,
                    exchange=("binance" if mode == "binance" else "sim"),
                    testnet=bool(testnet or mode == "sim"),
                    symbol=str(symbol).upper(),
                    side=None,
                    type=None,
                    qty=None,
                    status=cancel_res.get("status") or "CANCELED",
                    order_id=str(cancel_res.get("orderId") or order_id or ""),
                    client_order_id=str(cancel_res.get("clientOrderId") or client_order_id or ""),
                    raw={"response": cancel_res, "ts_ms": int(time.time() * 1000)},
                )

            # Плоский ответ
            return {"symbol": symbol, "cancel": cancel_res}

        # Режим 2: «закрыть руками» встречным MARKET (не передаём side в метод)
        result = await ex.close_order(symbol=symbol, qty=qty, type="market")

        if realized_pnl is not None:
            try:
                register_realized_pnl(
                    ensure_day(load_state(load_risk_config().tz_name), load_risk_config().tz_name, current_equity=0.0),
                    float(realized_pnl),
                )
            except Exception:
                pass

        if session is not None and crud_orders is not None:
            await _safe_log_order(
                session=session,
                exchange=("binance" if mode == "binance" else "sim"),
                testnet=bool(testnet or mode == "sim"),
                symbol=str(symbol).upper(),
                side=(side or "sell"),  # только для лога
                type="market",
                qty=qty,
                status=result.get("status"),
                order_id=result.get("order_id"),
                client_order_id=result.get("client_order_id"),
                raw={"response": result, "ts_ms": int(time.time() * 1000), "realized_pnl": realized_pnl},
            )

        return result

    except Exception as e:
        return _map_exc(e)
    finally:
        try:
            await ex.close()
        except Exception:
            pass

@router.post("/cancel")
async def cancel_order(
    mode: Literal["binance", "sim"] = Query("binance"),
    testnet: bool = Query(True),
    symbol: str = Query(...),
    order_id: Optional[str] = Query(None),
    client_order_id: Optional[str] = Query(None),
    session: AsyncSession = Depends(get_session) if get_session else None,  # type: ignore
):
    if not (order_id or client_order_id):
        return _err("validation", http=422, error={"message": "order_id or client_order_id is required"})

    ex = _select_executor(mode, testnet)
    _heartbeat_touch()

    try:
        client = getattr(ex, "client", None)
        if client and hasattr(client, "delete_order"):
            result = await client.delete_order(
                symbol=symbol,
                orderId=_order_id_int(order_id),
                origClientOrderId=client_order_id,
            )
        else:
            result = {"status": "CANCELED", "orderId": order_id, "clientOrderId": client_order_id}

        if session is not None and crud_orders is not None:
            await _safe_log_order(
                session=session,
                exchange=("binance" if mode == "binance" else "sim"),
                testnet=bool(testnet or mode == "sim"),
                symbol=str(symbol).upper(),
                side=None,
                type=None,
                qty=None,
                status=result.get("status"),
                order_id=str(result.get("orderId") or order_id or ""),
                client_order_id=str(result.get("ClientOrderId") or result.get("clientOrderId") or client_order_id or ""),
                raw={"response": result, "ts_ms": int(time.time() * 1000)},
            )

        return result

    except Exception as e:
        return _map_exc(e)
    finally:
        try:
            await ex.close()
        except Exception:
            pass

# ---------- POSITIONS ----------
@router.get("/positions")
async def list_positions(
    mode: Literal["binance", "sim"] = Query("binance"),
    testnet: bool = Query(True),
):
    try:
        positions = await list_positions_data(mode, testnet)
        # ВОЗВРАЩАЕМ непосредственно список (без _ok-обёртки)
        return positions
    except Exception as e:
        return _map_exc(e)


# ---------- BALANCE ----------
@router.get("/balance")
async def get_balance(
    mode: Literal["binance", "sim"] = Query("binance"),
    testnet: bool = Query(True),
):
    try:
        data = await balance_data(mode, testnet)
        # Плоский ответ: на верхнем уровне есть "exchange"
        return data
    except Exception as e:
        return _map_exc(e)
