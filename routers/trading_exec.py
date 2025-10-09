# ai_trader/routers/trading_exec.py
from __future__ import annotations

import os
import time
import logging
import inspect
import asyncio
from typing import Optional, Literal, Dict, Any, List, Tuple, Callable

from fastapi import APIRouter, Query, Body, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from executors.api_binance import BinanceExecutor

# ──────────────────────────────────────────────────────────────────────────────
# Опциональная специфичная ошибка Binance
# ──────────────────────────────────────────────────────────────────────────────
try:  # type: ignore
    from executors.api_binance import BinanceAPIError  # noqa: F401
    _HAS_BINANCE_ERROR = True
except Exception:  # pragma: no cover
    BinanceAPIError = RuntimeError  # type: ignore
    _HAS_BINANCE_ERROR = False

# ──────────────────────────────────────────────────────────────────────────────
# Ключи/конфиг (env-фолбэк при отсутствии utils.secrets)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from utils.secrets import get_binance_keys, load_exec_config  # type: ignore
except Exception:
    def load_exec_config() -> Dict[str, Any]:
        """Фолбэк: пустой конфиг, если нет utils.secrets.load_exec_config."""
        return {}

    def get_binance_keys(*, testnet: bool) -> Tuple[str, str]:
        """Фолбэк: берём ключи из ENV, если нет utils.secrets.get_binance_keys."""
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

# ──────────────────────────────────────────────────────────────────────────────
# Исполнители: sim (опционально)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from executors.simulated import SimulatedExecutor  # type: ignore
except Exception:
    class SimulatedExecutor(BinanceExecutor):  # type: ignore
        """Простая сим-заглушка поверх интерфейса BinanceExecutor (минимально совместимая)."""
        name = "sim"

        async def open_order(self, **kwargs):
            symbol = (kwargs.get("symbol") or "BTCUSDT").upper()
            side = (kwargs.get("side") or "buy").lower()
            typ = (kwargs.get("type") or "market").lower()
            aliases = {
                "stop": "stop_limit",
                "stop_loss_limit": "stop_limit",
                "stop_loss": "stop_market",
                "take_profit": "take_profit_market",
            }
            typ = aliases.get(typ, typ)
            stop_price = kwargs.get("stop_price")
            price = kwargs.get("price")
            if typ in {"stop_limit", "take_profit_limit"}:
                price = price if price is not None else stop_price
            if typ in {"stop_market", "take_profit_market"}:
                price = stop_price if stop_price is not None else price
            price_f = float(price or 60000.0)

            qty_val = kwargs.get("qty")
            if (qty_val is None or float(qty_val or 0) <= 0) and kwargs.get("quote_qty") is not None:
                quote = float(kwargs.get("quote_qty") or 0)
                qty_val = quote / price_f if price_f else quote
            qty = float(qty_val or 0)
            if qty <= 0:
                qty = 0.001

            order_id = f"sim-{os.urandom(3).hex()}"
            return {
                "exchange": "sim",
                "testnet": True,
                "order_id": order_id,
                "client_order_id": kwargs.get("client_order_id"),
                "symbol": symbol,
                "side": side,
                "type": typ,
                "price": price_f,
                "qty": qty,
                "stop_price": float(stop_price) if stop_price is not None else None,
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

# ──────────────────────────────────────────────────────────────────────────────
# DB: сессия и логирование ордеров (опционально)
# ──────────────────────────────────────────────────────────────────────────────
try:
    from db.session import get_session  # type: ignore
    from db import crud_orders  # type: ignore
except Exception:
    get_session = None  # type: ignore
    crud_orders = None  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# RISK (опционально)
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# Router + logger
# ──────────────────────────────────────────────────────────────────────────────
router = APIRouter(prefix="/exec", tags=["exec"])
LOG = logging.getLogger("ai_trader.exec")

# Жёсткие лимиты времени, чтобы UI не висел навсегда
DEFAULT_OP_TIMEOUT = float(os.getenv("EXEC_OP_TIMEOUT_SEC", "20"))

# Кеш исполнителей на процесс: (mode,testnet) -> executor
_EXEC_CACHE: Dict[Tuple[str, bool], BinanceExecutor] = {}

# Короткий кеш доступных USDT-символов по экземпляру исполнителя: id(ex) -> set[str]
_USDT_SYMBOLS_CACHE: Dict[int, set[str]] = {}

# ──────────────────────────────────────────────────────────────────────────────
# helpers (ошибки)
# ──────────────────────────────────────────────────────────────────────────────
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
        msg = getattr(e, "msg", str(e)) or repr(e)
        if int(status) == 451:
            friendly = (
                "Binance API недоступен из вашего региона (HTTP 451). "
                "Переключитесь в режим симуляции или используйте VPN/другой доступ."
            )
            details = {"message": friendly, "original": msg}
            return _err("binance_restricted_location", http=int(status), code=code, error=details)
        return _err("binance_error", http=int(status), code=code, error={"message": msg})
    return _err("exec_error", http=400, error={"message": (str(e) or repr(e))})

# ──────────────────────────────────────────────────────────────────────────────
# executors
# ──────────────────────────────────────────────────────────────────────────────
def _get_binance_executor(testnet: bool) -> BinanceExecutor:
    """Создаём (или берём из кеша) реальный бинансовский исполнитель."""
    key = ("binance", bool(testnet))
    if key in _EXEC_CACHE:
        return _EXEC_CACHE[key]

    try:
        api_key, api_secret = get_binance_keys(testnet=testnet)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"config_error: {e}")

    cfg = load_exec_config() or {}
    bin_cfg = dict(cfg.get("binance") or {})
    # Явно уважаем значения из exec.yaml, но даём дефолты
    config = {
        "base_url": bin_cfg.get("base_url") or ("https://testnet.binance.vision" if testnet else "https://api.binance.com"),
        "recv_window": int(bin_cfg.get("recv_window", 5000)),
        "timeout": float(bin_cfg.get("timeout", 20.0)),
        "max_retries": int(bin_cfg.get("max_retries", 5)),
        "backoff_base": float(bin_cfg.get("backoff_base", 0.5)),
        "backoff_cap": float(bin_cfg.get("backoff_cap", 8.0)),
        "track_local_pos": bool(bin_cfg.get("track_local_pos", False)),
    }

    ex = BinanceExecutor(api_key=api_key, api_secret=api_secret, testnet=testnet, config=config)  # type: ignore[return-value]
    _EXEC_CACHE[key] = ex
    return ex

def _select_executor(mode: Literal["binance", "sim"], testnet: bool):
    """
    ВАЖНО: SimulatedExecutor в разных реализациях может не принимать api_key/api_secret.
    Также кешируем, чтобы не плодить клиентов на каждый запрос.
    """
    key = (mode, bool(testnet))
    if key in _EXEC_CACHE:
        return _EXEC_CACHE[key]

    if mode == "sim":
        try:
            ex = SimulatedExecutor(testnet=testnet)  # type: ignore[call-arg]
        except TypeError:
            ex = SimulatedExecutor()  # type: ignore[call-arg]
        _EXEC_CACHE[key] = ex
        return ex

    ex = _get_binance_executor(testnet=testnet)
    _EXEC_CACHE[key] = ex
    return ex

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
    stop_price: Optional[float],
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
        "stop_price": (stop_price if stop_price is not None else payload.get("stop_price") or payload.get("stopPrice")),
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
    log_fn: Optional[Callable[..., Any]] = getattr(crud_orders, "log_order", None)
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

async def _get_all_tickers_prices(executor) -> Dict[str, float]:
    """
    Пытаемся получить все цены одним батчем. Поддерживаем:
      - executor.client.get_all_tickers() -> Dict[str, Decimal] | List[{'symbol','price'}]
      - executor.get_all_tickers()       -> Dict[str, Decimal] | List[{'symbol','price'}]
    Возвращаем {SYMBOL: float(price)}. Пустой словарь — если не удалось.
    """
    prices: Dict[str, float] = {}

    async def _normalize(res: Any) -> Dict[str, float]:
        out: Dict[str, float] = {}
        if isinstance(res, dict):
            for k, v in res.items():
                try:
                    out[str(k).upper()] = float(v)
                except Exception:
                    continue
        elif isinstance(res, list):
            for it in res:
                try:
                    s = str(it.get("symbol") or "").upper()
                    p = it.get("price") or it.get("lastPrice") or it.get("markPrice")
                    if s and p is not None:
                        out[s] = float(p)
                except Exception:
                    continue
        return out

    for attr in ("client", None):  # сначала client, затем сам executor
        target = getattr(executor, attr, executor) if attr else executor
        fn = getattr(target, "get_all_tickers", None)
        if callable(fn):
            try:
                res = await asyncio.wait_for(fn(), timeout=5.0)
                norm = await _normalize(res)
                if norm:
                    prices.update(norm)
                    break
            except Exception as e:
                LOG.warning("get_all_tickers (%s) failed: %r", "client" if attr else "exec", e)

    return prices

async def _get_usdt_symbols(executor) -> set[str]:
    """
    Возвращает множество доступных символов вида XXXUSDT для текущего исполнения.
    Результат кешируется на время жизни процесса, чтобы не спамить exchangeInfo.
    """
    ex_id = id(executor)
    cached = _USDT_SYMBOLS_CACHE.get(ex_id)
    if cached is not None:
        return cached

    symbols: set[str] = set()
    try:
        client = getattr(executor, "client", None)
        if client and hasattr(client, "get_exchange_info"):
            async with asyncio.timeout(8.0):
                info = await client.get_exchange_info()
            items = info.get("symbols") if isinstance(info, dict) else None
            if isinstance(items, list):
                for it in items:
                    try:
                        if (it.get("quoteAsset") == "USDT") and (it.get("status") in (None, "TRADING", "BREAK")):
                            s = str(it.get("symbol") or "").upper()
                            if s.endswith("USDT"):
                                symbols.add(s)
                    except Exception:
                        continue
    except Exception as e:
        LOG.warning("get_exchange_info failed: %r", e)

    _USDT_SYMBOLS_CACHE[ex_id] = symbols
    return symbols

def _sum_stables(free_map: Dict[str, float], locked_map: Dict[str, float]) -> float:
    """Сумма стейблов (free+locked) как «минимально достоверный» equity."""
    return sum((free_map.get(ccy, 0.0) + locked_map.get(ccy, 0.0)) for ccy in _STABLES)

# ──────────────────────────────────────────────────────────────────────────────
# Устойчивая оценка equity в USDT
# ──────────────────────────────────────────────────────────────────────────────
async def _estimate_equity_usdt(executor) -> float:
    """
    Устойчивая оценка equity в USDT:
      - Стейблы считаем напрямую (без сети).
      - Для остальных активов конвертируем ТОЛЬКО через пары, которые реально есть в get_all_tickers().
      - Никаких одиночных вызовов /ticker/price?symbol=... → исключаем 400 Invalid symbol.
      - Никогда не бросаем исключений; на любой ошибке возвращаем текущее накопленное значение.
    """
    try:
        bal = await asyncio.wait_for(executor.fetch_balance(), timeout=min(10.0, DEFAULT_OP_TIMEOUT))
    except Exception:
        return 0.0

    # 1) Собираем карты free/locked
    free_map: Dict[str, float] = {}
    locked_map: Dict[str, float] = {}

    try:
        if isinstance(bal, dict) and isinstance(bal.get("balances"), list):
            for b in bal.get("balances", []):
                asset = str(b.get("asset") or "").strip().upper()
                if not asset:
                    continue
                try:
                    free = float(b.get("free") or 0) or 0.0
                except Exception:
                    free = 0.0
                try:
                    locked = float(b.get("locked") or 0) or 0.0
                except Exception:
                    locked = 0.0
                free_map[asset] = free_map.get(asset, 0.0) + free
                locked_map[asset] = locked_map.get(asset, 0.0) + locked
        else:
            # Фолбэк: поддерживаем формат {"free": {...}, "locked": {...}}
            if isinstance(bal, dict):
                f = bal.get("free")
                if isinstance(f, dict):
                    for a, v in f.items():
                        try:
                            free_map[str(a).upper()] = free_map.get(str(a).upper(), 0.0) + float(v or 0.0)
                        except Exception:
                            continue
                l = bal.get("locked")
                if isinstance(l, dict):
                    for a, v in l.items():
                        try:
                            locked_map[str(a).upper()] = locked_map.get(str(a).upper(), 0.0) + float(v or 0.0)
                        except Exception:
                            continue
    except Exception:
        # не падаем
        pass

    # 2) Стейблы — сразу в USDT
    total = 0.0
    for ccy in _STABLES:
        if ccy in free_map or ccy in locked_map:
            total += free_map.get(ccy, 0.0) + locked_map.get(ccy, 0.0)

    # 3) Список остальных активов с ненулевым остатком
    assets = (set(free_map.keys()) | set(locked_map.keys())) - _STABLES
    assets = {a for a in assets if (free_map.get(a, 0.0) + locked_map.get(a, 0.0)) > 0.0}
    if not assets:
        return float(total)

    # 4) Берём батч всех цен и выделяем доступные пары к USDT
    prices_map: Dict[str, float] = await _get_all_tickers_prices(executor)
    usdt_set: set[str] = {s for s in prices_map.keys() if s.endswith("USDT")}
    if not prices_map or not usdt_set:
        # нет цен — возвращаем хотя бы сумму стейблов
        return float(total)

    # 5) Конвертируем TOP-12 по объёму в USDT
    top_assets = sorted(
        assets,
        key=lambda a: free_map.get(a, 0.0) + locked_map.get(a, 0.0),
        reverse=True,
    )[:12]

    add = 0.0
    for a in top_assets:
        symbol = f"{a}USDT"
        if symbol not in usdt_set:
            continue
        px = prices_map.get(symbol)
        if not px or px <= 0:
            continue
        amount = free_map.get(a, 0.0) + locked_map.get(a, 0.0)
        if amount <= 0:
            continue
        add += amount * px

    return float(total + add)

async def _entry_price_for_order(
    executor,
    *,
    symbol: str,
    order_type: str,
    provided_price: Optional[float],
    stop_price: Optional[float] = None,
) -> Optional[float]:
    t = (order_type or "market").lower()
    if t == "limit" and provided_price is not None:
        return float(provided_price)
    if t in {"stop_limit", "take_profit_limit"} and provided_price is not None:
        return float(provided_price)
    if t in {"stop_market", "take_profit_market"} and stop_price is not None:
        return float(stop_price)
    for name in ("get_price", "get_ticker_price", "price"):
        m = getattr(executor, name, None)
        if callable(m):
            try:
                v = await asyncio.wait_for(m(symbol), timeout=2.0)
                if isinstance(v, dict):
                    v = v.get("price") or v.get("last") or v.get("close") or v.get("markPrice") or v.get("lastPrice")
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
        async with asyncio.timeout(min(10.0, DEFAULT_OP_TIMEOUT)):
            return await ex.list_positions()
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="positions timeout")

async def balance_data(mode: Literal["binance", "sim"] = "binance", testnet: bool = True, fast: bool = False):
    ex = _select_executor(mode, testnet)
    try:
        async with asyncio.timeout(min(15.0, DEFAULT_OP_TIMEOUT)):
            bal = await ex.fetch_balance()
            equity: float = 0.0

            if not fast:
                try:
                    equity = await _estimate_equity_usdt(ex)
                except Exception:
                    equity = 0.0
            else:
                equity = 0.0

            # 🔥 Fallback: если equity == 0, пробуем взять просто free USDT из bal
            if (equity is None) or (equity == 0.0):
                if isinstance(bal, dict):
                    if isinstance(bal.get("balances"), list):
                        for b in bal["balances"]:
                            if str(b.get("asset")).upper() == "USDT":
                                try:
                                    equity = float(b.get("free", 0.0)) + float(b.get("locked", 0.0))
                                except Exception:
                                    pass
                                break
                    elif "USDT" in (bal.get("free") or {}):
                        try:
                            equity = float((bal.get("free") or {}).get("USDT", 0.0)) + float((bal.get("locked") or {}).get("USDT", 0.0))
                        except Exception:
                            equity = float((bal.get("free") or {}).get("USDT", 0.0))

            risk_cfg = load_risk_config()
            state = ensure_day(
                load_state(getattr(risk_cfg, "tz_name", "Asia/Almaty")),
                getattr(risk_cfg, "tz_name", "Asia/Almaty"),
                current_equity=float(equity or 0.0),
            )
            base = bal if isinstance(bal, dict) else {}
            out = {
                **base,
                "equity_usdt": float(equity or 0.0),
                "risk": {
                    "daily_start_equity": start_of_day_equity(state),
                    "daily_max_loss_pct": getattr(risk_cfg, "daily_max_loss_pct", 0.02),
                    "max_trades_per_day": getattr(risk_cfg, "max_trades_per_day", 15),
                },
            }
            if fast:
                out["note"] = "fast_balance: equity_usdt computed as 0.0 (fast)"
            return out
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="balance timeout")

# ──────────────────────────────────────────────────────────────────────────────
# HEALTH
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/healthz")
async def healthz(
    mode: Literal["binance", "sim"] = Query("binance"),
    testnet: bool = Query(True),
):
    ex = _select_executor(mode, testnet)
    try:
        price = None
        try:
            async with asyncio.timeout(5.0):
                price = await ex.get_price("BTCUSDT")
        except Exception:
            pass
        info_ok = True
        try:
            client = getattr(ex, "client", None)
            if client and hasattr(client, "get_exchange_info"):
                async with asyncio.timeout(8.0):
                    await client.get_exchange_info()
        except Exception:
            info_ok = False
        return {"mode": mode, "testnet": testnet, "price_ok": price is not None, "exchange_info_ok": info_ok}
    except Exception as e:
        return _map_exc(e)

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
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

# ──────────────────────────────────────────────────────────────────────────────
# OPEN
# ──────────────────────────────────────────────────────────────────────────────
@router.post("/open")
async def open_order(
    mode: Literal["binance", "sim"] = Query("binance"),
    testnet: bool = Query(True),

    # query/body
    symbol: Optional[str] = Query(None),
    side: Optional[Literal["buy", "sell"]] = Query(None),
    type: Literal[
        "market",
        "limit",
        "stop_limit",
        "stop_market",
        "take_profit_limit",
        "take_profit_market",
        "stop",
        "stop_loss",
        "stop_loss_limit",
        "take_profit",
    ] = Query("market"),
    qty: Optional[float] = Query(None),
    price: Optional[float] = Query(None),
    stop_price: Optional[float] = Query(None, description="Stop trigger price for stop/TP orders"),
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
        symbol=symbol,
        side=side,
        qty=qty,
        price=price,
        stop_price=stop_price,
        type_default=type,
        timeInForce=timeInForce,
        body=body,
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
    raw_type = (p["type"] or "market").lower()
    type_aliases = {
        "stop": "stop_limit",
        "stop_loss_limit": "stop_limit",
        "stop_loss": "stop_market",
        "take_profit": "take_profit_market",
    }
    t = type_aliases.get(raw_type, raw_type)
    p["type"] = t
    if t == "limit":
        if p["qty"] is None or p["price"] is None:
            return _err("validation", http=422, error={"message": "LIMIT order requires both qty and price"})
    elif t == "market":
        if p["qty"] is None and p["quote_qty"] is None and p.get("sl_price") is None and p.get("sl_pct") is None:
            return _err(
                "validation",
                http=422,
                error={"message": "MARKET order requires qty or quote_qty (or sl to autosize)"},
            )
    elif t in {"stop_limit", "take_profit_limit"}:
        if p["qty"] is None or p.get("price") is None or p.get("stop_price") is None:
            return _err(
                "validation",
                http=422,
                error={"message": "STOP LIMIT order requires qty, price and stop_price"},
            )
    elif t in {"stop_market", "take_profit_market"}:
        if p.get("stop_price") is None:
            return _err("validation", http=422, error={"message": "STOP MARKET order requires stop_price"})
        if (
            p.get("qty") is None
            and p.get("quote_qty") is None
            and p.get("sl_price") is None
            and p.get("sl_pct") is None
        ):
            return _err(
                "validation",
                http=422,
                error={"message": "STOP MARKET order requires qty or quote_qty (or sl to autosize)"},
            )
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
        equity = await asyncio.wait_for(_estimate_equity_usdt(ex), timeout=min(10.0, DEFAULT_OP_TIMEOUT))
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
        return _err("risk_blocked", http=409, data=reason)

    # autosizing / SL-TP
    qty_adjusted = False
    risk_note = None
    entry_price = await _entry_price_for_order(
        ex,
        symbol=p["symbol"],
        order_type=p["type"],
        provided_price=p.get("price"),
        stop_price=p.get("stop_price"),
    )
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
        return _err("validation", http=422, error={"message": "qty computed is non-positive (check sl and risk config)"})

    # исполнение
    try:
        async with asyncio.timeout(min(20.0, DEFAULT_OP_TIMEOUT + 5)):
            open_with_protection = getattr(ex, "open_with_protection", None)
            if callable(open_with_protection) and p["type"] in {"market", "limit"}:
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
                    stop_price=p.get("stop_price"),
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
            "stop_price": result.get("stop_price") or p.get("stop_price"),
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
                    stop_price=response.get("stop_price"),
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
                            "stop_price": p.get("stop_price"),
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

        return response

    except asyncio.TimeoutError:
        return _err("exec_timeout", http=504, error={"message": "open order timeout"})
    except Exception as e:
        return _map_exc(e)

# ──────────────────────────────────────────────────────────────────────────────
# CLOSE / CANCEL
# ──────────────────────────────────────────────────────────────────────────────
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
                    async with asyncio.timeout(min(10.0, DEFAULT_OP_TIMEOUT)):
                        cancel_res = await client.delete_order(
                            symbol=symbol,
                            orderId=_order_id_int(order_id),
                            origClientOrderId=client_order_id,
                        )
                else:
                    cancel_res = {"status": "CANCELED", "orderId": order_id, "clientOrderId": client_order_id}
            except asyncio.TimeoutError:
                return _err("exec_timeout", http=504, error={"message": "cancel timeout"})
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

            return {"symbol": symbol, "cancel": cancel_res}

        # Режим 2: «закрыть руками» встречным MARKET (не передаём side в метод)
        async with asyncio.timeout(min(15.0, DEFAULT_OP_TIMEOUT)):
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

    except asyncio.TimeoutError:
        return _err("exec_timeout", http=504, error={"message": "close timeout"})
    except Exception as e:
        return _map_exc(e)

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
        async with asyncio.timeout(min(10.0, DEFAULT_OP_TIMEOUT)):
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

    except asyncio.TimeoutError:
        return _err("exec_timeout", http=504, error={"message": "cancel timeout"})
    except Exception as e:
        return _map_exc(e)

# ──────────────────────────────────────────────────────────────────────────────
# POSITIONS
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/positions")
async def list_positions(
    mode: Literal["binance", "sim"] = Query("binance"),
    testnet: bool = Query(True),
):
    try:
        positions = await list_positions_data(mode, testnet)
        return positions
    except Exception as e:
        return _map_exc(e)

# ---------- BALANCE ----------
@router.get("/balance")
async def get_balance(
    mode: Literal["binance", "sim"] = Query("binance"),
    testnet: bool = Query(True),
    fast: bool = Query(False, description="Если true — не считаем equity_usdt, только balances"),
):
    try:
        data = await balance_data(mode, testnet, fast=fast)
        return data
    except Exception as e:
        return _map_exc(e)

@router.get("/_debug/equity")
async def debug_equity(
    mode: Literal["binance", "sim"] = Query("binance"),
    testnet: bool = Query(True),
):
    ex = _select_executor(mode, testnet)
    try:
        eq = await _estimate_equity_usdt(ex)
        bal = await ex.fetch_balance()
        nonzero = []
        for b in (bal.get("balances") or []):
            try:
                t = float(b.get("free") or 0) + float(b.get("locked") or 0)
                if t > 0:
                    nonzero.append({"asset": b.get("asset"), "total": t})
            except Exception:
                continue
        nonzero.sort(key=lambda x: x["total"], reverse=True)
        return {"equity_usdt": eq, "top_assets": nonzero[:20]}
    except Exception as e:
        return _map_exc(e)
