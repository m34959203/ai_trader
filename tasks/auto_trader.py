# ai_trader/tasks/auto_trader.py
from __future__ import annotations

import os
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import httpx
import pandas as pd

from src.indicators import atr as atr_indicator
from src.strategy import ema_cross_signals
from utils.risk_config import load_risk_config, RiskConfig

LOG = logging.getLogger("ai_trader.auto_trader")

# ═════════════════════════════════════════════════════════════════════════════
# Конфигурация (ENV). Все значения безопасны для тестнета по умолчанию.
# ═════════════════════════════════════════════════════════════════════════════
ENABLED               = os.getenv("AUTO_TRADER_ENABLED", "1").strip() not in {"0", "false", "False", ""}
SYMBOLS               = [s.strip().upper() for s in os.getenv("AUTO_TRADER_SYMBOLS", "BTCUSDT").split(",") if s.strip()]
TIMEFRAME             = os.getenv("AUTO_TRADER_TIMEFRAME", "15m").strip()
_DEFAULT_APP_PORT     = os.getenv("APP_PORT") or os.getenv("PORT") or "8000"
_DEFAULT_API_BASE     = os.getenv("APP_API_BASE") or f"http://127.0.0.1:{_DEFAULT_APP_PORT}"
API_BASE              = os.getenv("AUTO_TRADER_API_BASE", _DEFAULT_API_BASE).rstrip("/")
SOURCE                = os.getenv("AUTO_TRADER_SOURCE", "binance")     # /ohlcv/* источник
MODE                  = os.getenv("AUTO_TRADER_MODE", "binance")       # для /exec/*
TESTNET               = os.getenv("AUTO_TRADER_TESTNET", "1").strip() not in {"0", "false", "False", ""}

# Стратегия (SMA crossover)
SMA_FAST              = int(os.getenv("AUTO_TRADER_SMA_FAST", "20"))
SMA_SLOW              = int(os.getenv("AUTO_TRADER_SMA_SLOW", "50"))
CONFIRM_ON_CLOSE      = os.getenv("AUTO_TRADER_CONFIRM_ON_CLOSE", "1").strip() not in {"0", "false", "False", ""}
# Если True — торгуем только один раз на закрытии новой свечи (no repaint).
SIGNAL_PERSIST        = int(os.getenv("AUTO_TRADER_SIGNAL_PERSIST", "1"))
SIGNAL_COOLDOWN       = int(os.getenv("AUTO_TRADER_SIGNAL_COOLDOWN", "0"))
SIGNAL_MIN_GAP_PCT    = float(os.getenv("AUTO_TRADER_SIGNAL_MIN_GAP_PCT", "0"))
USE_REGIME_FILTER     = os.getenv("AUTO_TRADER_USE_REGIME_FILTER", "0").strip() in {"1", "true", "True"}

# ATR / стопы
ATR_PERIOD            = int(os.getenv("AUTO_TRADER_ATR_PERIOD", "14"))
ATR_MULT              = float(os.getenv("AUTO_TRADER_ATR_MULT", "2.0"))
MIN_SL_PCT            = float(os.getenv("AUTO_TRADER_MIN_SL_PCT", "0.001"))
MAX_SL_PCT            = float(os.getenv("AUTO_TRADER_MAX_SL_PCT", "0.05"))

# Размер позиции / риск
QUOTE_USDT            = float(os.getenv("AUTO_TRADER_QUOTE_USDT", "50"))        # сумма покупки в USDT (потолок)
EQUITY_MIN_USDT       = float(os.getenv("AUTO_TRADER_EQUITY_MIN_USDT", "10"))   # минимальный порог equity
SL_PCT                = float(os.getenv("AUTO_TRADER_SL_PCT", "0"))             # 0..1 (например 0.01 = 1%); 0 — не задавать
TP_PCT                = float(os.getenv("AUTO_TRADER_TP_PCT", "0"))             # 0..1; 0 — не задавать
USE_RISK_AUTOSIZE     = os.getenv("AUTO_TRADER_USE_RISK_AUTOSIZE", "1").strip() not in {"0", "false", "False", ""}
# Если True, а стоп вычислен (>0) — ордер отправляется без qty/quote_qty, сервер сам рассчитает qty по риску.
USE_PORTFOLIO_RISK    = os.getenv("AUTO_TRADER_USE_PORTFOLIO_RISK", "1").strip() not in {"0", "false", "False", ""}
MAX_RISK_PORTFOLIO    = float(os.getenv("AUTO_TRADER_MAX_PORTFOLIO_RISK", "0"))  # 0 => брать из RiskConfig

# Продажа по death cross (для spot)
SELL_ON_DEATH         = os.getenv("AUTO_TRADER_SELL_ON_DEATH", "1").strip() not in {"0", "false", "False", ""}
SELL_PCT_OF_POSITION  = float(os.getenv("AUTO_TRADER_SELL_PCT_OF_POSITION", "1.0"))  # 0..1 — какую долю позиции продавать

# Контроль циклов / сеть
LOOP_SEC              = int(float(os.getenv("AUTO_TRADER_LOOP_SEC", "30")))
HTTP_TIMEOUT_SEC      = float(os.getenv("AUTO_TRADER_HTTP_TIMEOUT", "20"))
NET_MAX_RETRIES       = int(os.getenv("AUTO_TRADER_NET_MAX_RETRIES", "3"))
NET_RETRY_BASE        = float(os.getenv("AUTO_TRADER_NET_RETRY_BASE", "0.5"))  # сек
NET_RETRY_CAP         = float(os.getenv("AUTO_TRADER_NET_RETRY_CAP", "6.0"))   # сек

DRY_RUN               = os.getenv("AUTO_TRADER_DRY_RUN", "0").strip() in {"1", "true", "True"}

# Ограничение параллелизма (на случай множества символов)
CONCURRENCY           = max(1, int(os.getenv("AUTO_TRADER_CONCURRENCY", "4")))

# ═════════════════════════════════════════════════════════════════════════════
# Внутреннее состояние: чтобы не торговать много раз в одной и той же свече
# ═════════════════════════════════════════════════════════════════════════════
# symbol -> last_closed_candle_ts, last_signal(+1/-1)
_LAST_CANDLE_TS: Dict[str, int] = {}
_LAST_SIGNAL: Dict[str, int] = {}


@dataclass(frozen=True)
class RiskSnapshot:
    config: RiskConfig
    portfolio_used: float
    positions: List[Dict[str, Any]]


def _load_risk_config_cached() -> RiskConfig:
    try:
        cfg = load_risk_config().validate()
    except Exception:
        cfg = RiskConfig()  # type: ignore[call-arg]
    return cfg

# ═════════════════════════════════════════════════════════════════════════════
# Утилиты
# ═════════════════════════════════════════════════════════════════════════════
def _rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])
    df = pd.DataFrame(rows)
    rename = {"t": "ts", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df = df.rename(columns=rename)
    for col in ("ts", "open", "high", "low", "close", "volume"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ts", "close"]).copy()
    df["ts"] = df["ts"].astype(int)
    df = df.drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms" if df["ts"].max() > 10**12 else "s", utc=True)
    df = df.set_index("timestamp")
    return df


def _last_signal(df: pd.DataFrame) -> Tuple[int, int]:
    signals = ema_cross_signals(
        df,
        fast=SMA_FAST,
        slow=SMA_SLOW,
        persist=max(1, SIGNAL_PERSIST),
        cooldown=max(0, SIGNAL_COOLDOWN),
        min_gap_pct=max(0.0, SIGNAL_MIN_GAP_PCT),
        use_regime_filter=USE_REGIME_FILTER,
    )
    sig = int(signals["signal"].iloc[-1]) if not signals.empty else 0
    ts = int(signals["ts"].iloc[-1]) if not signals.empty else int(df.index[-1].timestamp())
    return sig, ts


def _atr_stop_pct(df: pd.DataFrame) -> float:
    if ATR_PERIOD <= 0 or ATR_MULT <= 0:
        return max(0.0, SL_PCT)
    try:
        atr_series = atr_indicator(df["high"], df["low"], df["close"], ATR_PERIOD)
        atr_val = float(atr_series.iloc[-1]) if len(atr_series) else 0.0
        last_close = float(df["close"].iloc[-1])
    except Exception:
        atr_val = 0.0
        last_close = 0.0
    if atr_val <= 0 or last_close <= 0:
        return max(0.0, SL_PCT)
    pct = (atr_val * ATR_MULT) / last_close
    pct = max(MIN_SL_PCT, pct)
    pct = max(pct, SL_PCT)
    pct = min(MAX_SL_PCT, pct)
    return pct

async def _sleep_backoff(attempt: int) -> None:
    # экспоненциальная задержка с капом
    delay = min(NET_RETRY_CAP, NET_RETRY_BASE * (2 ** attempt))
    await asyncio.sleep(delay)


def _position_risk_fraction(*, equity: float, qty: float, entry_price: float, stop_price: Optional[float], cfg: RiskConfig) -> float:
    if equity <= 0 or qty <= 0 or entry_price <= 0 or stop_price is None:
        return float(cfg.risk_pct_per_trade)
    dist = abs(entry_price - stop_price)
    min_dist = max(cfg.min_sl_distance_pct * entry_price, 1e-12)
    dist = max(dist, min_dist)
    return max(0.0, min(1.0, (dist * qty) / max(equity, 1e-9)))


def _portfolio_risk_used(positions: List[Dict[str, Any]], *, cfg: RiskConfig, equity: float) -> float:
    total = 0.0
    for pos in positions:
        try:
            qty = float(pos.get("qty") or pos.get("positionAmt") or 0.0)
        except Exception:
            qty = 0.0
        try:
            entry_price = float(pos.get("entry_price") or pos.get("entryPrice") or pos.get("avg_price") or 0.0)
        except Exception:
            entry_price = 0.0
        sl_price: Optional[float] = None
        raw_sl = pos.get("stop_loss_price") or pos.get("sl_price")
        try:
            if raw_sl is not None:
                sl_price = float(raw_sl)
        except Exception:
            sl_price = None
        if sl_price is None:
            try:
                sl_pct = float(pos.get("sl_pct") or 0.0)
                if sl_pct > 0 and entry_price > 0:
                    side = str(pos.get("side") or pos.get("positionSide") or "long").lower()
                    if side.startswith("short"):
                        sl_price = entry_price * (1.0 + sl_pct)
                    else:
                        sl_price = entry_price * (1.0 - sl_pct)
            except Exception:
                sl_price = None
        total += _position_risk_fraction(
            equity=equity,
            qty=float(qty),
            entry_price=float(entry_price),
            stop_price=sl_price,
            cfg=cfg,
        )
    return max(0.0, min(1.0, total))


def _risk_snapshot_from_positions(positions: List[Dict[str, Any]], *, equity: float) -> RiskSnapshot:
    cfg = _load_risk_config_cached()
    used = _portfolio_risk_used(positions, cfg=cfg, equity=equity) if USE_PORTFOLIO_RISK else 0.0
    if MAX_RISK_PORTFOLIO > 0:
        used = min(used, MAX_RISK_PORTFOLIO)
    return RiskSnapshot(config=cfg, portfolio_used=used, positions=positions)

async def _post_json(client: httpx.AsyncClient, url: str, json: Any) -> httpx.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(NET_MAX_RETRIES):
        try:
            r = await client.post(url, json=json, timeout=HTTP_TIMEOUT_SEC)
            if r.status_code >= 500:
                raise RuntimeError(f"{r.status_code} server error: {r.text[:200]}")
            return r
        except Exception as e:
            last_exc = e
            if attempt < NET_MAX_RETRIES - 1:
                await _sleep_backoff(attempt)
            else:
                break
    raise RuntimeError(f"POST {url} failed: {last_exc}")

async def _get(client: httpx.AsyncClient, url: str, params: Dict[str, Any]) -> httpx.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(NET_MAX_RETRIES):
        try:
            r = await client.get(url, params=params, timeout=HTTP_TIMEOUT_SEC)
            if r.status_code >= 500:
                raise RuntimeError(f"{r.status_code} server error: {r.text[:200]}")
            return r
        except Exception as e:
            last_exc = e
            if attempt < NET_MAX_RETRIES - 1:
                await _sleep_backoff(attempt)
            else:
                break
    raise RuntimeError(f"GET {url} failed: {last_exc}")

async def _fetch_candles(client: httpx.AsyncClient, symbol: str, timeframe: str, limit: int) -> List[Dict[str, Any]]:
    """
    /ohlcv/prices/query (POST JSON)
    допускает 2 формата ответа:
      1) {"rows":[{t,o,h,l,c,v}, ...]}
      2) [{t,o,h,l,c,v}, ...]
    """
    url = f"{API_BASE}/ohlcv/prices/query"
    payload = {
        "source": SOURCE,
        "symbol": symbol,
        "timeframe": timeframe,
        "limit": limit,
        "testnet": bool(TESTNET),
    }
    r = await _post_json(client, url, payload)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return data["rows"]
    if isinstance(data, list):
        return data
    raise RuntimeError(f"Unexpected /ohlcv/prices/query response: {type(data)}")

async def _fetch_equity(client: httpx.AsyncClient) -> float:
    """
    /exec/balance -> берем equity_usdt (если нет — 0)
    """
    url = f"{API_BASE}/exec/balance"
    params = {"mode": MODE, "testnet": str(TESTNET).lower(), "fast": "false"}
    r = await _get(client, url, params)
    r.raise_for_status()
    data = r.json()
    try:
        return float(data.get("equity_usdt") or 0.0)
    except Exception:
        return 0.0

async def _fetch_free_base_asset(client: httpx.AsyncClient, symbol: str) -> float:
    """
    Для spot продажи: узнаём, сколько базовой валюты (например BTC в BTCUSDT) свободно.
    """
    base = symbol.replace("USDT", "")
    url = f"{API_BASE}/exec/balance"
    params = {"mode": MODE, "testnet": str(TESTNET).lower(), "fast": "true"}
    r = await _get(client, url, params)
    r.raise_for_status()
    data = r.json()
    balances = data.get("balances") or []
    free = 0.0
    for b in balances:
        if str(b.get("asset")).upper() == base:
            try:
                free = float(b.get("free") or 0.0)
            except Exception:
                free = 0.0
            break
    return max(0.0, free)


async def _fetch_positions(client: httpx.AsyncClient) -> List[Dict[str, Any]]:
    url = f"{API_BASE}/exec/positions"
    params = {"mode": MODE, "testnet": str(TESTNET).lower()}
    r = await _get(client, url, params)
    if r.status_code >= 400:
        return []
    try:
        data = r.json()
    except Exception:
        return []
    if isinstance(data, dict) and "positions" in data:
        positions = data.get("positions") or []
        if isinstance(positions, list):
            return positions
    if isinstance(data, list):
        return data
    return []

async def _place_market_buy(
    client: httpx.AsyncClient,
    symbol: str,
    quote_usdt: float,
    *,
    sl_pct: float,
    tp_pct: float,
) -> Dict[str, Any]:
    """
    /exec/open (MARKET BUY).
    Если USE_RISK_AUTOSIZE=True и SL_PCT>0: не передаем qty/quote_qty — сервер сам рассчитает qty по риску.
    В остальных случаях передаём quote_qty (сумма в USDT).
    """
    url = f"{API_BASE}/exec/open"
    params = {
        "mode": MODE,
        "testnet": str(TESTNET).lower(),
        "symbol": symbol,
        "side": "buy",
        "type": "market",
    }
    if USE_RISK_AUTOSIZE and sl_pct > 0:
        # ничего не добавляем (qty/quote_qty), но передадим sl_pct/tp_pct
        pass
    else:
        params["quote_qty"] = f"{quote_usdt}"

    if sl_pct > 0:
        params["sl_pct"] = f"{sl_pct}"
    if tp_pct > 0:
        params["tp_pct"] = f"{tp_pct}"

    if DRY_RUN:
        return {"status": "DRY", "params": params}

    r = await client.post(url, params=params, timeout=HTTP_TIMEOUT_SEC)
    if r.status_code == 409:
        try:
            detail = r.json()
        except Exception:
            detail = {"error": "risk_blocked", "detail": r.text[:200]}
        return {"status": "BLOCKED", "detail": detail}
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text[:300]
        raise RuntimeError(f"open BUY failed {r.status_code}: {detail}")
    payload = r.json()
    payload.setdefault("status", "UNKNOWN")
    return payload

async def _place_market_sell(client: httpx.AsyncClient, symbol: str, qty: float) -> Dict[str, Any]:
    """
    /exec/open (MARKET SELL) для спота — просто рыночная продажа qty базового актива.
    """
    if qty <= 0:
        raise RuntimeError("sell qty must be positive")

    url = f"{API_BASE}/exec/open"
    params = {
        "mode": MODE,
        "testnet": str(TESTNET).lower(),
        "symbol": symbol,
        "side": "sell",
        "type": "market",
        "qty": f"{qty}",
    }
    if DRY_RUN:
        return {"status": "DRY", "params": params}
    # Для SELL SL/TP не имеет смысла в споте; опускаем.
    r = await client.post(url, params=params, timeout=HTTP_TIMEOUT_SEC)
    if r.status_code >= 400:
        raise RuntimeError(f"open SELL failed {r.status_code}: {r.text[:300]}")
    payload = r.json()
    payload.setdefault("status", "UNKNOWN")
    return payload

def _base_from_symbol(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def _calc_quote_amount(*, equity: float, sl_pct: float, cfg: RiskConfig) -> float:
    if equity <= 0:
        return 0.0
    base_cap = equity
    if QUOTE_USDT > 0:
        base_cap = min(base_cap, QUOTE_USDT)
    if sl_pct <= 0:
        return max(0.0, min(base_cap, equity))
    risk_fraction = cfg.risk_fraction_for_trade(sl_pct)
    if risk_fraction <= 0:
        return 0.0
    allowed = equity * risk_fraction
    return max(0.0, min(base_cap, allowed))

# ═════════════════════════════════════════════════════════════════════════════
# Основная логика по символу (один шаг цикла)
# ═════════════════════════════════════════════════════════════════════════════
async def _process_symbol(client: httpx.AsyncClient, symbol: str) -> None:
    try:
        limit = max(SMA_SLOW + 10, 200)
        candles = await _fetch_candles(client, symbol, TIMEFRAME, limit)
        if len(candles) < (SMA_SLOW + 2):
            LOG.info("[auto] %s %s — мало свечей: %d", symbol, TIMEFRAME, len(candles))
            return

        df = _rows_to_df(candles)
        if df.empty or len(df) < (SMA_SLOW + 2):
            LOG.info("[auto] %s %s — недостаточно данных для сигналов (%d)", symbol, TIMEFRAME, len(df))
            return

        sig, last_closed_ts = _last_signal(df)

        if CONFIRM_ON_CLOSE and _LAST_CANDLE_TS.get(symbol) == last_closed_ts and _LAST_SIGNAL.get(symbol) == sig:
            return

        _LAST_CANDLE_TS[symbol] = last_closed_ts
        _LAST_SIGNAL[symbol] = sig

        close_price = float(df["close"].iloc[-1])
        sl_pct_dynamic = _atr_stop_pct(df)
        tp_pct = TP_PCT if TP_PCT > 0 else 0.0

        if sig == +1:
            equity = await _fetch_equity(client)
            if equity <= 0:
                LOG.warning("[auto] %s — не удалось получить equity, пропуск", symbol)
                return
            if equity < max(EQUITY_MIN_USDT, 1.0) and not (USE_RISK_AUTOSIZE and sl_pct_dynamic > 0):
                LOG.info("[auto] %s — equity %.2f ниже порога %.2f", symbol, equity, max(EQUITY_MIN_USDT, 1.0))
                return

            positions = await _fetch_positions(client)
            snapshot = _risk_snapshot_from_positions(positions, equity=equity)

            if USE_PORTFOLIO_RISK:
                if len(snapshot.positions) >= snapshot.config.max_open_positions:
                    have_symbol = any(str(p.get("symbol", "")).upper() == symbol.upper() for p in snapshot.positions)
                    if not have_symbol:
                        LOG.info("[auto] %s — достигнут лимит позиций (%d)", symbol, snapshot.config.max_open_positions)
                        return
                next_risk = snapshot.config.risk_fraction_for_trade(sl_pct_dynamic)
                portfolio_cap = MAX_RISK_PORTFOLIO if MAX_RISK_PORTFOLIO > 0 else snapshot.config.portfolio_max_risk_pct
                if snapshot.portfolio_used + next_risk > portfolio_cap + 1e-6:
                    LOG.info("[auto] %s — портфельный риск %.3f > лимита %.3f, пропуск",
                             symbol, snapshot.portfolio_used + next_risk, portfolio_cap)
                    return

            quote_amt = _calc_quote_amount(equity=equity, sl_pct=sl_pct_dynamic, cfg=snapshot.config)
            if quote_amt <= 0 and not (USE_RISK_AUTOSIZE and sl_pct_dynamic > 0):
                LOG.info("[auto] %s — рассчитанный объём сделки <= 0 (equity=%.2f, sl_pct=%.4f)",
                         symbol, equity, sl_pct_dynamic)
                return

            try:
                res = await _place_market_buy(
                    client,
                    symbol,
                    quote_amt,
                    sl_pct=sl_pct_dynamic,
                    tp_pct=tp_pct,
                )
            except RuntimeError as exc:
                LOG.warning("[auto] BUY %s ошибка: %s", symbol, exc)
                return

            status = str(res.get("status") or "").upper()
            if status == "BLOCKED":
                LOG.warning("[auto] BUY %s отклонён риском: %s", symbol, res.get("detail"))
                return

            LOG.info(
                "[auto] BUY %s %s price≈%.4f quote≈%.2f sl_pct=%.4f tp_pct=%.4f -> %s (order_id=%s)",
                symbol,
                TIMEFRAME,
                close_price,
                quote_amt,
                sl_pct_dynamic,
                tp_pct,
                status or res.get("status"),
                res.get("order_id"),
            )
            return

        if sig == -1 and SELL_ON_DEATH:
            free_base = await _fetch_free_base_asset(client, symbol)
            sell_qty = max(0.0, free_base * SELL_PCT_OF_POSITION)
            if sell_qty <= 0:
                return
            try:
                res = await _place_market_sell(client, symbol, sell_qty)
            except RuntimeError as exc:
                LOG.warning("[auto] SELL %s ошибка: %s", symbol, exc)
                return
            LOG.info(
                "[auto] SELL %s qty≈%.8f -> %s (order_id=%s)",
                symbol,
                sell_qty,
                res.get("status"),
                res.get("order_id"),
            )

    except asyncio.CancelledError:
        raise
    except Exception as e:
        LOG.warning("[auto] %s ошибка шага: %r", symbol, e)

# ═════════════════════════════════════════════════════════════════════════════
# Публичная фон.корутина — добавьте в main.py
# ═════════════════════════════════════════════════════════════════════════════
async def background_loop() -> None:
    """
    Подключение в main.py:

        from tasks.auto_trader import background_loop

        @app.on_event("startup")
        async def start_auto_trader():
            asyncio.create_task(background_loop())

    Управление — через ENV (см. переменные вверху файла).
    """
    if not ENABLED:
        LOG.warning("[auto] AUTO_TRADER_ENABLED=0 — автотрейдер выключен.")
        return

    # Базовая валидация параметров
    if SMA_FAST <= 1 or SMA_SLOW <= 2 or SMA_FAST >= SMA_SLOW:
        LOG.error("[auto] Некорректные SMA: fast=%d slow=%d (fast < slow и >=2)", SMA_FAST, SMA_SLOW)
        return

    LOG.info(
        "[auto] старт: symbols=%s tf=%s ema(%d/%d) loop=%ss quote_cap=%.2f testnet=%s dry=%s risk_auto=%s "
        "atr=(period=%d mult=%.2f) sell_on_death=%s api=%s exec=%s/%s",
        ",".join(SYMBOLS),
        TIMEFRAME,
        SMA_FAST,
        SMA_SLOW,
        LOOP_SEC,
        QUOTE_USDT,
        TESTNET,
        DRY_RUN,
        USE_RISK_AUTOSIZE,
        ATR_PERIOD,
        ATR_MULT,
        SELL_ON_DEATH,
        API_BASE,
        MODE,
        SOURCE,
    )
    if USE_RISK_AUTOSIZE and SL_PCT <= 0 and (ATR_PERIOD <= 0 or ATR_MULT <= 0):
        LOG.warning("[auto] USE_RISK_AUTOSIZE включен, но стоп не задан (SL_PCT/ATR) — qty не будет рассчитан.")

    sem = asyncio.Semaphore(CONCURRENCY)

    async def _guarded(sym: str) -> None:
        async with sem:
            await _process_symbol(client, sym)

    # Один httpx-клиент на весь цикл
    async with httpx.AsyncClient() as client:
        try:
            while True:
                started = time.time()
                try:
                    if not SYMBOLS:
                        await asyncio.sleep(max(1, LOOP_SEC))
                        continue
                    tasks = [asyncio.create_task(_guarded(s)) for s in SYMBOLS]
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=False)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    LOG.error("[auto] ошибка итерации: %r", e)
                # Пауза между итерациями (учтём длительность шага)
                took = time.time() - started
                await asyncio.sleep(max(1.0, LOOP_SEC - min(LOOP_SEC - 1.0, took)))
        except asyncio.CancelledError:
            LOG.info("[auto] остановка по CancelledError")
        except Exception as e:
            LOG.exception("[auto] критическая ошибка цикла: %r", e)
        finally:
            LOG.info("[auto] фон-петля завершена")
