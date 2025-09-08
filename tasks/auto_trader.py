# ai_trader/tasks/auto_trader.py
from __future__ import annotations

import os
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple

import httpx

LOG = logging.getLogger("ai_trader.auto_trader")

# ═════════════════════════════════════════════════════════════════════════════
# Конфигурация (ENV). Все значения безопасны для тестнета по умолчанию.
# ═════════════════════════════════════════════════════════════════════════════
ENABLED               = os.getenv("AUTO_TRADER_ENABLED", "1").strip() not in {"0", "false", "False", ""}
SYMBOLS               = [s.strip().upper() for s in os.getenv("AUTO_TRADER_SYMBOLS", "BTCUSDT").split(",") if s.strip()]
TIMEFRAME             = os.getenv("AUTO_TRADER_TIMEFRAME", "15m").strip()
API_BASE              = os.getenv("AUTO_TRADER_API_BASE", "http://127.0.0.1:8001").rstrip("/")
SOURCE                = os.getenv("AUTO_TRADER_SOURCE", "binance")     # /ohlcv/* источник
MODE                  = os.getenv("AUTO_TRADER_MODE", "binance")       # для /exec/*
TESTNET               = os.getenv("AUTO_TRADER_TESTNET", "1").strip() not in {"0", "false", "False", ""}

# Стратегия (SMA crossover)
SMA_FAST              = int(os.getenv("AUTO_TRADER_SMA_FAST", "20"))
SMA_SLOW              = int(os.getenv("AUTO_TRADER_SMA_SLOW", "50"))
CONFIRM_ON_CLOSE      = os.getenv("AUTO_TRADER_CONFIRM_ON_CLOSE", "1").strip() not in {"0", "false", "False", ""}
# Если True — торгуем только один раз на закрытии новой свечи (no repaint).

# Размер позиции / риск
QUOTE_USDT            = float(os.getenv("AUTO_TRADER_QUOTE_USDT", "50"))        # сумма покупки в USDT
EQUITY_MIN_USDT       = float(os.getenv("AUTO_TRADER_EQUITY_MIN_USDT", "10"))   # минимальный порог equity
SL_PCT                = float(os.getenv("AUTO_TRADER_SL_PCT", "0"))             # 0..1 (например 0.01 = 1%); 0 — не задавать
TP_PCT                = float(os.getenv("AUTO_TRADER_TP_PCT", "0"))             # 0..1; 0 — не задавать
USE_RISK_AUTOSIZE     = os.getenv("AUTO_TRADER_USE_RISK_AUTOSIZE", "0").strip() in {"1", "true", "True"}
# Если True, а SL_PCT > 0 — ордер отправляется без qty/quote_qty, сервер сам рассчитает qty по риску.

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

# ═════════════════════════════════════════════════════════════════════════════
# Утилиты
# ═════════════════════════════════════════════════════════════════════════════
def _sma(values: List[float], length: int) -> Optional[float]:
    if length <= 0 or len(values) < length:
        return None
    return sum(values[-length:]) / float(length)

def _cross(prev_fast: float, prev_slow: float, cur_fast: float, cur_slow: float) -> int:
    """
    +1: golden cross (fast пересёк снизу вверх)
    -1: death cross  (fast пересёк сверху вниз)
     0: нет события
    """
    if prev_fast is None or prev_slow is None or cur_fast is None or cur_slow is None:
        return 0
    if prev_fast <= prev_slow and cur_fast > cur_slow:
        return +1
    if prev_fast >= prev_slow and cur_fast < cur_slow:
        return -1
    return 0

async def _sleep_backoff(attempt: int) -> None:
    # экспоненциальная задержка с капом
    delay = min(NET_RETRY_CAP, NET_RETRY_BASE * (2 ** attempt))
    await asyncio.sleep(delay)

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

async def _place_market_buy(client: httpx.AsyncClient, symbol: str, quote_usdt: float) -> Dict[str, Any]:
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
    if USE_RISK_AUTOSIZE and SL_PCT > 0:
        # ничего не добавляем (qty/quote_qty), но передадим sl_pct/tp_pct
        pass
    else:
        params["quote_qty"] = f"{quote_usdt}"

    if SL_PCT > 0:
        params["sl_pct"] = f"{SL_PCT}"
    if TP_PCT > 0:
        params["tp_pct"] = f"{TP_PCT}"

    r = await _get(client, url, params) if DRY_RUN else await client.post(url, params=params, timeout=HTTP_TIMEOUT_SEC)
    if r.status_code >= 400:
        raise RuntimeError(f"open BUY failed {r.status_code}: {r.text[:300]}")
    return r.json()

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
    # Для SELL SL/TP не имеет смысла в споте; опускаем.
    r = await _get(client, url, params) if DRY_RUN else await client.post(url, params=params, timeout=HTTP_TIMEOUT_SEC)
    if r.status_code >= 400:
        raise RuntimeError(f"open SELL failed {r.status_code}: {r.text[:300]}")
    return r.json()

def _base_from_symbol(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol

# ═════════════════════════════════════════════════════════════════════════════
# Основная логика по символу (один шаг цикла)
# ═════════════════════════════════════════════════════════════════════════════
async def _process_symbol(client: httpx.AsyncClient, symbol: str) -> None:
    try:
        limit = max(SMA_SLOW + 5, 120)
        candles = await _fetch_candles(client, symbol, TIMEFRAME, limit)
        if len(candles) < (SMA_SLOW + 2):
            LOG.info("[auto] %s %s — мало свечей: %d", symbol, TIMEFRAME, len(candles))
            return

        # сортируем на всякий
        candles = sorted(candles, key=lambda r: int(r.get("t", 0)))
        closes: List[float] = []
        times:  List[int] = []
        for row in candles:
            t = row.get("t")
            c = row.get("c", row.get("close"))
            if t is None or c is None:
                continue
            try:
                times.append(int(t))
                closes.append(float(c))
            except Exception:
                continue

        if len(closes) < (SMA_SLOW + 2):
            LOG.info("[auto] %s %s — мало закрытий: %d", symbol, TIMEFRAME, len(closes))
            return

        # считаем по закрытым свечам:
        # prev_* — на предпоследней свече, cur_* — на последней (текущей закрытой)
        prev_fast = _sma(closes[:-1], SMA_FAST)
        prev_slow = _sma(closes[:-1], SMA_SLOW)
        cur_fast  = _sma(closes,      SMA_FAST)
        cur_slow  = _sma(closes,      SMA_SLOW)

        sig = _cross(prev_fast, prev_slow, cur_fast, cur_slow)  # +1 / -1 / 0

        last_closed_ts = int(times[-1])  # метка последней закрытой свечи
        if CONFIRM_ON_CLOSE:
            # чтобы не открывать несколько раз на одной свече:
            if _LAST_CANDLE_TS.get(symbol) == last_closed_ts and _LAST_SIGNAL.get(symbol) == sig:
                # этот сигнал уже обработан для этой свечи
                return

        LOG.debug("[auto] %s %s SMA(%d/%d): prev(%.4f/%.4f) -> cur(%.4f/%.4f) => sig=%d (ts=%s)",
                  symbol, TIMEFRAME, SMA_FAST, SMA_SLOW,
                  prev_fast or -1, prev_slow or -1, cur_fast or -1, cur_slow or -1,
                  sig, last_closed_ts)

        # Обновим метки сразу, чтобы при быстрых повторах не задвоить
        _LAST_CANDLE_TS[symbol] = last_closed_ts
        _LAST_SIGNAL[symbol] = sig

        # === Покупка по «золотому кресту» ===
        if sig == +1:
            equity = await _fetch_equity(client)
            if equity < max(EQUITY_MIN_USDT, QUOTE_USDT) and not (USE_RISK_AUTOSIZE and SL_PCT > 0):
                LOG.warning("[auto] %s — equity_usdt=%.2f < требуемого (%.2f). Пропуск.",
                            symbol, equity, max(EQUITY_MIN_USDT, QUOTE_USDT))
                return

            if DRY_RUN:
                LOG.info("[auto][dry] BUY %s %s на ~%.2f USDT (equity %.2f, sl=%.3f tp=%.3f autosize=%s)",
                         symbol, TIMEFRAME, QUOTE_USDT, equity, SL_PCT, TP_PCT, USE_RISK_AUTOSIZE)
                return

            res = await _place_market_buy(client, symbol, QUOTE_USDT)
            LOG.info("[auto] BUY %s %s на ~%.2f USDT -> %s (order_id=%s)",
                     symbol, TIMEFRAME, QUOTE_USDT, res.get("status"), res.get("order_id"))
            return  # на этой свече больше ничего не делаем

        # === Продажа по «death cross» (опционально) ===
        if sig == -1 and SELL_ON_DEATH:
            free_base = await _fetch_free_base_asset(client, symbol)
            sell_qty = max(0.0, free_base * SELL_PCT_OF_POSITION)
            if sell_qty <= 0:
                return
            if DRY_RUN:
                LOG.info("[auto][dry] SELL %s qty≈%.8f (%.0f%% of free %.8f)",
                         symbol, sell_qty, SELL_PCT_OF_POSITION * 100.0, free_base)
                return
            res = await _place_market_sell(client, symbol, sell_qty)
            LOG.info("[auto] SELL %s qty≈%.8f -> %s (order_id=%s)",
                     symbol, sell_qty, res.get("status"), res.get("order_id"))

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
        "[auto] старт: symbols=%s, tf=%s, fast=%d, slow=%d, quote=%.2f, loop=%ds, testnet=%s, dry=%s, sell_on_death=%s",
        ",".join(SYMBOLS), TIMEFRAME, SMA_FAST, SMA_SLOW, QUOTE_USDT, LOOP_SEC, TESTNET, DRY_RUN, SELL_ON_DEATH
    )
    if USE_RISK_AUTOSIZE and SL_PCT <= 0:
        LOG.warning("[auto] USE_RISK_AUTOSIZE включен, но SL_PCT=0 — сервер не сможет рассчитать qty по риску.")

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
