from __future__ import annotations

import os
import time
import logging
from typing import Iterable, Dict, List, Optional, Tuple

import requests

from .base import SourceError

# ------------------------------------------------------------------------------
# Конфигурация
# ------------------------------------------------------------------------------
_API = "https://www.alphavantage.co/query"
_LOG = logging.getLogger("ai_trader.alpha_vantage")

# Поддерживаемые таймфреймы -> (function, interval)
# Нативного 4h у Alpha Vantage нет — агрегируем из 60min.
_AV_TF = {
    "1m": ("TIME_SERIES_INTRADAY", "1min"),
    "5m": ("TIME_SERIES_INTRADAY", "5min"),
    "15m": ("TIME_SERIES_INTRADAY", "15min"),
    "30m": ("TIME_SERIES_INTRADAY", "30min"),
    "1h": ("TIME_SERIES_INTRADAY", "60min"),
    "4h": None,  # будет агрегироваться из 1h
    "1d": ("TIME_SERIES_DAILY", None),
}

# Ретраи/таймауты
_REQ_TIMEOUT = 30  # сек
_MAX_RETRIES = 4
_BACKOFF_BASE = 1.2  # 1.2s, 2.4s, 4.8s, ...

# Признаки сообщений о лимитах/ошибках
_LIMIT_SIGNS = (
    "Thank you for using Alpha Vantage",
    "Our standard API call frequency",
    "Please visit https://www.alphavantage.co/premium",
)
_ERR_KEYS = ("Error Message", "Information", "Note")

# ------------------------------------------------------------------------------
# Простой in-memory кэш (TTL)
# ------------------------------------------------------------------------------
_CACHE_TTL = int(os.getenv("AV_CACHE_TTL", "120"))  # секунд
# ключ -> (ts_put, data)
_CACHE: dict[Tuple, Tuple[float, List[Dict]]] = {}


def _cache_get(key: Tuple) -> Optional[List[Dict]]:
    if _CACHE_TTL <= 0:
        return None
    item = _CACHE.get(key)
    if not item:
        return None
    ts_put, data = item
    if (time.time() - ts_put) <= _CACHE_TTL:
        return data
    # протухло
    _CACHE.pop(key, None)
    return None


def _cache_set(key: Tuple, data: List[Dict]) -> None:
    if _CACHE_TTL <= 0:
        return
    _CACHE[key] = (time.time(), data)


# ------------------------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------------------------
def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _sleep_backoff(attempt: int) -> None:
    delay = _BACKOFF_BASE * (2 ** attempt)
    time.sleep(delay)


def _parse_series_key(payload: dict) -> Optional[str]:
    for k in payload.keys():
        if "Time Series" in k:
            return k
    return None


def _parse_ts_to_epoch(ts_str: str) -> int:
    if " " in ts_str:
        ts_struct = time.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    else:
        ts_struct = time.strptime(ts_str, "%Y-%m-%d")
    return int(time.mktime(ts_struct))


def _request_with_retries(params: Dict[str, str]) -> dict:
    last_exc: Optional[Exception] = None
    with requests.Session() as s:
        for attempt in range(_MAX_RETRIES):
            try:
                r = s.get(_API, params=params, timeout=_REQ_TIMEOUT)
                if 200 <= r.status_code < 300:
                    data = r.json()

                    if any(k in data for k in _ERR_KEYS) or any(sig in str(data) for sig in _LIMIT_SIGNS):
                        if attempt < _MAX_RETRIES - 1:
                            _LOG.warning(
                                "Alpha Vantage throttled/info, retry %s: %s",
                                attempt + 1, str(data)[:200]
                            )
                            _sleep_backoff(attempt)
                            continue
                        msg = data.get("Error Message") or data.get("Information") or data.get("Note") or "Unknown AV error"
                        raise SourceError(f"Alpha Vantage error: {msg}")

                    return data

                if r.status_code >= 500 or r.status_code == 429:
                    _LOG.warning("Alpha Vantage HTTP %s, retry %s", r.status_code, attempt + 1)
                    _sleep_backoff(attempt)
                    continue

                raise SourceError(f"Alpha Vantage HTTP {r.status_code}: {r.text[:200]}")

            except requests.RequestException as e:
                last_exc = e
                _LOG.warning("Network error to Alpha Vantage, retry %s: %r", attempt + 1, e)
                _sleep_backoff(attempt)
                continue

    if last_exc:
        raise SourceError(f"Alpha Vantage request failed after retries: {last_exc!r}")
    raise SourceError("Alpha Vantage request failed after retries (unknown error)")


def _fetch_av_series_raw(
    symbol: str,
    timeframe: str,
    ts_from: Optional[int],
    ts_to: Optional[int],
) -> List[Dict]:
    """
    Базовая загрузка данных AV для нативных ТФ (без limit).
    Результат кэшируется по (symbol, timeframe, ts_from, ts_to).
    """
    key = os.getenv("ALPHAVANTAGE_KEY", "")
    if not key:
        raise SourceError("ALPHAVANTAGE_KEY не задан в configs/.env")

    meta = _AV_TF.get(timeframe)
    if not meta or meta[0] is None:
        raise SourceError(f"timeframe '{timeframe}' не поддерживается Alpha Vantage напрямую")

    # Кэш-попытка
    cache_key = ("series", symbol, timeframe, ts_from, ts_to)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    function, interval = meta
    params = {
        "function": function,
        "symbol": symbol,
        "apikey": key,
        "datatype": "json",
        "outputsize": "full",
    }
    if function == "TIME_SERIES_INTRADAY":
        params["interval"] = interval  # '1min'..'60min'

    payload = _request_with_retries(params)
    series_key = _parse_series_key(payload)
    if not series_key or series_key not in payload:
        raise SourceError(f"Не удалось найти временной ряд в ответе AV: {str(payload)[:200]}")

    series = payload[series_key]
    items: List[Dict] = []

    for ts_str in sorted(series.keys()):
        ts = _parse_ts_to_epoch(ts_str)
        if ts_from and ts < ts_from:
            continue
        if ts_to and ts > ts_to:
            continue

        row = series[ts_str]
        o = row.get("1. open") or row.get("1. Open")
        h = row.get("2. high") or row.get("2. High")
        l = row.get("3. low") or row.get("3. Low")
        c = row.get("4. close") or row.get("4. Close")
        v = row.get("5. volume") or row.get("5. Volume") or 0.0

        items.append(
            {
                "source": "alpha_vantage",
                "asset": symbol,
                "tf": timeframe,
                "ts": int(ts),
                "open": _safe_float(o),
                "high": _safe_float(h),
                "low": _safe_float(l),
                "close": _safe_float(c),
                "volume": _safe_float(v),
            }
        )

    items.sort(key=lambda r: r["ts"])
    _cache_set(cache_key, items)
    return items


# ------------------------------------------------------------------------------
# Публичная функция
# ------------------------------------------------------------------------------
def fetch(
    symbol: str,
    timeframe: str,
    limit: int | None,
    ts_from: int | None,
    ts_to: int | None,
) -> Iterable[Dict]:
    """
    Возвращает нормализованный список свечей dict.
    Кэширует базовую серию (а для 4h — агрегированную) на AV_CACHE_TTL секунд.
    """
    # Особый случай: 4h — агрегируем из 1h
    if timeframe == "4h":
        # пробуем кэш агрегата
        cache_key = ("series_agg4h", symbol, ts_from, ts_to)
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached[-int(limit):] if limit else cached

        base = _fetch_av_series_raw(symbol, "1h", ts_from, ts_to)
        if not base:
            return []

        BUCKET_SEC = 4 * 60 * 60  # 14400
        buckets: Dict[int, Dict] = {}
        for r in base:
            bts = (int(r["ts"]) // BUCKET_SEC) * BUCKET_SEC
            g = buckets.get(bts)
            if g is None:
                buckets[bts] = {
                    "source": "alpha_vantage",
                    "asset": symbol,
                    "tf": "4h",
                    "ts": bts,
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "volume": float(r["volume"]),
                }
            else:
                g["high"] = max(g["high"], float(r["high"]))
                g["low"] = min(g["low"], float(r["low"]))
                g["close"] = float(r["close"])
                g["volume"] += float(r["volume"])

        items = list(buckets.values())
        items.sort(key=lambda x: x["ts"])
        _cache_set(cache_key, items)
        return items[-int(limit):] if limit else items

    # Нативно поддерживаемые ТФ
    full = _fetch_av_series_raw(symbol, timeframe, ts_from, ts_to)
    return full[-int(limit):] if limit else full
