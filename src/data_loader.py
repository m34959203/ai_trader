# src/data_loader.py
from __future__ import annotations

"""
Единый загрузчик OHLCV для AI-Trader.

Поддержка источников:
  • "ccxt"          — криптобиржи (binance/bybit/okx)
  • "yfinance"      — акции/ETF/FX (без ключей)
  • "stooq"         — ежедневные бары (без ключей)
  • "alphavantage"  — intraday по ключу ALPHAVANTAGE_KEY

Публичный API (стабильный):
  get_prices(
      source: "ccxt"|"yfinance"|"stooq"|"alphavantage" = "ccxt",
      symbol: Optional[str] = None,
      timeframe: str = "1h",
      limit: int = 200,
      ticker: Optional[str] = None,
      interval: str = "1h",
      period: str = "7d",
      exchange_name: Optional[str] = None,
  ) -> pd.DataFrame

Гарантируется возврат DataFrame с колонками:
  ["timestamp(UTC)", "open", "high", "low", "close", "volume", ("adj_close"?)]
— timestamp всегда UTC-aware; числа — float.

Особенности:
  • Надёжная загрузка .env из configs/.env или .env (с защитой от BOM)
  • Везде дружелюбные сообщения об ошибках
  • Мягкие сетевые ретраи на сетевых источниках
  • Маленький LRU-кэш на 60 сек для однотипных запросов
"""

import os
import time
from typing import Literal, Optional, Tuple, Callable, Any
from datetime import datetime, timedelta
from functools import lru_cache, wraps
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv, dotenv_values


__all__ = [
    "get_prices",
    "fetch_crypto_ohlcv_ccxt",
    "fetch_ohlc_yfinance",
    "fetch_stooq",
    "fetch_alpha_vantage",
]

# ──────────────────────────────────────────────────────────────────────────────
# Надёжная загрузка .env (ищем configs/.env, затем .env; чиним BOM/пробелы)
# ──────────────────────────────────────────────────────────────────────────────

def _load_env_robust() -> str:
    """
    Возвращает реальный путь к загруженному .env (или пустую строку).
    Загружает переменные и вручную корректирует возможный BOM в ключах.
    """
    base = Path(__file__).resolve().parents[1]  # .../ai_trader
    candidates = [base / "configs" / ".env", base / ".env"]
    for p in candidates:
        if p.exists():
            load_dotenv(p, override=True, encoding="utf-8")
            vals = dotenv_values(p, encoding="utf-8")
            for k, v in vals.items():
                if k is None or v is None:
                    continue
                k = k.strip().lstrip("\ufeff")  # защитимся от BOM
                os.environ[k] = v
            return str(p)
    return ""


ENV_PATH = _load_env_robust()


# ──────────────────────────────────────────────────────────────────────────────
# Общие утилиты
# ──────────────────────────────────────────────────────────────────────────────

class DataSourceError(RuntimeError):
    """Единый тип ошибки загрузчика с дружелюбным сообщением."""


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводим к колонкам: timestamp, open, high, low, close, volume (+ adj_close).
    timestamp → pandas datetime (UTC). Сортировка и очистка NA/дублей.
    """
    if df is None or df.empty:
        return _empty_df()

    # 1) Переименования и эвристики
    cols_lower = {c.lower(): c for c in df.columns}
    renames: dict[str, str] = {}

    if "timestamp" not in cols_lower:
        for cand in ("Datetime", "datetime", "Date", "date", "time"):
            if cand in df.columns:
                renames[cand] = "timestamp"
                break

    for k in ("Open", "High", "Low", "Close", "Volume", "Adj Close"):
        if k in df.columns:
            renames[k] = k.lower().replace(" ", "_")

    if renames:
        df = df.rename(columns=renames)

    cols_lower = {c.lower(): c for c in df.columns}
    # 2) Базовая валидация
    need = ["timestamp", "open", "high", "low", "close"]
    if not all(c in cols_lower for c in need):
        # Не OHLC — отдадим пустое (вызвавший код корректно обработает)
        return _empty_df()

    # 3) Обеспечим volume (если нет — создадим 0)
    if "volume" not in cols_lower:
        df["volume"] = 0.0
        cols_lower = {c.lower(): c for c in df.columns}

    out_cols = ["timestamp", "open", "high", "low", "close", "volume"]
    has_adj = "adj_close" in cols_lower
    if has_adj:
        out_cols.append("adj_close")

    out = df[[cols_lower[c] for c in out_cols]].copy()

    # 4) timestamp → UTC
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # 5) Числовые типы
    for c in [c for c in out.columns if c != "timestamp"]:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")

    out = out.dropna(subset=["open", "high", "low", "close"])
    # Удалим дубликаты времени (берём последнее)
    out = out[~out["timestamp"].duplicated(keep="last")].reset_index(drop=True)
    return out


def _with_retries(tries: int = 3, backoff: float = 0.6) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Простой декоратор ретраев для сетевых вызовов.
    Повторяет при исключениях: DataSourceError/requests/httpx/общие сетевые.
    """
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_err: Optional[Exception] = None
            for attempt in range(1, max(1, tries) + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_err = e
                    # Первый промах — быстрый повтор, далее — backoff
                    if attempt < tries:
                        time.sleep(backoff * (2 ** (attempt - 1)))
                    else:
                        raise
            raise last_err  # не достигнется
        return wrapper
    return deco


# ──────────────────────────────────────────────────────────────────────────────
# CCXT (крипторынки)
# ──────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=16)
def _ccxt_exchange_cached(exchange_name: str):
    import ccxt  # noqa: WPS433
    exchange_name = exchange_name.lower().strip()
    if not hasattr(ccxt, exchange_name):
        raise DataSourceError(f"CCXT: неизвестная биржа '{exchange_name}'")
    exchange_cls = getattr(ccxt, exchange_name)

    params = {"enableRateLimit": True}
    http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    if http_proxy or https_proxy:
        params["proxies"] = {}
        if http_proxy:
            params["proxies"]["http"] = http_proxy
        if https_proxy:
            params["proxies"]["https"] = https_proxy

    ex = exchange_cls(params)
    ex.load_markets()
    return ex


def _validate_ccxt_timeframe(ex, timeframe: str) -> str:
    tf = (timeframe or "1h").strip()
    # Многие биржи предоставляют список поддерживаемых таймфреймов
    try:
        supported = getattr(ex, "timeframes", None)
        if isinstance(supported, dict) and supported:
            if tf not in supported:
                # подберём ближайший популярный
                candidates = ("1m", "5m", "15m", "30m", "1h", "4h", "1d")
                for c in candidates:
                    if c in supported:
                        return c
                # если нет — берём первый доступный
                return next(iter(supported.keys()))
        return tf
    except Exception:
        return tf


@_with_retries(tries=3, backoff=0.7)
def fetch_crypto_ohlcv_ccxt(
    exchange_name: Literal["binance", "bybit", "okx"] = "binance",
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    limit: int = 200,
) -> pd.DataFrame:
    """
    Получение OHLCV через CCXT. Возвращает нормализованный DataFrame.
    """
    if not symbol:
        raise DataSourceError("CCXT: параметр 'symbol' обязателен (например, 'BTC/USDT').")
    try:
        exchange = _ccxt_exchange_cached(exchange_name)
        tf = _validate_ccxt_timeframe(exchange, timeframe)
        raw = exchange.fetch_ohlcv(symbol, timeframe=tf, limit=int(limit))
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        # CCXT timestamps → ms epoch
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("UTC")
        return _normalize_df(df)
    except Exception as e:
        raise DataSourceError(
            f"CCXT fetch failed: exchange={exchange_name}, symbol={symbol}, tf={timeframe}, limit={limit}. "
            f"Причина: {e}"
        ) from e


# ──────────────────────────────────────────────────────────────────────────────
# YFinance (акции/ETF/FX) — с проверками и фолбэком
# ──────────────────────────────────────────────────────────────────────────────

_YF_OK: dict[str, set[str]] = {
    "1d": {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d"},
    "5d": {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d"},
    "7d": {"5m", "15m", "30m", "60m", "90m", "1h", "1d"},
    "1mo": {"30m", "60m", "90m", "1h", "1d"},
    "3mo": {"1h", "1d"},
    "6mo": {"1d"},
    "1y": {"1d"},
    "2y": {"1d"},
    "5y": {"1d"},
    "10y": {"1d"},
    "ytd": {"1d"},
    "max": {"1d"},
}


def _normalize_period_interval(period: str, interval: str) -> Tuple[str, str]:
    p = (period or "").lower()
    i = (interval or "").lower()
    if p in _YF_OK and i in _YF_OK[p]:
        return p, i
    # полезные фолбэки
    if i in {"60m", "90m", "1h"}:
        return ("1mo", "1h")
    if i == "1d" and p not in _YF_OK:
        return ("7d", "1d")
    return ("1mo", "1d")


@_with_retries(tries=2, backoff=0.8)
def fetch_ohlc_yfinance(
    ticker: str = "AAPL",
    interval: str = "1h",
    period: str = "7d",
) -> pd.DataFrame:
    if not ticker:
        raise DataSourceError("YFinance: параметр 'ticker' обязателен (напр., 'AAPL', 'MSFT', 'EURUSD=X').")
    try:
        import yfinance as yf  # noqa: WPS433
    except Exception as e:
        raise DataSourceError("YFinance не установлен (pip install yfinance).") from e

    p1, i1 = (period.lower(), interval.lower())
    p2, i2 = _normalize_period_interval(period, interval)

    for (pp, ii) in {(p1, i1), (p2, i2)}:
        data = yf.download(
            tickers=ticker,
            interval=ii,
            period=pp,
            progress=False,
            auto_adjust=False,
            threads=True,
        )
        if not data.empty:
            df = data.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Adj Close": "adj_close",
                    "Volume": "volume",
                }
            ).reset_index()

            if "Datetime" in df.columns:
                df = df.rename(columns={"Datetime": "timestamp"})
            elif "Date" in df.columns:
                df = df.rename(columns={"Date": "timestamp"})

            return _normalize_df(df)

    # пусто — корректный случай
    return _empty_df()


# ──────────────────────────────────────────────────────────────────────────────
# STOOQ (без ключей) — дневные бары
# ──────────────────────────────────────────────────────────────────────────────

@_with_retries(tries=2, backoff=0.5)
def fetch_stooq(ticker: str = "AAPL", days: int = 60) -> pd.DataFrame:
    if not ticker:
        raise DataSourceError("Stooq: параметр 'ticker' обязателен (напр., 'AAPL').")
    try:
        import pandas_datareader.data as web  # noqa: WPS433
    except Exception as e:
        # не критично — вернём пусто
        return _empty_df()

    try:
        end = datetime.utcnow()
        start = end - timedelta(days=int(days))
        df = web.DataReader(ticker, "stooq", start, end)
        if df.empty:
            return _empty_df()
        df = (
            df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
              .reset_index()
              .rename(columns={"Date": "timestamp"})
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return _normalize_df(df)
    except Exception:
        return _empty_df()


# ──────────────────────────────────────────────────────────────────────────────
# Alpha Vantage (ALPHAVANTAGE_KEY обязателен)
# ──────────────────────────────────────────────────────────────────────────────

def _to_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return 0.0


@_with_retries(tries=2, backoff=1.0)
def fetch_alpha_vantage(
    ticker: str = "AAPL",
    interval: Literal["1min", "5min", "15min", "30min", "60min"] = "60min",
    outputsize: Literal["compact", "full"] = "compact",
) -> pd.DataFrame:
    if not ticker:
        raise DataSourceError("AlphaVantage: параметр 'ticker' обязателен (напр., 'AAPL').")

    api_key = os.getenv("ALPHAVANTAGE_KEY")
    if not api_key:
        where = ENV_PATH or "<not found>"
        raise DataSourceError(f"ALPHAVANTAGE_KEY не задан (env loaded from: {where})")

    try:
        import requests  # noqa: WPS433
    except Exception as e:
        raise DataSourceError("Отсутствует библиотека 'requests' (pip install requests).") from e

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": ticker,
        "interval": interval,
        "apikey": api_key,
        "datatype": "json",
        "outputsize": outputsize,
    }
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        raise DataSourceError(f"Alpha Vantage: сетевой сбой — {e}") from e

    if "Note" in data:
        # Лимиты API — это не фатальная ошибка для приложения, но пользователю важно знать
        raise DataSourceError(f"Alpha Vantage API limit: {data['Note']}")
    if "Error Message" in data:
        raise DataSourceError(f"Alpha Vantage error: {data['Error Message']}")

    series_key = next((k for k in data.keys() if "Time Series" in k), None)
    if not series_key or not isinstance(data.get(series_key), dict):
        raise DataSourceError(f"Alpha Vantage: нет таймсерии в ответе. Ключи: {list(data.keys())}")

    rows = []
    for ts, row in data[series_key].items():
        rows.append({
            "timestamp": pd.to_datetime(ts, utc=True, errors="coerce"),
            "open": _to_float(row.get("1. open", "0")),
            "high": _to_float(row.get("2. high", "0")),
            "low": _to_float(row.get("3. low", "0")),
            "close": _to_float(row.get("4. close", "0")),
            "volume": _to_float(row.get("5. volume", "0")),
        })

    if not rows:
        return _empty_df()

    df = pd.DataFrame(rows)
    return _normalize_df(df)


# ──────────────────────────────────────────────────────────────────────────────
# Небольшой кэш на 60 секунд для повторяющихся запросов
# ──────────────────────────────────────────────────────────────────────────────

_CACHE_TTL_SEC = int(os.getenv("DATA_LOADER_CACHE_TTL", "60"))
_cache_store: dict[tuple, tuple[float, pd.DataFrame]] = {}


def _cache_key(*, source: str, symbol: Optional[str], timeframe: str, limit: int,
               ticker: Optional[str], interval: str, period: str, exchange_name: Optional[str]) -> tuple:
    return (
        source.lower().strip(),
        (symbol or "").upper(),
        timeframe.lower().strip(),
        int(limit),
        (ticker or "").upper(),
        interval.lower().strip(),
        period.lower().strip(),
        (exchange_name or "").lower().strip(),
    )


def _cache_get(key: tuple) -> Optional[pd.DataFrame]:
    ttl = max(0, _CACHE_TTL_SEC)
    if ttl == 0:
        return None
    hit = _cache_store.get(key)
    if not hit:
        return None
    ts, df = hit
    if (time.time() - ts) > ttl:
        _cache_store.pop(key, None)
        return None
    # отдаём копию, чтобы не мутировали кэш случайно
    return df.copy()


def _cache_put(key: tuple, df: pd.DataFrame) -> None:
    ttl = max(0, _CACHE_TTL_SEC)
    if ttl == 0:
        return
    _cache_store[key] = (time.time(), df.copy())


# ──────────────────────────────────────────────────────────────────────────────
# Унифицированная обёртка
# ──────────────────────────────────────────────────────────────────────────────

def get_prices(
    source: Literal["ccxt", "yfinance", "stooq", "alphavantage"] = "ccxt",
    symbol: Optional[str] = None,
    timeframe: str = "1h",
    limit: int = 200,
    ticker: Optional[str] = None,
    interval: str = "1h",
    period: str = "7d",
    exchange_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Единая точка входа для загрузки OHLCV из разных источников.
    Возвращает DataFrame с колонками: timestamp(UTC), open, high, low, close, volume (+ adj_close).
    Исключения бросаются как DataSourceError с понятными сообщениями.
    """
    src = (source or os.getenv("DATA_SOURCE", "ccxt")).lower().strip()
    key = _cache_key(
        source=src, symbol=symbol, timeframe=timeframe, limit=limit,
        ticker=ticker, interval=interval, period=period, exchange_name=exchange_name,
    )
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        if src == "ccxt":
            ex = (exchange_name or os.getenv("EXCHANGE", "binance")).lower().strip()
            sym = (symbol or os.getenv("CCXT_SYMBOL", "BTC/USDT")).strip()
            df = fetch_crypto_ohlcv_ccxt(exchange_name=ex, symbol=sym, timeframe=timeframe, limit=limit)

        elif src == "yfinance":
            tkr = (ticker or os.getenv("YF_DEFAULT_TICKER", "AAPL")).strip()
            df = fetch_ohlc_yfinance(ticker=tkr, interval=interval, period=period)

        elif src == "stooq":
            tkr = (ticker or "AAPL").strip()
            df = fetch_stooq(tkr, days=60)

        elif src == "alphavantage":
            tkr = (ticker or "AAPL").strip()
            iv_map = {"1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
                      "60m": "60min", "1h": "60min"}
            iv = iv_map.get(interval.lower().strip(), "60min")
            df = fetch_alpha_vantage(tkr, interval=iv, outputsize="compact")

        else:
            # безопасный дефолт — yfinance
            tkr = (ticker or os.getenv("YF_DEFAULT_TICKER", "AAPL")).strip()
            df = fetch_ohlc_yfinance(ticker=tkr, interval=interval, period=period)

    except DataSourceError:
        # пробрасываем как есть
        raise
    except Exception as e:
        # оборачиваем все остальные как дружественный DataSourceError
        raise DataSourceError(str(e)) from e

    _cache_put(key, df)
    return df
