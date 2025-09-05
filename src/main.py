# src/main.py
from __future__ import annotations

import json
import os
import time
import logging
import asyncio
from typing import Literal, Optional, Any, Dict, List
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response, PlainTextResponse

# ──────────────────────────────────────────────────────────────────────────────
# Локальные импорты
# ──────────────────────────────────────────────────────────────────────────────
# Загрузка котировок (единый интерфейс)
from .data_loader import get_prices

# Аналитика (Спринт 2)
try:
    from .analysis.analyze_market import analyze_market, DEFAULT_CONFIG, AnalysisConfig  # type: ignore
except Exception:  # pragma: no cover
    analyze_market = None  # type: ignore
    DEFAULT_CONFIG = None  # type: ignore
    AnalysisConfig = None  # type: ignore

# БД (схема)
from db import models          # noqa: F401  # регистрирует таблицы OHLCV
from db import models_orders   # noqa: F401  # регистрирует таблицу orders
from db.session import engine, Base

# Роутеры
from routers.ohlcv import router as ohlcv_router            # /prices/store, /ohlcv, /ohlcv.csv, /ohlcv/count, /ohlcv/stats
from routers.trading import router as trading_router        # /strategy/*, /paper/*
from routers.trading_exec import router as exec_router      # /exec/*

# UI (опционально)
try:
    from routers.ui import router as ui_router              # /ui/*
    _HAS_UI = True
except Exception:  # pragma: no cover
    _HAS_UI = False

# Сверка (опционально)
try:
    from services.reconcile import reconcile_positions, reconcile_periodic  # type: ignore
    _HAS_RECONCILE = True
except Exception:  # pragma: no cover
    _HAS_RECONCILE = False

# Потоки рынка (опционально)
try:
    from services.stream_router import StreamRouter, StreamConfig  # type: ignore
    from sources.binance_ws import BinanceWS  # type: ignore
    from sources.binance import BinanceREST   # type: ignore
    _HAS_STREAMS = True
except Exception:  # pragma: no cover
    _HAS_STREAMS = False

APP_VERSION = os.getenv("APP_VERSION", "0.10.2")

# ──────────────────────────────────────────────────────────────────────────────
# ENV utils
# ──────────────────────────────────────────────────────────────────────────────
def env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "on"}

def env_list(name: str, default: str = "", sep: str = ",") -> List[str]:
    raw = os.getenv(name, default)
    if not raw:
        return []
    return [item.strip() for item in raw.split(sep) if item.strip()]

def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

# ──────────────────────────────────────────────────────────────────────────────
# Логирование (дружим с uvicorn)
# ──────────────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | ai_trader | %(name)s | %(message)s",
)
LOG = logging.getLogger("ai_trader.app")

# ──────────────────────────────────────────────────────────────────────────────
# Флаги/конфиг
# ──────────────────────────────────────────────────────────────────────────────
FEATURES: Dict[str, bool] = {
    "ohlcv_storage": env_bool("FEATURE_OHLCV", True),
    "signals":       env_bool("FEATURE_SIGNALS", True),
    "paper_trading": env_bool("FEATURE_PAPER", True),
    "execution_api": env_bool("FEATURE_EXEC", True),
    "ui":            env_bool("FEATURE_UI", True) and _HAS_UI,
    "market_ws":     env_bool("FEATURE_MARKET_WS", True) and _HAS_STREAMS,
    "user_ws":       env_bool("FEATURE_USER_WS", True) and _HAS_STREAMS,
}

# Binance / WS окружение
BINANCE_API_KEY     = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET  = os.getenv("BINANCE_API_SECRET", "")
BINANCE_TESTNET     = env_bool("BINANCE_TESTNET", True)

MARKET_WS_SYMBOLS   = env_list("MARKET_WS_SYMBOLS", "BTCUSDT")
MARKET_WS_INTERVALS = env_list("MARKET_WS_INTERVALS", "1m,5m")
MARKET_BACKFILL_N   = env_int("MARKET_BACKFILL_BARS", 1000)
MARKET_REST_POLL    = env_float("MARKET_REST_POLL_SEC", 2.0)

# Watchdog/Keepalive
WS_HEARTBEAT_SEC    = env_int("WS_HEARTBEAT_SEC", 90)

# Доп. флаги сверки
RECONCILE_AT_START   = env_bool("RECONCILE_AT_START", False)
RECONCILE_AUTO_FIX   = env_bool("RECONCILE_AUTO_FIX", False)
RECONCILE_MODE       = os.getenv("RECONCILE_MODE", "binance")
RECONCILE_TESTNET    = env_bool("RECONCILE_TESTNET", True)
RECONCILE_PERIODIC   = env_bool("RECONCILE_PERIODIC", False)
RECONCILE_INT_SEC    = env_int("RECONCILE_INTERVAL_SEC", 300)
RECONCILE_ABS_TOL    = env_float("RECONCILE_ABS_TOL", 1e-8)
RECONCILE_JOURNAL_N  = env_int("RECONCILE_JOURNAL_LIMIT", 500)
RECONCILE_SYMBOLS    = os.getenv("RECONCILE_SYMBOLS")

# ──────────────────────────────────────────────────────────────────────────────
# SQLite PRAGMA (ENV)
# ──────────────────────────────────────────────────────────────────────────────
PRAGMA_WAL         = env_bool("SQLITE_WAL", True)
PRAGMA_SYNC_NORMAL = env_bool("SQLITE_SYNC_NORMAL", True)
PRAGMA_FK          = env_bool("SQLITE_FOREIGN_KEYS", True)

# ──────────────────────────────────────────────────────────────────────────────
# Lifespan: PRAGMA + schema + сверка + WS-задачи
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    async with engine.begin() as conn:
        if PRAGMA_WAL:
            await conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
        if PRAGMA_SYNC_NORMAL:
            await conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
        if PRAGMA_FK:
            await conn.exec_driver_sql("PRAGMA foreign_keys=ON;")
        await conn.run_sync(Base.metadata.create_all)

    # Разовая сверка при старте (если включена и доступен модуль)
    if RECONCILE_AT_START and _HAS_RECONCILE and FEATURES.get("execution_api", False):
        try:
            res = await reconcile_positions(
                mode=RECONCILE_MODE,
                testnet=RECONCILE_TESTNET,
                auto_fix=RECONCILE_AUTO_FIX,
                abs_tol=RECONCILE_ABS_TOL,
                journal_limit=RECONCILE_JOURNAL_N,
                symbols=[s.strip().upper() for s in RECONCILE_SYMBOLS.split(",")] if RECONCILE_SYMBOLS else None,
            )
            LOG.info("Reconcile at start -> ok=%s mismatches=%s",
                     bool(res.get("ok")), len(res.get("mismatches", [])))
        except Exception as e:  # pragma: no cover
            LOG.warning("Reconcile at start failed: %r", e)

    # Фоновые задачи
    app.state.reconcile_task = None
    app.state.market_ws_task = None
    app.state.user_ws_task   = None
    app.state.keepalive_task = None

    # Объекты потоков
    app.state.stream_router = None
    app.state.user_ws = None
    app.state.binance_rest = None

    # Runtime-состояние для watchdog
    app.state.market_last_event_ts = 0  # unix sec
    app.state.user_last_event_ts   = 0  # unix sec
    app.state.market_ring: List[Dict[str, Any]] = []

    # Периодическая сверка
    if RECONCILE_PERIODIC and _HAS_RECONCILE and FEATURES.get("execution_api", False):
        try:
            app.state.reconcile_task = asyncio.create_task(
                reconcile_periodic(
                    interval_sec=max(5, int(RECONCILE_INT_SEC)),
                    mode=("sim" if RECONCILE_MODE.strip().lower() == "sim" else "binance"),
                    testnet=RECONCILE_TESTNET,
                    auto_fix=RECONCILE_AUTO_FIX,
                    abs_tol=RECONCILE_ABS_TOL,
                    journal_limit=RECONCILE_JOURNAL_N,
                    symbols=[s.strip().upper() for s in RECONCILE_SYMBOLS.split(",")] if RECONCILE_SYMBOLS else None,
                )
            )
            LOG.info("Reconcile periodic task started: every %ds", int(RECONCILE_INT_SEC))
        except Exception as e:  # pragma: no cover
            LOG.warning("Failed to start reconcile periodic task: %r", e)

    # ── MARKET WS TASK ─────────────────────────────────────────────────────
    if FEATURES.get("market_ws", False) and _HAS_STREAMS:
        try:
            cfg = StreamConfig(
                symbols=MARKET_WS_SYMBOLS,
                intervals=MARKET_WS_INTERVALS,
                backfill_bars=max(100, int(MARKET_BACKFILL_N)),
                rest_poll_sec=float(MARKET_REST_POLL),
                testnet=BINANCE_TESTNET,
                with_depth=False,
            )
            app.state.stream_router = StreamRouter(
                cfg,
                api_key=BINANCE_API_KEY or None,
                api_secret=BINANCE_API_SECRET or None,
                logger=logging.getLogger("ai_trader.stream_router"),
            )
            app.state.market_ws_task = asyncio.create_task(_consume_market_stream(app))
            LOG.info("Market WS task started: %s @ %s", MARKET_WS_SYMBOLS, MARKET_WS_INTERVALS)
        except Exception as e:  # pragma: no cover
            LOG.warning("Failed to start Market WS task: %r", e)

    # ── USER WS TASK ───────────────────────────────────────────────────────
    if FEATURES.get("user_ws", False) and _HAS_STREAMS and BINANCE_API_KEY:
        try:
            app.state.user_ws = BinanceWS(
                api_key=BINANCE_API_KEY,
                api_secret=BINANCE_API_SECRET or None,
                testnet=BINANCE_TESTNET,
                logger=logging.getLogger("ai_trader.binance.ws.user"),
            )
            app.state.binance_rest = BinanceREST(
                api_key=BINANCE_API_KEY,
                api_secret=BINANCE_API_SECRET or None,
                testnet=BINANCE_TESTNET,
                logger=logging.getLogger("ai_trader.binance.rest"),
            )
            app.state.user_ws_task = asyncio.create_task(_consume_user_stream(app))
            LOG.info("User WS task started (testnet=%s)", BINANCE_TESTNET)
        except Exception as e:  # pragma: no cover
            LOG.warning("Failed to start User WS task: %r", e)

    # ── KEEPALIVE/WATCHDOG TASK ───────────────────────────────────────────
    if (app.state.market_ws_task or app.state.user_ws_task):
        try:
            app.state.keepalive_task = asyncio.create_task(_ws_watchdog(app, WS_HEARTBEAT_SEC))
            LOG.info("WS watchdog started: heartbeat=%ss", int(WS_HEARTBEAT_SEC))
        except Exception as e:  # pragma: no cover
            LOG.warning("Failed to start WS watchdog: %r", e)

    LOG.info("Startup complete: DB schema ensured, version=%s, features=%s", APP_VERSION, FEATURES)
    try:
        yield
    finally:
        # Shutdown: останавливаем фоновые задачи
        try:
            tasks = [
                getattr(app.state, "keepalive_task", None),
                getattr(app.state, "user_ws_task", None),
                getattr(app.state, "market_ws_task", None),
                getattr(app.state, "reconcile_task", None),
            ]
            for t in tasks:
                if t is not None:
                    t.cancel()
            for t in tasks:
                if t is not None:
                    try:
                        await t
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        LOG.debug("Task finished with error: %r", e)
        except Exception:  # pragma: no cover
            pass

        # Закрытия клиентов/соединений
        try:
            if app.state.stream_router is not None:
                await app.state.stream_router.stop()
        except Exception:
            pass
        try:
            if app.state.user_ws is not None:
                app.state.user_ws.stop()
        except Exception:
            pass
        try:
            if app.state.binance_rest is not None:
                await app.state.binance_rest.aclose()
        except Exception:
            pass
        try:
            await engine.dispose()
        except Exception:  # pragma: no cover
            pass
        LOG.info("Shutdown complete")

# ──────────────────────────────────────────────────────────────────────────────
# Фоновые корутины (market/user/keepalive)
# ──────────────────────────────────────────────────────────────────────────────
async def _consume_market_stream(app: FastAPI) -> None:
    """
    Читает бары из StreamRouter, обновляет last_event_ts и складывает последние N баров в ring.
    """
    assert app.state.stream_router is not None
    RING_MAX = 1024
    async for bar in app.state.stream_router.run():
        app.state.market_last_event_ts = int(time.time())
        try:
            app.state.market_ring.append({
                "asset": bar.asset, "tf": bar.tf, "ts": bar.ts,
                "open": bar.open, "high": bar.high, "low": bar.low, "close": bar.close, "volume": bar.volume,
            })
            if len(app.state.market_ring) > RING_MAX:
                app.state.market_ring = app.state.market_ring[-RING_MAX:]
        except Exception:
            pass
        # Здесь можно: писать OHLCV в БД, дергать обработчики сигналов и т.п.

async def _consume_user_stream(app: FastAPI) -> None:
    """
    Подписка на user data stream (executionReport, account updates). Keepalive внутри BinanceWS.
    """
    assert app.state.user_ws is not None
    async for evt in app.state.user_ws.start_user():
        app.state.user_last_event_ts = int(time.time())
        t = evt.get("type")
        if t == "executionReport":
            data = evt.get("data", {})
            logging.getLogger("ai_trader.exec_report").debug("ExecReport: %s", data)
        # outboundAccountPosition / balanceUpdate / accountUpdate — при необходимости

async def _ws_watchdog(app: FastAPI, heartbeat_sec: int) -> None:
    """
    Логирует лаги по событиям в потоках. Не вмешивается в reconnect (это делает клиент).
    """
    interval = max(5, int(heartbeat_sec // 3))
    while True:
        await asyncio.sleep(interval)
        now = int(time.time())
        if app.state.market_ws_task:
            lag = now - int(getattr(app.state, "market_last_event_ts", 0) or 0)
            if lag > heartbeat_sec:
                LOG.warning("Market WS heartbeat lag: %ss (>%ss)", lag, heartbeat_sec)
        if app.state.user_ws_task:
            lag = now - int(getattr(app.state, "user_last_event_ts", 0) or 0)
            if lag > heartbeat_sec:
                LOG.warning("User WS heartbeat lag: %ss (>%ss)", lag, heartbeat_sec)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_prices_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Превращает колонку 'timestamp' в DatetimeIndex (UTC), как ожидает analyze_market.
    Возвращает None, если входной df пустой/None.
    """
    if df is None or df.empty:
        return None
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    # если уже индекс-датавремя — приведём к UTC
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df.sort_index()
    return df

# ──────────────────────────────────────────────────────────────────────────────
# Приложение
# ──────────────────────────────────────────────────────────────────────────────
docs_disabled = env_bool("DISABLE_DOCS", False)
openapi_url = None if docs_disabled else "/openapi.json"
docs_url = None if docs_disabled else "/docs"
redoc_url = None if docs_disabled else "/redoc"

tags_metadata = [
    {"name": "prices",  "description": "Загрузка котировок из внешних источников (без БД)."},
    {"name": "ohlcv",   "description": "Хранилище свечей и выборка из БД."},
    {"name": "strategy","description": "Индикаторы, сигналы, стратегии."},
    {"name": "paper",   "description": "Бэктест и paper-trading (виртуальные сделки)."},
    {"name": "exec",    "description": "Исполнение ордеров: симулятор, Binance API, UI-агент."},
    {"name": "meta",    "description": "Служебные и диагностические эндпоинты."},
    {"name": "ui",      "description": "Мониторинг и быстрая ручная панель."},
]

app = FastAPI(
    title="AI Trader",
    version=APP_VERSION,
    lifespan=lifespan,
    openapi_tags=tags_metadata,
    openapi_url=openapi_url,
    docs_url=docs_url,
    redoc_url=redoc_url,
)

# ──────────────────────────────────────────────────────────────────────────────
# Middlewares: TrustedHost + CORS + GZip
# ──────────────────────────────────────────────────────────────────────────────
trusted_hosts = env_list("TRUSTED_HOSTS", "")  # пример: "localhost,127.0.0.1,example.com"
if trusted_hosts:
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=trusted_hosts)

extra_origins = env_list("CORS_ORIGINS", "")
default_origins = [
    "http://127.0.0.1",
    "http://127.0.0.1:8001",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8001",
]
allow_origins = list(dict.fromkeys(default_origins + extra_origins))  # без дублей

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600,
)
gzip_min = int(os.getenv("GZIP_MIN_SIZE", "1024"))
app.add_middleware(GZipMiddleware, minimum_size=gzip_min)

# ──────────────────────────────────────────────────────────────────────────────
# Глобальные обработчики ошибок (чистый JSON)
# ──────────────────────────────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        headers={"Cache-Control": "no-store"},
        content={"ok": False, "error": str(exc.detail), "path": request.url.path},
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    LOG.exception("Unhandled exception at %s: %r", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        headers={"Cache-Control": "no-store"},
        content={"ok": False, "error": "Internal Server Error", "path": request.url.path},
    )

# ──────────────────────────────────────────────────────────────────────────────
# META
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/ping", tags=["meta"])
async def ping() -> Dict[str, Any]:
    return {"status": "ok", "message": "AI Trader is running 🚀"}

@app.get("/healthz", tags=["meta"])
async def healthz() -> Dict[str, Any]:
    db_ok = True
    try:
        async with engine.begin() as conn:
            await conn.exec_driver_sql("SELECT 1;")
    except Exception as e:  # pragma: no cover
        LOG.warning("DB health check failed: %r", e)
        db_ok = False
    return {
        "ok": bool(db_ok),
        "service": "ai-trader",
        "version": APP_VERSION,
        "db_ok": db_ok,
        "features": FEATURES,
    }

@app.get("/_livez", tags=["meta"])
async def livez() -> PlainTextResponse:
    return PlainTextResponse("OK")

@app.get("/_readyz", tags=["meta"])
async def readyz() -> PlainTextResponse:
    try:
        async with engine.begin() as conn:
            await conn.exec_driver_sql("SELECT 1;")
        return PlainTextResponse("READY")
    except Exception:  # pragma: no cover
        return PlainTextResponse("NOT_READY", status_code=503)

@app.get("/version", tags=["meta"])
async def version() -> Dict[str, Any]:
    return {"version": APP_VERSION}

# Мини-favicon (1×1 PNG), чтобы не видеть 404
@app.get("/favicon.ico")
def favicon() -> Response:
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0cIDATx\x9cc``\x00\x00\x00\x02\x00\x01"
        b"\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    return Response(content=png_bytes, media_type="image/png", headers={"Cache-Control": "public, max-age=86400"})

@app.get("/robots.txt", tags=["meta"])
def robots() -> PlainTextResponse:
    return PlainTextResponse("User-agent: *\nDisallow:\n", headers={"Cache-Control": "public, max-age=86400"})

# ──────────────────────────────────────────────────────────────────────────────
# /prices: чтение котировок из внешних провайдеров (без БД)
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/prices", tags=["prices"])
async def prices(
    source: Literal["ccxt", "yfinance", "stooq", "alphavantage"] = Query(
        "ccxt", description="Источник данных"
    ),
    # CCXT
    symbol: Optional[str] = Query(None, description="Напр.: BTC/USDT (для ccxt)"),
    timeframe: str = Query("1h", description="1m,5m,15m,1h,4h,1d... (ccxt)"),
    limit: int = Query(200, ge=1, le=5000, description="Количество свечей (ccxt)"),
    exchange_name: Optional[str] = Query(None, description="binance/bybit/okx..."),
    # YFinance / Alpha Vantage / Stooq
    ticker: Optional[str] = Query(None, description="Напр.: AAPL, MSFT, EURUSD=X"),
    interval: str = Query("1h", description="yfinance/alphavantage: 1m..1h; stooq: игнорируется"),
    period: str = Query("7d", description="yfinance: 1d..1mo..; stooq/alphavantage: игнорируется"),
    # Формат ответа
    fmt: Literal["json", "csv"] = Query("json", description="Формат ответа"),
) -> Response:
    """
    Возвращает OHLCV-таблицу из указанного источника в JSON или CSV.
    Индекс/колонка 'timestamp' нормализуется в ISO-UTC строку.
    """
    t0 = time.perf_counter()
    try:
        df = get_prices(
            source=source,
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            ticker=ticker,
            interval=interval,
            period=period,
            exchange_name=exchange_name,
        )
    except Exception as e:
        LOG.exception("GET /prices failed: %r", e)
        raise HTTPException(status_code=400, detail=str(e))

    if df is None or df.empty:
        LOG.info("GET /prices source=%s symbol=%s ticker=%s -> rows=0 in %.3fs",
                 source, symbol, ticker, time.perf_counter() - t0)
        return JSONResponse({"rows": 0, "data": []}, headers={"Cache-Control": "no-store"})

    # Нормализуем timestamp в ISO-строки (UTC)
    if "timestamp" in df.columns:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    rows_count = int(len(df))

    if fmt == "csv":
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        LOG.info("GET /prices CSV source=%s symbol=%s ticker=%s -> rows=%d in %.3fs",
                 source, symbol, ticker, rows_count, time.perf_counter() - t0)
        return StreamingResponse(
            iter([csv_bytes]),
            media_type="text/csv",
            headers={
                "Content-Disposition": 'attachment; filename="prices.csv"',
                "Cache-Control": "no-store",
            },
        )

    # JSON через pandas → стабильная сериализация numpy-типов
    records: List[Dict[str, Any]] = json.loads(df.to_json(orient="records"))
    LOG.info("GET /prices JSON source=%s symbol=%s ticker=%s -> rows=%d in %.3fs",
             source, symbol, ticker, rows_count, time.perf_counter() - t0)
    return JSONResponse({"rows": len(records), "data": records}, headers={"Cache-Control": "no-store"})

# ──────────────────────────────────────────────────────────────────────────────
# /analyze: единый анализ рынка (использует get_prices + analyze_market)
# ──────────────────────────────────────────────────────────────────────────────
if FEATURES["signals"]:
    @app.get("/analyze", tags=["strategy"])
    async def analyze(
        source: Literal["ccxt", "yfinance", "stooq", "alphavantage"] = Query("ccxt"),
        symbol: Optional[str] = Query("BTC/USDT", description="Напр.: BTC/USDT (ccxt)"),
        exchange_name: Optional[str] = Query("binance", description="binance/bybit/okx..."),
        tf_fast: str = Query("1h", description="таймфрейм быстрого анализа"),
        tf_slow: Optional[str] = Query("4h", description="таймфрейм подтверждения (опц.)"),
        limit_fast: int = Query(360, ge=120, le=5000),
        limit_slow: int = Query(240, ge=50, le=5000),
        no_mtf: bool = Query(False, description="true — игнорировать tf_slow"),
        buy_th: Optional[int] = Query(None, ge=1, le=99),
        sell_th: Optional[int] = Query(None, ge=1, le=99),
    ) -> JSONResponse:
        """
        Стандартизованный анализ рынка: быстрый TF + (опционально) медленный TF для подтверждения.
        """
        if analyze_market is None:
            raise HTTPException(status_code=500, detail="Analysis module not available")

        # 1) Загрузка данных
        try:
            df_fast = get_prices(
                source=source,
                symbol=symbol,
                timeframe=tf_fast,
                limit=limit_fast,
                exchange_name=exchange_name,
            )
            df_slow = None
            if not no_mtf and tf_slow:
                df_slow = get_prices(
                    source=source,
                    symbol=symbol,
                    timeframe=tf_slow,
                    limit=limit_slow,
                    exchange_name=exchange_name,
                )
        except Exception as e:
            LOG.exception("GET /analyze: get_prices failed: %r", e)
            raise HTTPException(status_code=400, detail=f"Failed to load prices: {e}")

        if df_fast is None or df_fast.empty:
            raise HTTPException(status_code=400, detail="No data for fast timeframe")

        # 2) Нормализация под ожидания analyze_market → DatetimeIndex(UTC)
        df_fast = _normalize_prices_df(df_fast)
        df_slow = _normalize_prices_df(df_slow) if df_slow is not None else None
        if df_fast is None or not isinstance(df_fast.index, pd.DatetimeIndex):
            raise HTTPException(status_code=400, detail="Analysis assertion: DataFrame index must be a pandas.DatetimeIndex (UTC).")

        # 3) Конфигурация анализа (опциональные пороги)
        cfg = DEFAULT_CONFIG
        if (buy_th is not None or sell_th is not None) and AnalysisConfig is not None and DEFAULT_CONFIG is not None:
            cfg_dict = dict(DEFAULT_CONFIG.__dict__)  # type: ignore
            if buy_th is not None:
                cfg_dict["buy_threshold"] = int(buy_th)
            if sell_th is not None:
                cfg_dict["sell_threshold"] = int(sell_th)
            cfg = AnalysisConfig(**cfg_dict)  # type: ignore

        # 4) Запуск анализа
        t0 = time.perf_counter()
        try:
            result = analyze_market(df_fast, df_slow, config=cfg)  # type: ignore
        except AssertionError as e:
            raise HTTPException(status_code=400, detail=f"Analysis assertion: {e}")
        except Exception as e:
            LOG.exception("GET /analyze failed: %r", e)
            raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

        out: Dict[str, Any] = {
            "exchange": exchange_name,
            "symbol": symbol,
            "tf_fast": tf_fast,
            "tf_slow": (None if no_mtf else tf_slow),
            "fetched_bars_fast": int(len(df_fast)),
            "fetched_bars_slow": (0 if (no_mtf or df_slow is None) else int(len(df_slow))),
            "result": result,
            "elapsed_sec": round(time.perf_counter() - t0, 3),
            "fetched_at": pd.Timestamp.utcnow().isoformat(),
        }
        return JSONResponse(out, headers={"Cache-Control": "no-store"})

# ──────────────────────────────────────────────────────────────────────────────
# Роутеры: OHLCV + Trading + Execution + UI
# ──────────────────────────────────────────────────────────────────────────────
if FEATURES["ohlcv_storage"]:
    app.include_router(ohlcv_router, tags=["ohlcv"])

if FEATURES["signals"] or FEATURES["paper_trading"]:
    app.include_router(trading_router, tags=["strategy", "paper"])

if FEATURES["execution_api"]:
    app.include_router(exec_router, tags=["exec"])

if FEATURES["ui"] and _HAS_UI:
    app.include_router(ui_router, tags=["ui"])

# ──────────────────────────────────────────────────────────────────────────────
# Root index
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["meta"])
async def root() -> Dict[str, Any]:
    endpoints: List[str] = [
        "/",
        "/ping",
        "/healthz",
        "/_livez",
        "/_readyz",
        "/version",
        "/prices",
    ]
    if not docs_disabled:
        endpoints += ["/docs", "/redoc", "/openapi.json"]

    if FEATURES["ohlcv_storage"]:
        endpoints += [
            "/prices/store",
            "/ohlcv",
            "/ohlcv.csv",
            "/ohlcv/count",
            "/ohlcv/stats",
        ]
    if FEATURES["signals"] or FEATURES["paper_trading"]:
        endpoints += [
            "/strategy/signals",
            "/paper/backtest",
        ]
        if FEATURES["signals"]:
            endpoints.append("/analyze")
    if FEATURES["execution_api"]:
        endpoints += [
            "/exec/open",
            "/exec/close",
            "/exec/cancel",
            "/exec/positions",
            "/exec/balance",
        ]
    if FEATURES["ui"] and _HAS_UI:
        endpoints += ["/ui/"]

    # Небольшие runtime-поля для проверки фоновых задач
    now_ts = int(time.time())
    ws_state = {
        "market_enabled": bool(app.state.market_ws_task),
        "user_enabled": bool(app.state.user_ws_task),
        "market_last_event_ago": (now_ts - int(getattr(app.state, "market_last_event_ts", 0) or 0)) if getattr(app.state, "market_last_event_ts", 0) else None,
        "user_last_event_ago": (now_ts - int(getattr(app.state, "user_last_event_ts", 0) or 0)) if getattr(app.state, "user_last_event_ts", 0) else None,
    }

    return {
        "ok": True,
        "service": "ai-trader",
        "version": APP_VERSION,
        "features": FEATURES,
        "endpoints": endpoints,
        "cors_allow_origins": allow_origins,
        "ws": ws_state,
        "env_hints": {
            "LOG_LEVEL": LOG_LEVEL,
            "CORS_ORIGINS": extra_origins,
            "TRUSTED_HOSTS": trusted_hosts,
            "DISABLE_DOCS": docs_disabled,
            "GZIP_MIN_SIZE": gzip_min,
            "RECONCILE_AT_START": RECONCILE_AT_START,
            "RECONCILE_AUTO_FIX": RECONCILE_AUTO_FIX,
            "RECONCILE_MODE": RECONCILE_MODE,
            "RECONCILE_TESTNET": RECONCILE_TESTNET,
            "RECONCILE_PERIODIC": RECONCILE_PERIODIC,
            "RECONCILE_INTERVAL_SEC": RECONCILE_INT_SEC,
            "RECONCILE_ABS_TOL": RECONCILE_ABS_TOL,
            "RECONCILE_JOURNAL_LIMIT": RECONCILE_JOURNAL_N,
            "RECONCILE_SYMBOLS": RECONCILE_SYMBOLS,
            "BINANCE_TESTNET": BINANCE_TESTNET,
            "MARKET_WS_SYMBOLS": MARKET_WS_SYMBOLS,
            "MARKET_WS_INTERVALS": MARKET_WS_INTERVALS,
            "MARKET_BACKFILL_BARS": MARKET_BACKFILL_N,
            "MARKET_REST_POLL_SEC": MARKET_REST_POLL,
            "WS_HEARTBEAT_SEC": WS_HEARTBEAT_SEC,
        },
    }
