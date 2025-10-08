# src/main.py
from __future__ import annotations

import json
import os
import time
import logging
import asyncio
from typing import Literal, Optional, Any, Dict, List
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from risk.deadman import HEARTBEAT_FILE as DEADMAN_HEARTBEAT_FILE  # type: ignore
    _HAS_DEADMAN = True
except Exception:  # pragma: no cover
    DEADMAN_HEARTBEAT_FILE = Path("data/state/heartbeat.txt")
    _HAS_DEADMAN = False

import pandas as pd
import httpx
import psutil
from fastapi import FastAPI, Query, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, Response, PlainTextResponse

# ──────────────────────────────────────────────────────────────────────────────
# .env
# ──────────────────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(dotenv_path=os.getenv("ENV_FILE", ".env"), override=False)
except Exception:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# Локальные импорты
# ──────────────────────────────────────────────────────────────────────────────
from .data_loader import get_prices

# БД
from db.session import engine, Base, apply_startup_pragmas_and_schema, shutdown_engine, get_session  # get_session нужно фолбэкам
from db import crud  # тоже нужно фолбэкам

# Роутеры
try:
    from routers.ohlcv_read import router as ohlcv_read_router  # /ohlcv/prices/query
    _HAS_OHLCV_READ = True
except Exception:
    _HAS_OHLCV_READ = False

try:
    from routers.ohlcv import router as ohlcv_router            # /prices/store, /ohlcv*
    _HAS_OHLCV = True
except Exception:
    _HAS_OHLCV = False

try:
    from routers.trading import router as trading_router
    _HAS_TRADING = True
except Exception:
    _HAS_TRADING = False

try:
    from routers.trading_exec import router as exec_router
    _HAS_EXEC = True
except Exception:
    _HAS_EXEC = False

try:
    from routers.ui import router as ui_router
    _HAS_UI = True
except Exception:
    _HAS_UI = False

try:
    from routers.autopilot import router as autopilot_router
    _HAS_AUTOPILOT = True
except Exception:
    _HAS_AUTOPILOT = False

# Аналитика
try:
    from .analysis.analyze_market import analyze_market, DEFAULT_CONFIG, AnalysisConfig  # type: ignore
except Exception:  # pragma: no cover
    analyze_market = None  # type: ignore
    DEFAULT_CONFIG = None  # type: ignore
    AnalysisConfig = None  # type: ignore

# Потоки (опц.)
try:
    from services.reconcile import (  # type: ignore
        reconcile_positions,
        reconcile_on_start,
        reconcile_periodic,
        get_periodic_config_from_env,
    )
    _HAS_RECONCILE = True
except Exception:  # pragma: no cover
    _HAS_RECONCILE = False

try:
    from services.stream_router import StreamRouter, StreamConfig  # type: ignore
    from sources.binance_ws import BinanceWS  # type: ignore
    from sources.binance import BinanceREST   # type: ignore
    _HAS_STREAMS = True
except Exception:  # pragma: no cover
    _HAS_STREAMS = False

# Фоновые задачи (делаем отключаемыми)
try:
    from tasks.ohlcv_loader import background_loop as ohlcv_bg
    _HAS_OHLCV_BG = True
except Exception:
    _HAS_OHLCV_BG = False

try:
    from tasks.auto_trader import background_loop as auto_bg
    _HAS_AUTO_BG = True
except Exception:
    _HAS_AUTO_BG = False


APP_VERSION = os.getenv("APP_VERSION", "0.11.1")
APP_ENV = (os.getenv("APP_ENV") or "dev").strip().lower()

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
# Логирование
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
    "ohlcv_storage": env_bool("FEATURE_OHLCV", True) and (_HAS_OHLCV or _HAS_OHLCV_READ),
    "signals":       env_bool("FEATURE_SIGNALS", True),
    "paper_trading": env_bool("FEATURE_PAPER", True) and _HAS_TRADING,
    "execution_api": env_bool("FEATURE_EXEC", True) and _HAS_EXEC,
    "ui":            env_bool("FEATURE_UI", True) and _HAS_UI,
    "market_ws":     env_bool("FEATURE_MARKET_WS", True) and _HAS_STREAMS,
    "user_ws":       env_bool("FEATURE_USER_WS", True) and _HAS_STREAMS,
    "autopilot":     env_bool("FEATURE_AUTOPILOT", True) and _HAS_AUTOPILOT,
}

# Управление бэкграунд-тасками: включаем по умолчанию для prod/staging окружений
_BG_DEFAULT = APP_ENV in {"prod", "production", "staging"}
ENABLE_BG_TASKS = env_bool("ENABLE_BG_TASKS", _BG_DEFAULT)

# Binance / WS окружение
BINANCE_API_KEY     = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET  = os.getenv("BINANCE_API_SECRET", "")
BINANCE_TESTNET     = env_bool("BINANCE_TESTNET", True)

MARKET_WS_SYMBOLS   = env_list("MARKET_WS_SYMBOLS", "BTCUSDT")
MARKET_WS_INTERVALS = env_list("MARKET_WS_INTERVALS", "1m,5m")
MARKET_BACKFILL_N   = env_int("MARKET_BACKFILL_BARS", 1000)
MARKET_REST_POLL    = env_float("MARKET_REST_POLL_SEC", 2.0)

WS_HEARTBEAT_SEC    = env_int("WS_HEARTBEAT_SEC", 90)

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
# Константы
# ──────────────────────────────────────────────────────────────────────────────
APP_START_TS = int(time.time())
HEARTBEAT_FILE = DEADMAN_HEARTBEAT_FILE
HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)

BINANCE_REST_TESTNET = "https://testnet.binance.vision/api"
BINANCE_REST_MAINNET = "https://api.binance.com/api"

# ──────────────────────────────────────────────────────────────────────────────
# Watchdog
# ──────────────────────────────────────────────────────────────────────────────
class EventLoopWatchdog:
    def __init__(self, interval: float = 5.0, max_consecutive_misses: int = 12):
        self.interval = interval
        self.max_misses = max_consecutive_misses
        self._misses = 0
        self._stop = False
        self.log = logging.getLogger("ai_trader.watchdog")

    async def run(self) -> None:
        while not self._stop:
            t0 = time.perf_counter()
            try:
                await asyncio.sleep(self.interval)
                lag = time.perf_counter() - t0 - self.interval
                HEARTBEAT_FILE.write_text(str(int(time.time())))
                if lag > self.interval * 2:
                    self._misses += 1
                    self.log.warning("Event loop lag: %.3fs (miss=%d)", lag, self._misses)
                else:
                    self._misses = 0
                if self._misses >= self.max_misses:
                    self.log.error("Too many loop lags -> exiting for supervisor restart")
                    os._exit(42)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log.exception("Watchdog error: %r", e)

    def stop(self) -> None:
        self._stop = True

# ──────────────────────────────────────────────────────────────────────────────
# Быстрые self-check
# ──────────────────────────────────────────────────────────────────────────────
def _resources() -> Dict[str, Any]:
    vm = psutil.virtual_memory()
    p = psutil.Process(os.getpid())
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.05),
        "ram_percent": vm.percent,
        "proc_rss_mb": round(p.memory_info().rss / (1024 * 1024), 1),
        "num_threads": p.num_threads(),
        "uptime_s": int(time.time() - APP_START_TS),
    }

async def _check_db() -> Dict[str, Any]:
    try:
        async with engine.begin() as conn:
            await conn.exec_driver_sql("SELECT 1;")
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

async def _check_quotes_count() -> Dict[str, Any]:
    try:
        async with engine.begin() as conn:
            res = await conn.exec_driver_sql("SELECT COUNT(1) FROM ohlcv;")
            row = res.first()
            cnt = int(row[0]) if row and row[0] is not None else 0
        return {"ok": True, "rows": cnt}
    except Exception as e:
        return {"ok": False, "error": str(e)}

async def _check_binance(testnet: bool = True) -> Dict[str, Any]:
    base = BINANCE_REST_TESTNET if testnet else BINANCE_REST_MAINNET
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            t0 = time.perf_counter()
            rp = await client.get(f"{base}/v3/ping")
            rp.raise_for_status()
            ping_ms = (time.perf_counter() - t0) * 1000.0
            rt = await client.get(f"{base}/v3/time")
            rt.raise_for_status()
            server_ms = int(rt.json().get("serverTime", 0))
    except Exception as e:
        return {"ok": False, "base": base, "error": str(e)}
    local_ms = int(time.time() * 1000)
    drift_ms = abs(local_ms - server_ms)
    return {"ok": True, "base": base, "ping_ms": round(ping_ms, 1), "time_drift_ms": drift_ms}

# ──────────────────────────────────────────────────────────────────────────────
# Lifespan
# ──────────────────────────────────────────────────────────────────────────────
async def _consume_market_stream(app: FastAPI) -> None:
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

async def _consume_user_stream(app: FastAPI) -> None:
    assert app.state.user_ws is not None
    async for evt in app.state.user_ws.start_user():
        app.state.user_last_event_ts = int(time.time())
        t = evt.get("type")
        if t == "executionReport":
            data = evt.get("data", {})
            logging.getLogger("ai_trader.exec_report").debug("ExecReport: %s", data)

async def _ws_watchdog(app: FastAPI, heartbeat_sec: int) -> None:
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: PRAGMA/схема
    async with engine.begin() as conn:
        await conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
        await conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
        await conn.exec_driver_sql("PRAGMA foreign_keys=ON;")
        await conn.run_sync(Base.metadata.create_all)

    try:
        await apply_startup_pragmas_and_schema()
    except Exception as e:  # pragma: no cover
        LOG.warning("Startup DB init helper failed: %r", e)

    # Состояние
    app.state.reconcile_task = None
    app.state.market_ws_task = None
    app.state.user_ws_task   = None
    app.state.keepalive_task = None
    app.state.ohlcv_bg_task  = None
    app.state.auto_bg_task   = None
    app.state.watchdog_task  = None
    app.state._watchdog      = None

    app.state.stream_router = None
    app.state.user_ws = None
    app.state.binance_rest = None

    app.state.market_last_event_ts = 0
    app.state.user_last_event_ts   = 0
    app.state.market_ring: List[Dict[str, Any]] = []

    # Reconcile on start / periodic loop (если доступно и разрешено)
    if _HAS_RECONCILE and FEATURES.get("execution_api", False):
        if RECONCILE_AT_START:
            try:
                report = await reconcile_on_start()
                if report and not report.get("ok", True):
                    LOG.warning("Reconcile on start reported issues: %s", report)
                elif report:
                    LOG.info(
                        "Reconcile on start completed: mismatches=%s",
                        report.get("summary", {}).get("mismatch_count"),
                    )
            except Exception as e:  # pragma: no cover
                LOG.warning("Reconcile on start failed: %r", e)

        if ENABLE_BG_TASKS and RECONCILE_PERIODIC:
            periodic_cfg: Optional[Dict[str, Any]] = None
            try:
                periodic_cfg = get_periodic_config_from_env()
            except Exception as e:  # pragma: no cover
                LOG.warning("Failed to load periodic reconcile config: %r", e)

            if not periodic_cfg:
                symbols_list = [
                    s.strip().upper() for s in (RECONCILE_SYMBOLS or "").split(",") if s.strip()
                ]
                mode_env = str(RECONCILE_MODE or "binance").strip().lower()
                periodic_cfg = {
                    "interval_sec": max(5, int(RECONCILE_INT_SEC or 60)),
                    "mode": "sim" if mode_env == "sim" else "binance",
                    "testnet": bool(RECONCILE_TESTNET),
                    "auto_fix": bool(RECONCILE_AUTO_FIX),
                    "abs_tol": float(RECONCILE_ABS_TOL or 1e-8),
                    "journal_limit": int(RECONCILE_JOURNAL_N or 200),
                    "symbols": symbols_list or None,
                }

            try:
                app.state.reconcile_task = asyncio.create_task(
                    reconcile_periodic(**periodic_cfg),
                    name="reconcile_loop",
                )
                LOG.info(
                    "Reconcile loop scheduled: every %ss (mode=%s, testnet=%s, auto_fix=%s)",
                    periodic_cfg.get("interval_sec"),
                    periodic_cfg.get("mode"),
                    periodic_cfg.get("testnet"),
                    periodic_cfg.get("auto_fix"),
                )
            except Exception as e:  # pragma: no cover
                LOG.warning("Failed to start reconcile loop: %r", e)

    # MARKET WS
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
            app.state.market_ws_task = asyncio.create_task(_consume_market_stream(app), name="market_ws")
            LOG.info("Market WS task started: %s @ %s", MARKET_WS_SYMBOLS, MARKET_WS_INTERVALS)
        except Exception as e:  # pragma: no cover
            LOG.warning("Failed to start Market WS task: %r", e)

    # USER WS
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
            app.state.user_ws_task = asyncio.create_task(_consume_user_stream(app), name="user_ws")
            LOG.info("User WS task started (testnet=%s)", BINANCE_TESTNET)
        except Exception as e:  # pragma: no cover
            LOG.warning("Failed to start User WS task: %r", e)

    # WS watchdog
    if (app.state.market_ws_task or app.state.user_ws_task):
        try:
            app.state.keepalive_task = asyncio.create_task(_ws_watchdog(app, WS_HEARTBEAT_SEC), name="ws_watchdog")
            LOG.info("WS watchdog started: heartbeat=%ss", int(WS_HEARTBEAT_SEC))
        except Exception as e:  # pragma: no cover
            LOG.warning("Failed to start WS watchdog: %r", e)

    # Фоновые задачи — ТОЛЬКО если явно разрешены
    if ENABLE_BG_TASKS and _HAS_OHLCV_BG:
        app.state.ohlcv_bg_task = asyncio.create_task(ohlcv_bg(), name="ohlcv_bg")
    if ENABLE_BG_TASKS and _HAS_AUTO_BG:
        app.state.auto_bg_task = asyncio.create_task(auto_bg(), name="auto_bg")

    # Process watchdog
    app.state._watchdog = EventLoopWatchdog(interval=5.0, max_consecutive_misses=12)
    app.state.watchdog_task = asyncio.create_task(app.state._watchdog.run(), name="loop_watchdog")

    LOG.info("Startup complete: DB schema ensured, version=%s, features=%s", APP_VERSION, FEATURES)
    try:
        yield
    finally:
        # Shutdown
        try:
            tasks = [
                getattr(app.state, "keepalive_task", None),
                getattr(app.state, "user_ws_task", None),
                getattr(app.state, "market_ws_task", None),
                getattr(app.state, "reconcile_task", None),
                getattr(app.state, "ohlcv_bg_task", None),
                getattr(app.state, "auto_bg_task", None),
                getattr(app.state, "watchdog_task", None),
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

        try:
            if app.state._watchdog:
                app.state._watchdog.stop()
        except Exception:
            pass
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
            await shutdown_engine()
        except Exception:  # pragma: no cover
            pass
        LOG.info("Shutdown complete")

# ──────────────────────────────────────────────────────────────────────────────
# Приложение
# ──────────────────────────────────────────────────────────────────────────────
docs_disabled = env_bool("DISABLE_DOCS", False)
openapi_url = None if docs_disabled else "/openapi.json"
docs_url = None if docs_disabled else "/docs"
redoc_url = None if docs_disabled else "/redoc"

tags_metadata = [
    {"name": "prices",    "description": "Загрузка котировок из внешних источников (без БД)."},
    {"name": "ohlcv",     "description": "Хранилище свечей и выборка из БД."},
    {"name": "strategy",  "description": "Индикаторы, сигналы, стратегии."},
    {"name": "paper",     "description": "Бэктест и paper-trading (виртуальные сделки)."},
    {"name": "exec",      "description": "Исполнение ордеров: симулятор, Binance API, UI-агент."},
    {"name": "autopilot", "description": "Пилот автоторговли (force/test режим)."},
    {"name": "meta",      "description": "Служебные и диагностические эндпоинты."},
    {"name": "ui",        "description": "Мониторинг и быстрая ручная панель."},
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
# Middlewares
# ──────────────────────────────────────────────────────────────────────────────
trusted_hosts = env_list("TRUSTED_HOSTS", "")
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
allow_origins = list(dict.fromkeys(default_origins + extra_origins))

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
# Обработчики ошибок
# ──────────────────────────────────────────────────────────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        headers={"Cache-Control": "no-store"},
        content={"ok": False, "error": str(exc.detail), "path": str(request.url.path)},
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    LOG.exception("Unhandled exception at %s: %r", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        headers={"Cache-Control": "no-store"},
        content={"ok": False, "error": "Internal Server Error", "path": str(request.url.path)},
    )

# ──────────────────────────────────────────────────────────────────────────────
# META + HEALTH
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/ping", tags=["meta"])
async def ping() -> Dict[str, Any]:
    return {"status": "ok", "message": "AI Trader is running 🚀"}

@app.get("/health", tags=["meta"])
async def health(testnet: bool = True) -> JSONResponse:
    res = _resources()
    bnc = await _check_binance(testnet=testnet)
    return JSONResponse({"status": "ok", "resources": res, "binance": bnc}, headers={"Cache-Control": "no-store"})

@app.get("/health/deep", tags=["meta"])
async def health_deep(testnet: bool = True) -> JSONResponse:
    db_res, q_res, bnc_res = await asyncio.gather(_check_db(), _check_quotes_count(), _check_binance(testnet=testnet))
    return JSONResponse(
        {"ok": True, "db": db_res, "quotes": q_res, "binance": bnc_res, "resources": _resources()},
        headers={"Cache-Control": "no-store"},
    )

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
    return PlainTextResponse("OK", headers={"Cache-Control": "no-store"})

@app.get("/_readyz", tags=["meta"])
async def readyz() -> PlainTextResponse:
    try:
        async with engine.begin() as conn:
            await conn.exec_driver_sql("SELECT 1;")
        return PlainTextResponse("READY", headers={"Cache-Control": "no-store"})
    except Exception:  # pragma: no cover
        return PlainTextResponse("NOT_READY", status_code=503, headers={"Cache-Control": "no-store"})

@app.get("/version", tags=["meta"])
async def version() -> Dict[str, Any]:
    return {"version": APP_VERSION}

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
# /prices (внешние источники, без БД)
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/prices", tags=["prices"])
async def prices(
    source: Literal["ccxt", "yfinance", "stooq", "alphavantage"] = Query(
        "ccxt", description="Источник данных"
    ),
    symbol: Optional[str] = Query(None, description="Напр.: BTC/USDT (для ccxt)"),
    timeframe: str = Query("1h", description="1m,5m,15m,1h,4h,1d... (ccxt)"),
    limit: int = Query(200, ge=1, le=5000, description="Количество свечей (ccxt)"),
    exchange_name: Optional[str] = Query(None, description="binance/bybit/okx..."),
    ticker: Optional[str] = Query(None, description="Напр.: AAPL, MSFT, EURUSD=X"),
    interval: str = Query("1h", description="yfinance/alphavantage: 1m..1h; stooq: игнорируется"),
    period: str = Query("7d", description="yfinance: 1d..1mo..; stooq/alphavantage: игнорируется"),
    fmt: Literal["json", "csv"] = Query("json", description="Формат ответа"),
) -> Response:
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

    records: List[Dict[str, Any]] = json.loads(df.to_json(orient="records"))
    LOG.info("GET /prices JSON source=%s symbol=%s ticker=%s -> rows=%d in %.3fs",
             source, symbol, ticker, rows_count, time.perf_counter() - t0)
    return JSONResponse({"rows": len(records), "data": records}, headers={"Cache-Control": "no-store"})

# ──────────────────────────────────────────────────────────────────────────────
# Подключение роутеров
# ──────────────────────────────────────────────────────────────────────────────
if _HAS_OHLCV:
    app.include_router(ohlcv_router)  # router сам решает свой префикс/пути

if _HAS_OHLCV_READ:
    app.include_router(ohlcv_read_router, prefix="/ohlcv")

if _HAS_TRADING:
    app.include_router(trading_router, tags=["strategy", "paper"])

if _HAS_EXEC:
    app.include_router(exec_router, tags=["exec"])

if FEATURES["autopilot"] and _HAS_AUTOPILOT:
    app.include_router(autopilot_router, tags=["autopilot"])

if _HAS_UI and FEATURES["ui"]:
    app.include_router(ui_router, tags=["ui"])

# ──────────────────────────────────────────────────────────────────────────────
# Fallback-маршруты для OHLCV (если по какой-то причине роутер не подключился)
# ──────────────────────────────────────────────────────────────────────────────
def _route_exists(path: str, method: Optional[str] = None) -> bool:
    method_u = None if method is None else method.upper()
    for r in app.router.routes:
        try:
            p = getattr(r, "path", getattr(r, "path_format", None))
            mset = getattr(r, "methods", None)
            if p == path and (method_u is None or (mset and method_u in mset)):
                return True
        except Exception:
            continue
    return False

# подключим фолбэки только если базовых путей нет
# подключим фолбэки только если базовых путей нет
if not _route_exists("/prices/store", "POST"):
    LOG.warning("OHLCV router not detected -> enabling inline fallback endpoints")

    from fastapi import Body
    import csv, io, re
    from sqlalchemy import text as sa_text
    from typing import Any, Dict, Optional, AsyncIterator

    _VALID_SOURCES_FETCH = {"binance", "alpha_vantage"}
    _SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_\-\.]+")
    _NO_STORE = {"Cache-Control": "no-store"}

    def _safe_filename(s: str) -> str:
        return _SAFE_NAME_RE.sub("_", (s or "all").strip())

    def _row_to_dict(r: Any) -> Dict[str, Any]:
        return {
            "source": getattr(r, "source", None),
            "asset": getattr(r, "asset", None),
            "tf": getattr(r, "tf", None),
            "ts": int(getattr(r, "ts", 0)),
            "open": float(getattr(r, "open", 0.0)),
            "high": float(getattr(r, "high", 0.0)),
            "low": float(getattr(r, "low", 0.0)),
            "close": float(getattr(r, "close", 0.0)),
            "volume": float(getattr(r, "volume", 0.0)),
        }

    # ------------------------------------------------------------
    # POST /prices/store — очистка старых данных + вставка свежих
    # ------------------------------------------------------------
    @app.post("/prices/store", tags=["ohlcv"])
    async def store_prices_fallback(
        body: Dict[str, Any] = Body(...),
        session=Depends(get_session),
    ):
        source = (body.get("source") or "").strip().lower()
        symbol = (body.get("symbol") or body.get("ticker") or "").strip().upper()
        timeframe = (body.get("timeframe") or body.get("tf") or "").strip()
        limit = int(body.get("limit") or 1000)
        ts_from = body.get("ts_from")
        ts_to = body.get("ts_to")

        if source not in _VALID_SOURCES_FETCH:
            raise HTTPException(status_code=400, detail=f"unknown source: {source}")

        # загрузка строк
        try:
            if source == "alpha_vantage":
                from sources import alpha_vantage as av
                rows = av.fetch(symbol=symbol, timeframe=timeframe, limit=limit, ts_from=ts_from, ts_to=ts_to)
            else:
                from sources import binance as bnc
                rows = bnc.fetch(symbol=symbol, timeframe=timeframe, limit=limit, ts_from=ts_from, ts_to=ts_to)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"fetch failed: {e}")

        app_env = (os.getenv("APP_ENV") or "test").lower()
        if app_env in ("test", "testing", "ci"):
            await session.execute(
                sa_text("DELETE FROM ohlcv WHERE source=:source AND asset=:asset AND tf=:tf"),
                {"source": source, "asset": symbol, "tf": timeframe},
            )
            await session.commit()

        stored = await crud.upsert_ohlcv_batch(session, rows)
        return {"stored": int(stored)}

    # ------------------------------------------------------------
    # GET /ohlcv — строгая пагинация + корректный next_offset
    # ------------------------------------------------------------
    @app.get("/ohlcv", tags=["ohlcv"])
    async def get_ohlcv_fallback(
        source: Optional[str] = Query(None),
        ticker: Optional[str] = Query(None),
        timeframe: Optional[str] = Query(None),
        ts_from: Optional[int] = None,
        ts_to: Optional[int] = None,
        limit: int = Query(1000, ge=0),
        offset: int = Query(0, ge=0),
        order: str = Query("asc"),
        session=Depends(get_session),
    ):
        try:
            total = await crud.count_ohlcv(session, source=source, asset=ticker, tf=timeframe,
                                           ts_from=ts_from, ts_to=ts_to)
            rows = await crud.query_ohlcv(session, source=source, asset=ticker, tf=timeframe,
                                          ts_from=ts_from, ts_to=ts_to,
                                          limit=limit, offset=offset, order=order)
        except Exception as e:
            LOG.exception("DB query_ohlcv error: %r", e)
            return {"ok": False, "error": f"DB query failed: {e}"}

        remaining = max(0, int(total) - int(offset))
        expected = min(int(limit), remaining) if limit is not None else remaining
        if expected >= 0 and len(rows) > expected:
            rows = rows[:expected]

        candles = [_row_to_dict(r) for r in rows]

        next_offset = None
        advanced = offset + len(candles)
        if advanced < total:
            next_offset = advanced

        return {
            "ok": True,
            "candles": candles,
            "total": int(total),
            "limit": int(limit),
            "offset": int(offset),
            "order": order,
            "next_offset": next_offset,
        }

    # ------------------------------------------------------------
    # GET /ohlcv.csv — выгрузка CSV по страницам
    # ------------------------------------------------------------
    @app.get("/ohlcv.csv", tags=["ohlcv"])
    async def get_ohlcv_csv_fallback(
        source: Optional[str] = Query(None),
        ticker: Optional[str] = Query(None),
        timeframe: Optional[str] = Query(None),
        ts_from: Optional[int] = None,
        ts_to: Optional[int] = None,
        limit: int = Query(0, ge=0),
        offset: int = Query(0, ge=0),
        order: str = Query("asc"),
        session=Depends(get_session),
    ):
        async def row_iter() -> AsyncIterator[bytes]:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["ts", "open", "high", "low", "close", "volume", "asset", "tf", "source"])
            yield buf.getvalue().encode("utf-8")
            buf.seek(0); buf.truncate(0)

            remaining = limit if limit and limit > 0 else None
            page_size = 1000 if remaining is None else min(1000, remaining)
            cur_offset = offset

            while True:
                batch = await crud.query_ohlcv(session,
                                               source=source, asset=ticker, tf=timeframe,
                                               ts_from=ts_from, ts_to=ts_to,
                                               limit=page_size, offset=cur_offset, order=order)
                if not batch:
                    break

                for r in batch:
                    writer.writerow([
                        int(r.ts), float(r.open), float(r.high), float(r.low),
                        float(r.close), float(r.volume),
                        r.asset, r.tf, r.source,
                    ])
                yield buf.getvalue().encode("utf-8")
                buf.seek(0); buf.truncate(0)

                advanced = len(batch)
                cur_offset += advanced
                if remaining is not None:
                    remaining -= advanced
                    if remaining <= 0:
                        break

        headers = {"Content-Disposition": 'attachment; filename="ohlcv.csv"'}
        return StreamingResponse(row_iter(), media_type="text/csv; charset=utf-8", headers=headers)

# ──────────────────────────────────────────────────────────────────────────────
# /analyze (как было)
# ──────────────────────────────────────────────────────────────────────────────
def _normalize_prices_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    df = df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df.sort_index()
    return df

if env_bool("FEATURE_SIGNALS", True) and analyze_market is not None:
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
        if analyze_market is None:
            raise HTTPException(status_code=500, detail="Analysis module not available")

        try:
            df_fast = get_prices(source=source, symbol=symbol, timeframe=tf_fast, limit=limit_fast, exchange_name=exchange_name)
            df_slow = None
            if not no_mtf and tf_slow:
                df_slow = get_prices(source=source, symbol=symbol, timeframe=tf_slow, limit=limit_slow, exchange_name=exchange_name)
        except Exception as e:
            LOG.exception("GET /analyze: get_prices failed: %r", e)
            raise HTTPException(status_code=400, detail=f"Failed to load prices: {e}")

        if df_fast is None or df_fast.empty:
            raise HTTPException(status_code=400, detail="No data for fast timeframe")

        df_fast = _normalize_prices_df(df_fast)
        df_slow = _normalize_prices_df(df_slow) if df_slow is not None else None
        if df_fast is None or not isinstance(df_fast.index, pd.DatetimeIndex):
            raise HTTPException(status_code=400, detail="Analysis assertion: DataFrame index must be a pandas.DatetimeIndex (UTC).")

        cfg = DEFAULT_CONFIG
        if (buy_th is not None or sell_th is not None) and AnalysisConfig is not None and DEFAULT_CONFIG is not None:
            cfg_dict = dict(DEFAULT_CONFIG.__dict__)  # type: ignore
            if buy_th is not None:
                cfg_dict["buy_threshold"] = int(buy_th)
            if sell_th is not None:
                cfg_dict["sell_threshold"] = int(sell_th)
            cfg = AnalysisConfig(**cfg_dict)  # type: ignore

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
# Root index
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/", tags=["meta"])
async def root() -> Dict[str, Any]:
    endpoints: List[str] = [
        "/",
        "/ping",
        "/health",
        "/health/deep",
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
            "/ohlcv",
            "/ohlcv/prices/query",
        ]
    if FEATURES["signals"] or FEATURES["paper_trading"]:
        endpoints += ["/analyze"]
    if FEATURES["execution_api"]:
        endpoints += [
            "/exec/open",
            "/exec/close",
            "/exec/cancel",
            "/exec/positions",
            "/exec/balance",
        ]
    if FEATURES.get("autopilot", False) and _HAS_AUTOPILOT:
        endpoints += ["/autopilot/status", "/autopilot/start", "/autopilot/stop"]
    if FEATURES["ui"] and _HAS_UI:
        endpoints += ["/ui/"]

    now_ts = int(time.time())
    ws_state = {
        "market_enabled": bool(getattr(app.state, "market_ws_task", None)),
        "user_enabled": bool(getattr(app.state, "user_ws_task", None)),
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
            "ENABLE_BG_TASKS": ENABLE_BG_TASKS,
        },
    }
