import os, time, asyncio
from typing import Any, Dict
import httpx
import psutil
from sqlalchemy import text
from db.session import SessionLocal
from utils.secrets import get_env
from contextlib import asynccontextmanager

BINANCE_TESTNET_BASE = "https://testnet.binance.vision/api"
BINANCE_MAIN_BASE = "https://api.binance.com/api"

async def _http_get_json(url: str, timeout_s: float = 5.0) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout_s) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.json()

async def check_binance_connectivity(testnet: bool = True) -> Dict[str, Any]:
    base = BINANCE_TESTNET_BASE if testnet else BINANCE_MAIN_BASE
    t0 = time.perf_counter()
    # ping
    async with httpx.AsyncClient(timeout=5.0) as client:
        rp = await client.get(f"{base}/v3/ping")
        rp.raise_for_status()
        ping_ms = (time.perf_counter() - t0) * 1000
        # server time & drift
        rt = await client.get(f"{base}/v3/time")
        rt.raise_for_status()
        server_ms = rt.json().get("serverTime")
    local_ms = int(time.time() * 1000)
    drift_ms = abs(local_ms - int(server_ms))
    return {"ok": True, "ping_ms": round(ping_ms, 1), "time_drift_ms": drift_ms, "base": base}

def check_resources() -> Dict[str, Any]:
    cpu = psutil.cpu_percent(interval=0.1)  # короткий сэмпл
    mem = psutil.virtual_memory()
    p = psutil.Process(os.getpid())
    return {
        "cpu_percent": cpu,
        "ram_percent": mem.percent,
        "proc_rss_mb": round(p.memory_info().rss / (1024*1024), 1),
        "num_threads": p.num_threads(),
    }

def check_db() -> Dict[str, Any]:
    try:
        with SessionLocal() as s:
            s.execute(text("SELECT 1"))
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def check_quotes_minimum() -> Dict[str, Any]:
    # Лёгкая проверка наличия свечей в БД (если таблица есть)
    try:
        with SessionLocal() as s:
            res = s.execute(text("SELECT COUNT(1) FROM ohlcv")).scalar_one()
            return {"ok": True, "rows": int(res)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

async def deep_health(testnet: bool = True) -> Dict[str, Any]:
    bnc = await check_binance_connectivity(testnet=testnet)
    db = check_db()
    q = check_quotes_minimum()
    res = check_resources()
    return {
        "binance": bnc,
        "db": db,
        "quotes": q,
        "resources": res,
    }
