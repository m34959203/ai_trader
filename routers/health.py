from fastapi import APIRouter, Response, Header
from typing import Optional
import time, platform, os
from services.selfcheck import deep_health, check_resources

router = APIRouter(prefix="/health", tags=["health"])

START_TIME = time.time()

@router.get("")
async def health_root(response: Response, mode: Optional[str] = "binance", testnet: bool = True):
    """Лёгкая проверка (не трогаем БД/котировки)."""
    response.headers["Cache-Control"] = "no-store"
    res = check_resources()
    return {
        "status": "ok",
        "uptime_s": int(time.time() - START_TIME),
        "mode": mode,
        "testnet": bool(testnet),
        "resources": res,
        "host": platform.node(),
        "pid": os.getpid(),
    }

@router.get("/deep")
async def health_deep(response: Response, testnet: bool = True):
    """Глубокая проверка: Binance ping/time, БД, котировки, ресурсы."""
    response.headers["Cache-Control"] = "no-store"
    return await deep_health(testnet=testnet)
