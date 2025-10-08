from fastapi import APIRouter, Response, Header, Request
from typing import Optional, Dict, Any
import time, platform, os
from services.selfcheck import deep_health, check_resources

router = APIRouter(prefix="/health", tags=["health"])

START_TIME = time.time()

def _resource_monitor_status(request: Request) -> Optional[Dict[str, Any]]:
    monitor = getattr(request.app.state, "resource_monitor", None)
    if monitor is None:
        return None
    try:
        return monitor.status()
    except Exception:
        return None


@router.get("")
async def health_root(
    request: Request,
    response: Response,
    mode: Optional[str] = "binance",
    testnet: bool = True,
):
    """Лёгкая проверка (не трогаем БД/котировки)."""
    response.headers["Cache-Control"] = "no-store"
    res = check_resources()
    monitor_status = _resource_monitor_status(request)
    return {
        "status": "ok",
        "uptime_s": int(time.time() - START_TIME),
        "mode": mode,
        "testnet": bool(testnet),
        "resources": res,
        "resource_monitor": monitor_status,
        "host": platform.node(),
        "pid": os.getpid(),
    }

@router.get("/deep")
async def health_deep(request: Request, response: Response, testnet: bool = True):
    """Глубокая проверка: Binance ping/time, БД, котировки, ресурсы."""
    response.headers["Cache-Control"] = "no-store"
    payload = await deep_health(testnet=testnet)
    monitor_status = _resource_monitor_status(request)
    if monitor_status is not None:
        payload["resource_monitor"] = monitor_status
    return payload
