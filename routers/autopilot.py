# routers/autopilot.py
from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional, cast

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from services.model_router import ModelRouter, router_singleton
from src.models.base import MarketFeatures

LOG = logging.getLogger("ai_trader.autopilot")

legacy_router = APIRouter(prefix="/autopilot", tags=["autopilot"])

try:  # опциональный модуль авто-трейдера
    from tasks import auto_trader as auto_trader_module  # type: ignore

    AutoTraderConfig = auto_trader_module.AutoTraderConfig
    AutoTraderState = auto_trader_module.AutoTraderState
    get_config = auto_trader_module.get_config
    set_config = auto_trader_module.set_config
    get_runtime_status = auto_trader_module.get_runtime_status
    background_loop = auto_trader_module.background_loop

    _HAS_AUTO_TRADER_RUNTIME = True
except Exception:  # pragma: no cover
    AutoTraderConfig = None  # type: ignore
    AutoTraderState = None  # type: ignore
    get_config = None  # type: ignore
    set_config = None  # type: ignore
    get_runtime_status = None  # type: ignore
    background_loop = None  # type: ignore

    _HAS_AUTO_TRADER_RUNTIME = False


# ──────────────────────────────────────────────────────────────────────────────
# Модели запросов/ответов
# ──────────────────────────────────────────────────────────────────────────────


class AutoTraderConfigPatch(BaseModel):
    enabled: Optional[bool] = None
    symbols: Optional[List[str]] = Field(default=None, min_length=1)
    timeframe: Optional[str] = Field(default=None, min_length=1, max_length=20)
    api_base: Optional[str] = Field(default=None, min_length=3)
    source: Optional[str] = Field(default=None, min_length=1)
    mode: Optional[str] = Field(default=None, min_length=1)
    testnet: Optional[bool] = None

    sma_fast: Optional[int] = Field(default=None, ge=1)
    sma_slow: Optional[int] = Field(default=None, ge=2)
    confirm_on_close: Optional[bool] = None
    signal_persist: Optional[int] = Field(default=None, ge=1)
    signal_cooldown: Optional[int] = Field(default=None, ge=0)
    signal_min_gap_pct: Optional[float] = Field(default=None, ge=0.0)
    use_regime_filter: Optional[bool] = None

    atr_period: Optional[int] = Field(default=None, ge=0)
    atr_mult: Optional[float] = Field(default=None, ge=0.0)
    min_sl_pct: Optional[float] = Field(default=None, ge=0.0)
    max_sl_pct: Optional[float] = Field(default=None, ge=0.0)

    quote_usdt: Optional[float] = Field(default=None, ge=0.0)
    equity_min_usdt: Optional[float] = Field(default=None, ge=0.0)
    sl_pct: Optional[float] = Field(default=None, ge=0.0)
    tp_pct: Optional[float] = Field(default=None, ge=0.0)
    use_risk_autosize: Optional[bool] = None
    use_portfolio_risk: Optional[bool] = None
    max_portfolio_risk: Optional[float] = Field(default=None, ge=0.0)

    sell_on_death: Optional[bool] = None
    sell_pct_of_position: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    loop_sec: Optional[float] = Field(default=None, ge=1.0)
    http_timeout_sec: Optional[float] = Field(default=None, ge=1.0)
    net_max_retries: Optional[int] = Field(default=None, ge=1)
    net_retry_base: Optional[float] = Field(default=None, ge=0.1)
    net_retry_cap: Optional[float] = Field(default=None, ge=0.1)

    dry_run: Optional[bool] = None
    concurrency: Optional[int] = Field(default=None, ge=1)

    model_config = ConfigDict(extra="forbid")

    @field_validator("symbols", mode="before")
    @classmethod
    def _normalize_symbols(cls, value):
        if value is None:
            return value
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise TypeError("symbols must be a list of strings")
        normalized = []
        for item in value:
            if item is None:
                continue
            item_str = str(item).strip().upper()
            if item_str:
                normalized.append(item_str)
        if not normalized:
            raise ValueError("symbols must contain at least one symbol")
        return normalized

    @field_validator("timeframe", "api_base", "source", "mode", mode="before")
    @classmethod
    def _strip_strings(cls, value):
        if value is None:
            return value
        return str(value).strip()

    @model_validator(mode="after")
    def _validate_sl_bounds(self):
        if self.min_sl_pct is not None and self.max_sl_pct is not None:
            if self.max_sl_pct < self.min_sl_pct:
                raise ValueError("max_sl_pct must be >= min_sl_pct")
        if self.sma_fast is not None and self.sma_slow is not None:
            if self.sma_fast >= self.sma_slow:
                raise ValueError("sma_slow must be greater than sma_fast")
        if self.net_retry_base is not None and self.net_retry_cap is not None:
            if self.net_retry_cap < self.net_retry_base:
                raise ValueError("net_retry_cap must be >= net_retry_base")
        return self


class StartResponse(BaseModel):
    status: str
    config: Dict[str, Any]


class StopResponse(BaseModel):
    status: str


class StatusResponse(BaseModel):
    running: bool
    task_state: Dict[str, Any]
    auto_trader: Dict[str, Any]


class ConfigResponse(BaseModel):
    config: Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# Глобальное состояние управления через API
# ──────────────────────────────────────────────────────────────────────────────


_TASK: Optional[asyncio.Task] = None
_TASK_STATE: Optional[AutoTraderState] = None
_LOCK = asyncio.Lock()


def _ensure_runtime() -> None:
    if not _HAS_AUTO_TRADER_RUNTIME:
        raise HTTPException(status_code=503, detail="Autonomous trader runtime not available")


def _apply_patch(base: AutoTraderConfig, patch: AutoTraderConfigPatch) -> AutoTraderConfig:
    data = patch.model_dump(exclude_unset=True)
    if not data:
        return base
    try:
        return replace(base, **data)
    except TypeError as exc:  # pragma: no cover - дополнительная страховка
        raise HTTPException(status_code=400, detail=f"Invalid config override: {exc}") from exc


async def _stop_task_locked(timeout: float = 10.0) -> bool:
    global _TASK, _TASK_STATE
    task = _TASK
    if task is None:
        return False

    task.cancel()
    try:
        await asyncio.wait_for(task, timeout=timeout)
    except asyncio.CancelledError:
        pass
    except asyncio.TimeoutError:
        LOG.warning("Autopilot task did not stop within %.1fs", timeout)
    except Exception as exc:
        LOG.warning("Autopilot task finished with error: %r", exc)
    finally:
        _TASK = None
        _TASK_STATE = None
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Роуты управления
# ──────────────────────────────────────────────────────────────────────────────


@legacy_router.get("/config", response_model=ConfigResponse)
async def get_autopilot_config() -> ConfigResponse:
    """Вернуть текущий конфиг авто-трейдера."""
    _ensure_runtime()
    cfg = get_config()
    return ConfigResponse(config=cfg.to_public_dict())


@legacy_router.put("/config", response_model=ConfigResponse)
async def update_autopilot_config(
    patch: AutoTraderConfigPatch = Body(default_factory=AutoTraderConfigPatch),
    restart: bool = Query(default=False),
) -> ConfigResponse:
    """Обновить конфигурацию; опционально перезапустить работающий цикл."""

    _ensure_runtime()

    global _TASK, _TASK_STATE

    async with _LOCK:
        base_cfg = get_config()
        new_cfg = _apply_patch(base_cfg, patch)
        set_config(new_cfg)

        if restart and _TASK is not None and not _TASK.done():
            LOG.info("Restarting autopilot with updated configuration")
            await _stop_task_locked()
            state = AutoTraderState()
            _TASK = asyncio.create_task(
                background_loop(config=new_cfg, state=state),
                name="auto_trader_api",
            )
            _TASK_STATE = state

    return ConfigResponse(config=get_config().to_public_dict())


@legacy_router.post("/start", response_model=StartResponse)
async def start_autopilot(
    patch: AutoTraderConfigPatch = Body(default_factory=AutoTraderConfigPatch),
    restart: bool = Query(default=False),
) -> StartResponse:
    """Запустить фоновый цикл авто-трейдера с указанными override-настройками."""

    _ensure_runtime()

    global _TASK, _TASK_STATE

    async with _LOCK:
        if _TASK is not None and not _TASK.done():
            if not restart:
                raise HTTPException(status_code=409, detail="Autopilot already running")
            LOG.info("Restart requested via /autopilot/start")
            await _stop_task_locked()

        base_cfg = get_config()
        cfg = _apply_patch(base_cfg, patch)
        set_config(cfg)

        state = AutoTraderState()
        _TASK = asyncio.create_task(
            background_loop(config=cfg, state=state),
            name="auto_trader_api",
        )
        _TASK_STATE = state

    return StartResponse(status="started", config=cfg.to_public_dict())


@legacy_router.post("/stop", response_model=StopResponse)
async def stop_autopilot() -> StopResponse:
    """Остановить фоновый цикл авто-трейдера."""

    _ensure_runtime()

    async with _LOCK:
        stopped = await _stop_task_locked()
    return StopResponse(status="stopped" if stopped else "idle")


@legacy_router.get("/status", response_model=StatusResponse)
async def status_autopilot() -> StatusResponse:
    """Получить состояние фонового автотрейдера."""

    running = _TASK is not None and not _TASK.done()
    task_state: Dict[str, Any] = {
        "running": running,
        "task_name": getattr(_TASK, "get_name", lambda: "auto_trader_api")(),
    }
    if _TASK_STATE is not None:
        task_state.update(
            {
                "iteration": _TASK_STATE.iteration,
                "started_at": _TASK_STATE.started_at,
                "last_cycle_started": _TASK_STATE.last_cycle_started,
                "last_cycle_finished": _TASK_STATE.last_cycle_finished,
                "last_error": _TASK_STATE.last_error,
            }
        )

    if not _HAS_AUTO_TRADER_RUNTIME:
        auto_status: Dict[str, Any] = {"available": False}
    else:
        try:
            auto_status = get_runtime_status()
        except Exception as exc:  # pragma: no cover
            LOG.warning("Failed to fetch runtime status: %r", exc)
            auto_status = {"error": repr(exc)}

    return StatusResponse(running=running, task_state=task_state, auto_trader=auto_status)



class MarketFeaturesPayload(BaseModel):
    symbol: Optional[str] = Field(default=None, description="Trading symbol")
    price: Optional[float] = Field(default=None)
    rsi: Optional[float] = Field(default=None)
    macd: Optional[float] = Field(default=None)
    macd_signal: Optional[float] = Field(default=None)
    atr: Optional[float] = Field(default=None)
    volume: Optional[float] = Field(default=None)
    trend_ma_fast: Optional[float] = Field(default=None)
    trend_ma_slow: Optional[float] = Field(default=None)
    volatility: Optional[float] = Field(default=None)
    equity: Optional[float] = Field(default=None)
    day_pnl: Optional[float] = Field(default=None)
    news_score: Optional[float] = Field(default=None)

    model_config = ConfigDict(extra="allow")

    def to_features(self) -> MarketFeatures:
        data = self.model_dump(exclude_none=True)
        return cast(MarketFeatures, data)


class SentimentPayload(BaseModel):
    text: str


def _get_router() -> ModelRouter:
    if router_singleton is None:
        raise HTTPException(status_code=503, detail="Model router not initialised")
    return router_singleton


router = APIRouter(prefix="/ai", tags=["ai-models"])


@router.post("/signal")
async def ai_signal(payload: MarketFeaturesPayload, model_router: ModelRouter = Depends(_get_router)) -> Dict[str, Any]:
    features = payload.to_features()
    result = model_router.signal(features)
    return {"features": features, "signal": result}


@router.post("/sentiment")
async def ai_sentiment(payload: SentimentPayload, model_router: ModelRouter = Depends(_get_router)) -> Dict[str, Any]:
    sentiment = model_router.sentiment(payload.text)
    return {"sentiment": sentiment, "text": payload.text}


@router.post("/regime")
async def ai_regime(payload: MarketFeaturesPayload, model_router: ModelRouter = Depends(_get_router)) -> Dict[str, Any]:
    features = payload.to_features()
    regime = model_router.regime(features)
    return {"features": features, "regime": regime}


__all__ = ["legacy_router", "router"]
