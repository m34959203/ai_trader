from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from routers.auth import verify_api_key
from services.broker_gateway import BrokerGatewayError
from schemas.live_trading import (
    LiveBrokerStatus,
    LiveCancelOrderResponse,
    LiveLimitsResponse,
    LiveOrdersResponse,
    LiveOrderStatusResponse,
    LivePnLSnapshot,
    LiveStatusResponse,
    LiveSyncResponse,
    LiveTradeRequest,
    LiveTradeResponse,
    LiveTradesResponse,
    StrategyListResponse,
    StrategyState,
    StrategyUpdateRequest,
)

router = APIRouter(prefix="/live", tags=["live"])


def _get_coordinator(request: Request):
    coordinator = getattr(request.app.state, "live_trading", None)
    if coordinator is None:
        raise HTTPException(status_code=503, detail="live_trading_not_configured")
    return coordinator


@router.get("/status", response_model=LiveStatusResponse)
async def live_status(request: Request) -> LiveStatusResponse:
    coordinator = _get_coordinator(request)
    description: Dict[str, Any] = coordinator.describe()
    return LiveStatusResponse(
        configured=True,
        gateway=description.get("gateway"),
        risk=description.get("risk", {}),
        limits={
            "min_quantity": description.get("min_quantity"),
            "quantity_rounding": description.get("quantity_rounding"),
        },
    )


@router.post("/trade", response_model=LiveTradeResponse)
async def live_trade(
    request: Request,
    payload: LiveTradeRequest,
    api_key: str = Depends(verify_api_key),
) -> LiveTradeResponse:
    coordinator = _get_coordinator(request)
    try:
        summary: Dict[str, Any] = coordinator.route_and_execute(
            payload.symbol,
            payload.features,
            news_text=payload.news_text,
            order_type=payload.order_type,
            strategy=payload.strategy,
        )
    except BrokerGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return LiveTradeResponse(**summary)


@router.get("/orders", response_model=LiveOrdersResponse)
async def live_orders(
    request: Request,
    symbol: Optional[str] = None,
    refresh: bool = False,
) -> LiveOrdersResponse:
    coordinator = _get_coordinator(request)
    try:
        orders = list(coordinator.list_orders(refresh=refresh, symbol=symbol))
    except BrokerGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return LiveOrdersResponse(orders=orders)


@router.get("/orders/{request_id}", response_model=LiveOrderStatusResponse)
async def live_order_status(
    request: Request,
    request_id: str,
    symbol: Optional[str] = None,
    refresh: bool = True,
) -> LiveOrderStatusResponse:
    coordinator = _get_coordinator(request)
    try:
        order = coordinator.get_order_status(request_id, symbol=symbol, refresh=refresh)
    except BrokerGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return LiveOrderStatusResponse(request_id=request_id, found=order is not None, order=order)


@router.post("/orders/{request_id}/cancel", response_model=LiveCancelOrderResponse)
async def live_cancel_order(
    request: Request,
    request_id: str,
    symbol: Optional[str] = None,
    api_key: str = Depends(verify_api_key),
) -> LiveCancelOrderResponse:
    coordinator = _get_coordinator(request)
    try:
        result = coordinator.cancel_order(request_id, symbol=symbol)
    except BrokerGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return LiveCancelOrderResponse(**result)


@router.post("/sync", response_model=LiveSyncResponse)
async def live_sync(
    request: Request,
    api_key: str = Depends(verify_api_key),
) -> LiveSyncResponse:
    coordinator = _get_coordinator(request)
    try:
        snapshot = coordinator.sync_account()
    except BrokerGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return LiveSyncResponse(**snapshot)


@router.get("/pnl", response_model=LivePnLSnapshot)
async def live_pnl(request: Request) -> LivePnLSnapshot:
    coordinator = _get_coordinator(request)
    snapshot = coordinator.pnl_snapshot()
    return LivePnLSnapshot(**snapshot)


@router.get("/broker", response_model=LiveBrokerStatus)
async def live_broker(request: Request, include_orders: bool = True) -> LiveBrokerStatus:
    coordinator = _get_coordinator(request)
    status = coordinator.broker_status(include_orders=include_orders)
    return LiveBrokerStatus(**status)


@router.get("/trades", response_model=LiveTradesResponse)
async def live_trades(
    request: Request,
    limit: int = 50,
) -> LiveTradesResponse:
    coordinator = _get_coordinator(request)
    trades = coordinator.list_trades(limit=limit)
    return LiveTradesResponse(trades=trades)


@router.get("/limits", response_model=LiveLimitsResponse)
async def live_limits(request: Request) -> LiveLimitsResponse:
    coordinator = _get_coordinator(request)
    payload = coordinator.limits_snapshot()
    return LiveLimitsResponse(
        risk_config=payload.get("risk_config", {}),
        daily=payload.get("daily", {}),
        strategies=payload.get("strategies", []),
    )


@router.get("/strategies", response_model=StrategyListResponse)
async def live_strategies(request: Request) -> StrategyListResponse:
    coordinator = _get_coordinator(request)
    return StrategyListResponse(strategies=coordinator.list_strategy_controls())


@router.patch("/strategies/{name}", response_model=StrategyState)
async def update_strategy(
    request: Request,
    name: str,
    payload: StrategyUpdateRequest,
    api_key: str = Depends(verify_api_key),
) -> StrategyState:
    coordinator = _get_coordinator(request)
    if hasattr(payload, "model_dump"):
        data = payload.model_dump(exclude_unset=True)
    else:  # pragma: no cover - pydantic v1 fallback
        data = payload.dict(exclude_unset=True)  # type: ignore[attr-defined]
    updated = coordinator.update_strategy_control(
        name,
        enabled=data.get("enabled"),
        max_risk_fraction=data.get("max_risk_fraction"),
        max_daily_trades=data.get("max_daily_trades"),
        notes=data.get("notes"),
        provided=data.keys(),
    )
    return StrategyState(**updated)
