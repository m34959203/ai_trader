from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request

from services.broker_gateway import BrokerGatewayError
from schemas.live_trading import (
    LiveCancelOrderResponse,
    LiveOrdersResponse,
    LiveOrderStatusResponse,
    LiveStatusResponse,
    LiveSyncResponse,
    LiveTradeRequest,
    LiveTradeResponse,
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
async def live_trade(request: Request, payload: LiveTradeRequest) -> LiveTradeResponse:
    coordinator = _get_coordinator(request)
    try:
        summary: Dict[str, Any] = coordinator.route_and_execute(
            payload.symbol,
            payload.features,
            news_text=payload.news_text,
            order_type=payload.order_type,
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
) -> LiveCancelOrderResponse:
    coordinator = _get_coordinator(request)
    try:
        result = coordinator.cancel_order(request_id, symbol=symbol)
    except BrokerGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return LiveCancelOrderResponse(**result)


@router.post("/sync", response_model=LiveSyncResponse)
async def live_sync(request: Request) -> LiveSyncResponse:
    coordinator = _get_coordinator(request)
    try:
        snapshot = coordinator.sync_account()
    except BrokerGatewayError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return LiveSyncResponse(**snapshot)
