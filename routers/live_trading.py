from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request

from schemas.live_trading import LiveStatusResponse, LiveTradeRequest, LiveTradeResponse

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
    summary: Dict[str, Any] = coordinator.route_and_execute(
        payload.symbol,
        payload.features,
        news_text=payload.news_text,
        order_type=payload.order_type,
    )
    return LiveTradeResponse(**summary)
