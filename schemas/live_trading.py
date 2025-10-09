from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class LiveTradeRequest(BaseModel):
    symbol: str = Field(..., description="Symbol/pair to trade (e.g. BTCUSDT)")
    features: Dict[str, Any] = Field(..., description="Latest market features used for decisioning")
    news_text: Optional[str] = Field(
        default=None,
        description="Optional headline/context to feed the sentiment model",
    )
    order_type: Literal["market", "limit"] = Field(
        "market",
        description="Order type to submit via the live gateway",
    )


class LiveTradeResponse(BaseModel):
    decision: Dict[str, Any]
    order: Optional[Dict[str, Any]] = None
    executed: bool
    error: Optional[str] = None
    request_id: Optional[str] = Field(
        default=None,
        description="Client order identifier assigned to the live broker request.",
    )
    retries: int = Field(0, description="Number of broker submission retries that occurred.")


class LiveStatusResponse(BaseModel):
    configured: bool
    gateway: Optional[str] = None
    risk: Dict[str, Any] = Field(default_factory=dict)
    limits: Dict[str, Any] = Field(default_factory=dict)


class LiveOrderStatusResponse(BaseModel):
    request_id: str
    found: bool
    order: Optional[Dict[str, Any]] = None


class LiveOrdersResponse(BaseModel):
    orders: List[Dict[str, Any]] = Field(default_factory=list)


class LiveCancelOrderResponse(BaseModel):
    request_id: str
    cancelled: bool
    order: Optional[Dict[str, Any]] = None


class LiveSyncResponse(BaseModel):
    positions: Dict[str, float] = Field(default_factory=dict)
    balances: Dict[str, Dict[str, float]] = Field(default_factory=dict)
