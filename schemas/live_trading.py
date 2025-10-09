from __future__ import annotations

from typing import Any, Dict, Literal, Optional

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


class LiveStatusResponse(BaseModel):
    configured: bool
    gateway: Optional[str] = None
    risk: Dict[str, Any] = Field(default_factory=dict)
    limits: Dict[str, Any] = Field(default_factory=dict)
