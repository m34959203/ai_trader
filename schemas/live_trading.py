from __future__ import annotations

from datetime import datetime
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
    strategy: Optional[str] = Field(
        default=None,
        description="Logical strategy identifier used for per-strategy controls.",
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
    strategy: Optional[str] = Field(
        default=None,
        description="Strategy name associated with the trade request.",
    )


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


class LivePnLSnapshot(BaseModel):
    ts: datetime
    start_equity: Optional[float] = None
    current_equity: Optional[float] = None
    realized_pnl: Optional[float] = None
    drawdown_pct: Optional[float] = None
    trades_count: int = 0


class LiveBrokerStatus(BaseModel):
    updated_at: datetime
    connected: bool
    gateway: Optional[str] = None
    open_orders: List[Dict[str, Any]] = Field(default_factory=list)
    positions: Dict[str, float] = Field(default_factory=dict)
    balances: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    last_error: Optional[str] = None


class LiveTradeRecord(BaseModel):
    ts: datetime
    symbol: str
    strategy: str
    side: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    executed: bool = False
    error: Optional[str] = None
    risk_fraction: Optional[float] = None
    notional: Optional[float] = None
    request_id: Optional[str] = None
    status: Optional[str] = None
    filled_quantity: Optional[float] = None
    average_price: Optional[float] = None
    confidence: Optional[float] = None
    day_pnl: Optional[float] = None
    equity: Optional[float] = None


class LiveTradesResponse(BaseModel):
    trades: List[LiveTradeRecord] = Field(default_factory=list)


class StrategyState(BaseModel):
    name: str
    enabled: bool
    max_risk_fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_daily_trades: Optional[int] = Field(default=None, ge=0)
    notes: Optional[str] = None
    trades_today: int = 0
    updated_at: datetime


class StrategyListResponse(BaseModel):
    strategies: List[StrategyState] = Field(default_factory=list)


class StrategyUpdateRequest(BaseModel):
    enabled: Optional[bool] = None
    max_risk_fraction: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    max_daily_trades: Optional[int] = Field(default=None, ge=0)
    notes: Optional[str] = None


class LiveLimitsResponse(BaseModel):
    risk_config: Dict[str, Any]
    daily: Dict[str, Any]
    strategies: List[StrategyState] = Field(default_factory=list)
