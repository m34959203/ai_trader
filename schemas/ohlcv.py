# schemas/ohlcv.py
from __future__ import annotations

from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class StoreRequest(BaseModel):
    # было: source: Literal["binance", "alpha_vantage"]
    source: str  # валидация источника происходит в роутере -> 400 для неизвестного
    symbol: str
    timeframe: Literal["1m","5m","15m","30m","1h","4h","1d"]
    limit: Optional[int] = Field(default=None, ge=1, le=5000)
    ts_from: Optional[int] = None
    ts_to: Optional[int] = None


class StoreResponse(BaseModel):
    stored: int
    source: str
    symbol: str
    timeframe: str


class Candle(BaseModel):
    source: str
    asset: str
    tf: str
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class QueryResponse(BaseModel):
    candles: List[Candle]


class QueryResponseWithPage(QueryResponse):
    next_offset: Optional[int] = Field(
        default=None,
        description="смещение для следующей страницы; null, если дальше данных нет",
    )
