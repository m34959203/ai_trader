from __future__ import annotations

from typing import List, Literal, Optional
from fastapi import APIRouter, Body
from pydantic import BaseModel, Field
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session
from db.models import OHLCV  # ВАЖНО: ваша актуальная модель с полями source, asset, tf, ts, open, high, low, close, volume

router = APIRouter(tags=["ohlcv-read"])

class QueryIn(BaseModel):
    source: Literal["binance", "sim"] = "binance"
    symbol: str = Field(..., examples=["BTCUSDT"])
    timeframe: str = Field(..., examples=["1m", "15m", "1h"])
    limit: int = Field(100, ge=1, le=1000)
    testnet: Optional[bool] = True  # транзитный флаг (на будущее)

class Candle(BaseModel):
    t: int
    o: float
    h: float
    l: float
    c: float
    v: float

@router.post("/prices/query")
async def prices_query(payload: QueryIn = Body(...)) -> List[Candle]:
    """
    Возвращает последние N свечей из БД в формате [{t,o,h,l,c,v}].
    Достаём из таблицы OHLCV, где:
      - source -> payload.source
      - asset  -> payload.symbol.upper()
      - tf     -> payload.timeframe
    """
    async for session in get_session():  # type: AsyncSession
        stmt = (
            select(
                OHLCV.ts.label("t"),
                OHLCV.open.label("o"),
                OHLCV.high.label("h"),
                OHLCV.low.label("l"),
                OHLCV.close.label("c"),
                OHLCV.volume.label("v"),
            )
            .where(OHLCV.source == payload.source)
            .where(OHLCV.asset == payload.symbol.upper())
            .where(OHLCV.tf == payload.timeframe)
            .order_by(desc(OHLCV.ts))
            .limit(payload.limit)
        )
        rows = (await session.execute(stmt)).all()

    if not rows:
        return []

    # Вернём по возрастанию времени
    rows = list(reversed(rows))
    return [Candle(t=r.t, o=float(r.o), h=float(r.h), l=float(r.l), c=float(r.c), v=float(r.v)) for r in rows]
