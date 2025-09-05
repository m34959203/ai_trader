from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from ..db.session import get_session, engine, Base
from ..db import crud
from ..schemas.ohlcv import StoreRequest, StoreResponse, QueryResponse, Candle
from ..sources import alpha_vantage, binance
from ..sources.base import SourceError

router = APIRouter()

# Создадим таблицы при первом обращении (или при старте приложения в main)
@router.on_event("startup")
async def _startup():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

@router.post("/prices/store", response_model=StoreResponse, tags=["ohlcv"])
async def store_prices(req: StoreRequest, session: AsyncSession = Depends(get_session)):
    try:
        if req.source == "alpha_vantage":
            rows = alpha_vantage.fetch(req.symbol, req.timeframe, req.limit, req.ts_from, req.ts_to)
        elif req.source == "binance":
            rows = binance.fetch(req.symbol, req.timeframe, req.limit, req.ts_from, req.ts_to)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown source {req.source}")
    except SourceError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    stored = await crud.upsert_ohlcv_batch(session, rows)
    return StoreResponse(stored=stored, source=req.source, symbol=req.symbol, timeframe=req.timeframe)

@router.get("/ohlcv", response_model=QueryResponse, tags=["ohlcv"])
async def get_ohlcv(
    source: str | None = Query(default=None),
    symbol: str | None = Query(default=None, alias="ticker"),
    interval: str | None = Query(default=None, alias="timeframe"),
    ts_from: int | None = Query(default=None),
    ts_to: int | None = Query(default=None),
    limit: int = Query(default=1000, ge=1, le=10000),
    session: AsyncSession = Depends(get_session),
):
    rows = await crud.query_ohlcv(
        session,
        source=source,
        asset=symbol,
        tf=interval,
        ts_from=ts_from,
        ts_to=ts_to,
        limit=limit,
    )
    candles = [
        Candle(
            source=r.source, asset=r.asset, tf=r.tf, ts=r.ts,
            open=r.open, high=r.high, low=r.low, close=r.close, volume=r.volume
        ) for r in rows
    ]
    return QueryResponse(candles=candles)
