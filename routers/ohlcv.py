# routers/ohlcv.py
from __future__ import annotations

import csv
import io
import time
import logging
import re
from typing import Literal, Optional, Iterable

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session
from db import crud
from schemas.ohlcv import (
    StoreRequest,
    StoreResponse,
    QueryResponseWithPage,
)
from sources import alpha_vantage, binance
from sources.base import SourceError

router = APIRouter(tags=["ohlcv"])
LOG = logging.getLogger("ai_trader.ohlcv")

_VALID_ORDER: set[str] = {"asc", "desc"}
_VALID_SOURCES_FETCH: set[str] = {"binance", "alpha_vantage"}
_VALID_INTERVALS_HINT = "1m/3m/5m/15m/30m/1h/4h/1d"
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_\-\.]+")


def _validate_time_range(ts_from: Optional[int], ts_to: Optional[int]) -> None:
    if ts_from is not None and ts_from < 0:
        raise HTTPException(status_code=422, detail="ts_from must be >= 0 (UNIX seconds)")
    if ts_to is not None and ts_to < 0:
        raise HTTPException(status_code=422, detail="ts_to must be >= 0 (UNIX seconds)")
    if ts_from is not None and ts_to is not None and ts_from > ts_to:
        raise HTTPException(status_code=422, detail="ts_from must be <= ts_to")


def _normalize_symbol(symbol: Optional[str], ticker: Optional[str]) -> Optional[str]:
    return (symbol or ticker or None)


def _normalize_interval(interval: Optional[str], timeframe: Optional[str]) -> Optional[str]:
    return (interval or timeframe or None)


def _safe_filename(s: str) -> str:
    s = (s or "").strip() or "all"
    return _SAFE_NAME_RE.sub("_", s)


@router.post("/prices/store", response_model=StoreResponse)
async def store_prices(
    req: StoreRequest,
    session: AsyncSession = Depends(get_session),
):
    if req.source not in _VALID_SOURCES_FETCH:
        # Тест ожидает 400
        raise HTTPException(
            status_code=400,
            detail=f"Unknown source '{req.source}'. Allowed: {sorted(_VALID_SOURCES_FETCH)}"
        )

    _validate_time_range(req.ts_from, req.ts_to)

    t0 = time.perf_counter()
    try:
        if req.source == "alpha_vantage":
            rows_iter = alpha_vantage.fetch(req.symbol, req.timeframe, req.limit, req.ts_from, req.ts_to)
        else:
            rows_iter = binance.fetch(req.symbol, req.timeframe, req.limit, req.ts_from, req.ts_to)
        rows = list(rows_iter)
    except SourceError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        LOG.exception("Fetcher error: %r", e)
        raise HTTPException(status_code=500, detail=f"Fetcher error: {e}") from e

    if not rows:
        dt_ms = (time.perf_counter() - t0) * 1000.0
        LOG.info(
            "POST /prices/store source=%s symbol=%s tf=%s rows=0 dt=%.1fms",
            req.source, req.symbol, req.timeframe, dt_ms,
        )
        return StoreResponse(stored=0, source=req.source, symbol=req.symbol, timeframe=req.timeframe)

    try:
        stored = await crud.upsert_ohlcv_batch(session, rows)
    except Exception as e:
        LOG.exception("DB upsert error: %r", e)
        raise HTTPException(status_code=500, detail=f"DB upsert error: {e}") from e

    dt_ms = (time.perf_counter() - t0) * 1000.0
    LOG.info(
        "POST /prices/store source=%s symbol=%s tf=%s stored=%d dt=%.1fms",
        req.source, req.symbol, req.timeframe, stored, dt_ms,
    )
    return StoreResponse(stored=stored, source=req.source, symbol=req.symbol, timeframe=req.timeframe)


@router.get("/ohlcv", response_model=QueryResponseWithPage)
async def get_ohlcv(
    symbol: Optional[str] = Query(default=None, description="Тикер/символ, напр. BTCUSDT или AAPL"),
    ticker: Optional[str] = Query(default=None, description="Алиас для symbol"),
    interval: Optional[str] = Query(default=None, description=f"Таймфрейм, напр. {_VALID_INTERVALS_HINT}"),
    timeframe: Optional[str] = Query(default=None, description="Алиас для interval"),
    source: Optional[str] = Query(default=None, description="Источник: binance/alpha_vantage (опционально)"),
    ts_from: Optional[int] = Query(default=None, description="UNIX seconds (start, inclusive)"),
    ts_to: Optional[int] = Query(default=None, description="UNIX seconds (end, inclusive)"),
    order: Literal["asc", "desc"] = Query(default="asc", description="Порядок сортировки по времени"),
    offset: int = Query(default=0, ge=0, description="Смещение для пагинации"),
    limit: int = Query(default=1000, ge=1, le=10000, description="Размер страницы"),
    session: AsyncSession = Depends(get_session),
):
    _validate_time_range(ts_from, ts_to)
    if order not in _VALID_ORDER:
        raise HTTPException(status_code=422, detail=f"order must be one of {_VALID_ORDER}")

    asset = _normalize_symbol(symbol, ticker)
    tf = _normalize_interval(interval, timeframe)

    t0 = time.perf_counter()
    try:
        rows = await crud.query_ohlcv(
            session,
            source=source,
            asset=asset,
            tf=tf,
            ts_from=ts_from,
            ts_to=ts_to,
            limit=limit,
            offset=offset,
            order=order,
        )
        total = await crud.count_ohlcv(
            session,
            source=source,
            asset=asset,
            tf=tf,
            ts_from=ts_from,
            ts_to=ts_to,
        )
    except Exception as e:
        LOG.exception("DB query_ohlcv error: %r", e)
        raise HTTPException(status_code=500, detail=f"DB query failed: {e}") from e

    # Страховка на случай грязной БД/рассинхронизации total
    expected = max(0, min(limit, max(0, total - offset)))
    if len(rows) > expected:
        rows = rows[:expected]

    candles = [r.to_dict() for r in rows]
    next_offset = (offset + len(rows)) if (offset + len(rows)) < total else None

    dt_ms = (time.perf_counter() - t0) * 1000.0
    LOG.info(
        "GET /ohlcv source=%s asset=%s tf=%s offset=%d limit=%d rows=%d total=%d dt=%.1fms",
        source, asset, tf, offset, limit, len(rows), total, dt_ms,
    )

    return {
        "candles": candles,
        "next_offset": next_offset,
    }


@router.get("/ohlcv/count")
async def get_ohlcv_count(
    symbol: Optional[str] = Query(default=None),
    ticker: Optional[str] = Query(default=None),
    interval: Optional[str] = Query(default=None),
    timeframe: Optional[str] = Query(default=None),
    source: Optional[str] = Query(default=None),
    ts_from: Optional[int] = Query(default=None),
    ts_to: Optional[int] = Query(default=None),
    session: AsyncSession = Depends(get_session),
):
    _validate_time_range(ts_from, ts_to)
    asset = _normalize_symbol(symbol, ticker)
    tf = _normalize_interval(interval, timeframe)

    total = await crud.count_ohlcv(
        session,
        source=source,
        asset=asset,
        tf=tf,
        ts_from=ts_from,
        ts_to=ts_to,
    )
    return {"count": int(total)}


@router.get("/ohlcv/stats")
async def get_ohlcv_stats(
    symbol: Optional[str] = Query(default=None),
    ticker: Optional[str] = Query(default=None),
    interval: Optional[str] = Query(default=None),
    timeframe: Optional[str] = Query(default=None),
    source: Optional[str] = Query(default=None),
    ts_from: Optional[int] = Query(default=None),
    ts_to: Optional[int] = Query(default=None),
    session: AsyncSession = Depends(get_session),
):
    _validate_time_range(ts_from, ts_to)
    asset = _normalize_symbol(symbol, ticker)
    tf = _normalize_interval(interval, timeframe)

    stats = await crud.stats_ohlcv(
        session,
        source=source,
        asset=asset,
        tf=tf,
        ts_from=ts_from,
        ts_to=ts_to,
    )
    # stats — уже dict {"min_ts":..., "max_ts":..., "count":...}
    return {
        "min_ts": stats["min_ts"],
        "max_ts": stats["max_ts"],
        "count": stats["count"],
    }


@router.get("/ohlcv.csv")
async def export_ohlcv_csv(
    symbol: Optional[str] = Query(default=None),
    ticker: Optional[str] = Query(default=None),
    interval: Optional[str] = Query(default=None),
    timeframe: Optional[str] = Query(default=None),
    source: Optional[str] = Query(default=None),
    ts_from: Optional[int] = Query(default=None),
    ts_to: Optional[int] = Query(default=None),
    order: Literal["asc", "desc"] = Query(default="asc"),
    session: AsyncSession = Depends(get_session),
):
    _validate_time_range(ts_from, ts_to)
    if order not in _VALID_ORDER:
        raise HTTPException(status_code=422, detail=f"order must be one of {_VALID_ORDER}")

    asset = _normalize_symbol(symbol, ticker) or "all"
    tf = _normalize_interval(interval, timeframe) or "all"

    async def row_iter() -> Iterable[bytes]:
        header = ["source", "asset", "tf", "ts", "open", "high", "low", "close", "volume"]
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(header)
        yield buf.getvalue().encode("utf-8")
        buf.seek(0)
        buf.truncate(0)

        offset = 0
        page_size = 5000
        while True:
            rows = await crud.query_ohlcv(
                session,
                source=source,
                asset=None if asset == "all" else asset,
                tf=None if tf == "all" else tf,
                ts_from=ts_from,
                ts_to=ts_to,
                limit=page_size,
                offset=offset,
                order=order,
            )
            if not rows:
                break
            for r in rows:
                writer.writerow([r.source, r.asset, r.tf, r.ts, r.open, r.high, r.low, r.close, r.volume])
            yield buf.getvalue().encode("utf-8")
            buf.seek(0)
            buf.truncate(0)
            offset += len(rows)

    fname = f"ohlcv_{_safe_filename(asset)}_{_safe_filename(tf)}.csv"
    return StreamingResponse(
        row_iter(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )
