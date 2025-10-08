# routers/ohlcv.py
from __future__ import annotations

import asyncio
import csv
import io
import logging
from typing import AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_async_session
from db import crud
from db.models import OHLCV

# Источники котировок, которые используем в /prices/store
# (в тестах мокается binance.fetch)
from sources import binance as src_binance  # type: ignore
try:
    from sources import alpha_vantage as src_av  # type: ignore
except Exception:  # pragma: no cover
    src_av = None  # не обязателен для тестов

logger = logging.getLogger("ai_trader.ohlcv")
router = APIRouter()

_SOURCES: Dict[str, object] = {
    "binance": src_binance,
}
if src_av:
    _SOURCES["alpha_vantage"] = src_av


def _row_to_dict(r: OHLCV) -> Dict:
    # Преобразование ORM-объекта в «плоский» словарь, как ожидают тесты
    return {
        "source": r.source,
        "asset": r.asset,
        "tf": r.tf,
        "ts": int(r.ts),
        "open": float(r.open),
        "high": float(r.high),
        "low": float(r.low),
        "close": float(r.close),
        "volume": float(r.volume),
    }


@router.post("/prices/store")
async def store_prices(
    source: str,
    symbol: str,
    timeframe: str,
    limit: int = 1000,
    ts_from: Optional[int] = None,
    ts_to: Optional[int] = None,
    session: AsyncSession = Depends(get_async_session),
):
    """
    Загружает свечи из указанного источника и апсертит их в БД.
    Тесты мокают sources.binance.fetch, поэтому сеть не трогаем.
    """
    mod = _SOURCES.get(source)
    if not mod or not hasattr(mod, "fetch"):
        raise HTTPException(status_code=400, detail="unknown source")

    try:
        rows = mod.fetch(symbol=symbol, timeframe=timeframe, limit=limit, ts_from=ts_from, ts_to=ts_to)  # type: ignore
    except Exception as e:  # pragma: no cover
        logger.exception("Fetch error from %s", source)
        raise HTTPException(status_code=502, detail=f"fetch failed: {e}")

    stored = await crud.upsert_ohlcv_batch(session, rows)
    return {"stored": int(stored)}


@router.get("/ohlcv")
async def get_ohlcv(
    source: Optional[str] = Query(None),
    ticker: Optional[str] = Query(None, alias="ticker"),  # asset
    timeframe: Optional[str] = Query(None),
    ts_from: Optional[int] = None,
    ts_to: Optional[int] = None,
    limit: int = Query(1000, ge=0),
    offset: int = Query(0, ge=0),
    order: str = Query("asc"),  # "asc" | "desc"
    session: AsyncSession = Depends(get_async_session),
):
    """
    Строгая limit/offset-пагинация БЕЗ «додобора».
    Возвращаемровно то, что попадает в окно [offset, offset+limit),
    и next_offset только если за окном ещё есть данные.
    """
    try:
        total = await crud.count_ohlcv(
            session,
            source=source,
            asset=ticker,
            tf=timeframe,
            ts_from=ts_from,
            ts_to=ts_to,
        )

        rows = await crud.query_ohlcv(
            session,
            source=source,
            asset=ticker,
            tf=timeframe,
            ts_from=ts_from,
            ts_to=ts_to,
            limit=limit,
            offset=offset,
            order=order,
        )
    except Exception as e:  # pragma: no cover
        logger.exception("DB query_ohlcv error: %r", e)
        return {"ok": False, "error": f"DB query failed: {e}", "path": "/ohlcv"}

    candles = [_row_to_dict(r) for r in rows]

    # next_offset по фактическому количеству выданных строк
    next_offset: Optional[int] = None
    advanced = offset + len(candles)
    if advanced < total:
        next_offset = advanced

    return {
        "ok": True,
        "candles": candles,
        "total": int(total),
        "limit": int(limit),
        "offset": int(offset),
        "order": order,
        "next_offset": next_offset,
    }


@router.get("/ohlcv.csv")
async def get_ohlcv_csv(
    source: Optional[str] = Query(None),
    ticker: Optional[str] = Query(None, alias="ticker"),
    timeframe: Optional[str] = Query(None),
    ts_from: Optional[int] = None,
    ts_to: Optional[int] = None,
    limit: int = Query(0, ge=0),  # 0 => нет верхнего лимита (выгрузить всё)
    offset: int = Query(0, ge=0),
    order: str = Query("asc"),
    session: AsyncSession = Depends(get_async_session),
):
    """
    CSV-стрим с теми же правилами пагинации:
    - никакого «додобора»;
    - двигаем offset = offset + len(batch) при стриминге;
    - если limit > 0 — останавливаемся, когда выгружено ровно limit строк.
    """

    async def row_iter() -> AsyncIterator[bytes]:
        buf = io.StringIO()
        writer = csv.writer(buf)
        # Заголовок
        writer.writerow(["ts", "open", "high", "low", "close", "volume", "asset", "tf", "source"])
        yield buf.getvalue().encode("utf-8")
        buf.seek(0)
        buf.truncate(0)

        remaining = limit if limit and limit > 0 else None
        page_size = 1000 if remaining is None else min(1000, remaining)

        cur_offset = offset
        while True:
            batch = await crud.query_ohlcv(
                session,
                source=source,
                asset=ticker,
                tf=timeframe,
                ts_from=ts_from,
                ts_to=ts_to,
                limit=page_size,
                offset=cur_offset,
                order=order,
            )
            if not batch:
                break

            for r in batch:
                writer.writerow(
                    [
                        int(r.ts),
                        float(r.open),
                        float(r.high),
                        float(r.low),
                        float(r.close),
                        float(r.volume),
                        r.asset,
                        r.tf,
                        r.source,
                    ]
                )
            yield buf.getvalue().encode("utf-8")
            buf.seek(0)
            buf.truncate(0)

            advanced = len(batch)
            cur_offset += advanced
            if remaining is not None:
                remaining -= advanced
                if remaining <= 0:
                    break

    headers = {"Content-Disposition": 'attachment; filename="ohlcv.csv"'}
    return StreamingResponse(row_iter(), media_type="text/csv; charset=utf-8", headers=headers)
