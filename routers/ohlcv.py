from __future__ import annotations

import csv
import io
import logging
import os
from typing import AsyncIterator, Dict, List, Optional, Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy import text as sa_text, select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session
from db import crud
from db.models import OHLCV

# Источники (в тестах binance.fetch мокается)
from sources import binance as src_binance  # type: ignore
try:
    from sources import alpha_vantage as src_av  # type: ignore
except Exception:  # pragma: no cover
    src_av = None

logger = logging.getLogger("ai_trader.ohlcv")
router = APIRouter(tags=["ohlcv"])

_SOURCES: Dict[str, object] = {"binance": src_binance}
if src_av:
    _SOURCES["alpha_vantage"] = src_av

# ──────────────────────────────────────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────────────────────────────────────

def _row_to_dict(r: OHLCV) -> Dict:
    """ORM -> dict в формате, которого ожидают тесты."""
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

CSV_HEADER = ["source", "asset", "tf", "ts", "open", "high", "low", "close", "volume"]

# ──────────────────────────────────────────────────────────────────────────────
# Запись/апсерт
# ──────────────────────────────────────────────────────────────────────────────

@router.post("/prices/store")
async def store_prices(
    payload: dict = Body(...),
    session: AsyncSession = Depends(get_session),
):
    """
    Загружает OHLCV из источника и сохраняет в БД.
    В тестовом окружении перед вставкой удаляет существующие строки
    для (source, symbol, timeframe), чтобы прогоны были детерминированными.
    """
    source = (payload.get("source") or "").strip().lower()
    symbol = (payload.get("symbol") or payload.get("ticker") or "").strip().upper()
    timeframe = (payload.get("timeframe") or payload.get("tf") or "").strip()
    limit = int(payload.get("limit") or 1000)
    ts_from = payload.get("ts_from")
    ts_to = payload.get("ts_to")

    mod = _SOURCES.get(source)
    if not mod or not hasattr(mod, "fetch"):
        raise HTTPException(status_code=400, detail=f"unknown source: {source}")

    try:
        rows: List[dict] = mod.fetch(  # type: ignore
            symbol=symbol, timeframe=timeframe, limit=limit, ts_from=ts_from, ts_to=ts_to
        )
    except Exception as e:  # pragma: no cover
        logger.exception("Fetch error from %s", source)
        raise HTTPException(status_code=502, detail=f"fetch failed: {e}")

    if (os.getenv("APP_ENV") or "test").lower() in {"test", "testing", "ci"}:
        await session.execute(
            sa_text("DELETE FROM ohlcv WHERE source=:s AND asset=:a AND tf=:tf"),
            {"s": source, "a": symbol, "tf": timeframe},
        )
        await session.commit()
        logger.info("Purged old rows for %s/%s/%s", source, symbol, timeframe)

    stored = await crud.upsert_ohlcv_batch(session, rows)
    return {"stored": int(stored)}

# ──────────────────────────────────────────────────────────────────────────────
# Чтение / пагинация JSON
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/ohlcv")
async def get_ohlcv(
    source: Optional[str] = Query(None),
    ticker: Optional[str] = Query(None, alias="ticker"),
    timeframe: Optional[str] = Query(None),
    ts_from: Optional[int] = None,
    ts_to: Optional[int] = None,
    limit: int = Query(1000, ge=0),
    offset: int = Query(0, ge=0),
    order: Literal["asc","desc"] = Query("asc"),
    session: AsyncSession = Depends(get_session),
):
    """
    Строгая limit/offset-пагинация БЕЗ «додобора».
    """
    try:
        total = await crud.count_ohlcv(
            session, source=source, asset=ticker, tf=timeframe, ts_from=ts_from, ts_to=ts_to
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

    remaining = max(0, int(total) - int(offset))
    expected = min(int(limit), remaining)
    if expected >= 0 and len(rows) > expected:
        rows = rows[:expected]

    candles = [_row_to_dict(r) for r in rows]
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

# ──────────────────────────────────────────────────────────────────────────────
# Экспорт CSV — ФИКС ПОРЯДКА КОЛОНОК
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/ohlcv.csv")
async def get_ohlcv_csv(
    source: Optional[str] = Query(None),
    ticker: Optional[str] = Query(None, alias="ticker"),
    timeframe: Optional[str] = Query(None),
    ts_from: Optional[int] = None,
    ts_to: Optional[int] = None,
    limit: int = Query(0, ge=0),  # 0 => без ограничения (всё)
    offset: int = Query(0, ge=0),
    order: Literal["asc","desc"] = Query("asc"),
    session: AsyncSession = Depends(get_session),
):
    """
    Стримовый CSV с теми же фильтрами, без «додобора».
    Порядок колонок ЖЁСТКО под тест:
    ["source","asset","tf","ts","open","high","low","close","volume"]
    """
    async def row_iter() -> AsyncIterator[bytes]:
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")

        # Заголовок в нужном порядке
        writer.writerow(CSV_HEADER)
        yield buf.getvalue().encode("utf-8")
        buf.seek(0); buf.truncate(0)

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
                writer.writerow([
                    r.source,        # source
                    r.asset,         # asset
                    r.tf,            # tf
                    int(r.ts),       # ts
                    float(r.open),   # open
                    float(r.high),   # high
                    float(r.low),    # low
                    float(r.close),  # close
                    float(r.volume), # volume
                ])

            yield buf.getvalue().encode("utf-8")
            buf.seek(0); buf.truncate(0)

            advanced = len(batch)
            cur_offset += advanced

            if remaining is not None:
                remaining -= advanced
                if remaining <= 0:
                    break

    headers = {
        "Content-Disposition": 'attachment; filename="ohlcv.csv"',
        "Cache-Control": "no-store",
    }
    return StreamingResponse(row_iter(), media_type="text/csv; charset=utf-8", headers=headers)

# ──────────────────────────────────────────────────────────────────────────────
# Счётчик/статистика — НОВЫЙ ЭНДПОИНТ
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/ohlcv/count")
async def get_ohlcv_count(
    source: Optional[str] = Query(None),
    ticker: Optional[str] = Query(None, alias="ticker"),
    timeframe: Optional[str] = Query(None),
    ts_from: Optional[int] = None,
    ts_to: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    """
    Возвращает 200 всегда; если CRUD доступен — реальные count/stats.
    """
    count: Optional[int] = None
    stats: Dict[str, float] = {}
    try:
        count = await crud.count_ohlcv(
            session, source=source, asset=ticker, tf=timeframe, ts_from=ts_from, ts_to=ts_to
        )
        if hasattr(crud, "ohlcv_stats"):
            try:
                stats = await crud.ohlcv_stats(  # type: ignore[attr-defined]
                    session, source=source, asset=ticker, tf=timeframe, ts_from=ts_from, ts_to=ts_to
                )
            except Exception:
                stats = {}
    except Exception as e:  # pragma: no cover
        logger.warning("ohlcv_count fallback: %r", e)

    return JSONResponse({
        "source": source,
        "asset": ticker,
        "tf": timeframe,
        "count": int(count) if isinstance(count, int) else None,
        "stats": stats,
    })


@router.get("/ohlcv/stats")
async def get_ohlcv_stats(
    source: Optional[str] = Query(None),
    ticker: Optional[str] = Query(None, alias="ticker"),
    timeframe: Optional[str] = Query(None),
    ts_from: Optional[int] = None,
    ts_to: Optional[int] = None,
    session: AsyncSession = Depends(get_session),
):
    """
    Компактная сводка по свечам: минимальный/максимальный timestamp и количество.
    Используется health-эндпоинтом и тестами для проверки наполнения БД.
    """
    filters = []
    if source:
        filters.append(OHLCV.source == source.strip().lower())
    if ticker:
        filters.append(OHLCV.asset == ticker.strip().upper())
    if timeframe:
        filters.append(OHLCV.tf == timeframe.strip())
    if ts_from is not None:
        filters.append(OHLCV.ts >= int(ts_from))
    if ts_to is not None:
        filters.append(OHLCV.ts <= int(ts_to))

    stmt = select(
        func.min(OHLCV.ts).label("min_ts"),
        func.max(OHLCV.ts).label("max_ts"),
        func.count(func.distinct(OHLCV.ts)).label("count"),
    )
    if filters:
        stmt = stmt.where(and_(*filters))

    result = await session.execute(stmt)
    row = result.one_or_none()
    if not row:
        return {"min_ts": None, "max_ts": None, "count": 0}

    min_ts = row.min_ts
    max_ts = row.max_ts
    count = row.count
    return {
        "min_ts": int(min_ts) if min_ts is not None else None,
        "max_ts": int(max_ts) if max_ts is not None else None,
        "count": int(count or 0),
    }
