# db/crud.py
from __future__ import annotations

from typing import Iterable, Optional, Sequence, List, Dict

from sqlalchemy import select, and_, func, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from .models import OHLCV

_BATCH_SIZE = 500  # безопасный размер батча для SQLite


def _build_filters(
    *,
    source: Optional[str],
    asset: Optional[str],
    tf: Optional[str],
    ts_from: Optional[int],
    ts_to: Optional[int],
):
    conds = []
    if source:
        conds.append(OHLCV.source == source)
    if asset:
        conds.append(OHLCV.asset == asset)
    if tf:
        conds.append(OHLCV.tf == tf)
    if ts_from is not None:
        conds.append(OHLCV.ts >= ts_from)
    if ts_to is not None:
        conds.append(OHLCV.ts <= ts_to)
    return and_(*conds) if conds else None


async def upsert_ohlcv_batch(session: AsyncSession, rows: Iterable[Dict]) -> int:
    """
    Вставляет/обновляет свечи батчами.
    rows — iterable/список словарей с ключами ТОЧНО как в БД:
      source, asset, tf, ts, open, high, low, close, volume
    Возвращает количество успешно обработанных строк.
    """
    total = 0
    buf: List[Dict] = []

    UPSERT_SQL = """
    INSERT INTO ohlcv (source, asset, tf, ts, open, high, low, close, volume)
    VALUES (:source, :asset, :tf, :ts, :open, :high, :low, :close, :volume)
    ON CONFLICT (source, asset, tf, ts) DO UPDATE SET
        open   = excluded.open,
        high   = excluded.high,
        low    = excluded.low,
        close  = excluded.close,
        volume = excluded.volume
    """

    async def _flush():
        nonlocal total, buf
        if not buf:
            return
        await session.execute(sa_text(UPSERT_SQL), buf)
        total += len(buf)
        buf = []

    required = {"source", "asset", "tf", "ts", "open", "high", "low", "close", "volume"}

    for row in rows:
        if not required <= row.keys():
            continue
        try:
            buf.append(
                {
                    "source": str(row["source"]),
                    "asset": str(row["asset"]),
                    "tf": str(row["tf"]),
                    "ts": int(row["ts"]),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )
        except Exception:
            continue

        if len(buf) >= _BATCH_SIZE:
            await _flush()

    await _flush()
    await session.commit()
    return total


async def query_ohlcv(
    session: AsyncSession,
    *,
    source: Optional[str] = None,
    asset: Optional[str] = None,
    tf: Optional[str] = None,
    ts_from: Optional[int] = None,
    ts_to: Optional[int] = None,
    limit: int = 1000,
    offset: int = 0,
    order: str = "asc",  # "asc" | "desc"
) -> Sequence[OHLCV]:
    cond = _build_filters(source=source, asset=asset, tf=tf, ts_from=ts_from, ts_to=ts_to)
    stmt = select(OHLCV)
    if cond is not None:
        stmt = stmt.where(cond)
    stmt = stmt.order_by(OHLCV.ts.desc() if order == "desc" else OHLCV.ts.asc())
    if offset:
        stmt = stmt.offset(offset)
    if limit:
        stmt = stmt.limit(limit)
    res = await session.execute(stmt)
    return res.scalars().all()


async def count_ohlcv(
    session: AsyncSession,
    *,
    source: Optional[str] = None,
    asset: Optional[str] = None,
    tf: Optional[str] = None,
    ts_from: Optional[int] = None,
    ts_to: Optional[int] = None,
) -> int:
    cond = _build_filters(source=source, asset=asset, tf=tf, ts_from=ts_from, ts_to=ts_to)
    stmt = select(func.count()).select_from(OHLCV)
    if cond is not None:
        stmt = stmt.where(cond)
    res = await session.execute(stmt)
    return int(res.scalar_one())


async def stats_ohlcv(
    session: AsyncSession,
    *,
    source: Optional[str] = None,
    asset: Optional[str] = None,
    tf: Optional[str] = None,
    ts_from: Optional[int] = None,
    ts_to: Optional[int] = None,
) -> Dict[str, Optional[int]]:
    """
    Возвращает dict со сводной статистикой:
      {"min_ts": <int|None>, "max_ts": <int|None>, "count": <int>}
    """
    cond = _build_filters(source=source, asset=asset, tf=tf, ts_from=ts_from, ts_to=ts_to)
    stmt = select(func.min(OHLCV.ts), func.max(OHLCV.ts), func.count()).select_from(OHLCV)
    if cond is not None:
        stmt = stmt.where(cond)
    res = await session.execute(stmt)
    min_ts, max_ts, cnt = res.one()
    return {
        "min_ts": int(min_ts) if min_ts is not None else None,
        "max_ts": int(max_ts) if max_ts is not None else None,
        "count": int(cnt),
    }
