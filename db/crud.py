from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence, List, Dict, Any
from decimal import Decimal


from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select, and_, func, insert, update, text as sa_text, literal_column

from .models import OHLCV

logger = logging.getLogger(__name__)

_BATCH_SIZE = 1000  # Увеличен размер батча для лучшей производительности

# Определяем ожидаемую структуру данных для type safety
OHLCVRow = Dict[str, Any]

def _build_filters(
    *,
    source: Optional[str] = None,
    asset: Optional[str] = None,
    tf: Optional[str] = None,
    ts_from: Optional[int] = None,
    ts_to: Optional[int] = None,
) -> Optional[Any]:
    """
    Единая точка сборки WHERE-условий с улучшенной типобезопасностью.
    
    Returns:
        SQLAlchemy expression или None если условий нет
    """
    conditions = []
    if source:
        conditions.append(OHLCV.source == source)
    if asset:
        conditions.append(OHLCV.asset == asset.upper())  # Нормализация символа
    if tf:
        conditions.append(OHLCV.tf == tf)
    if ts_from is not None:
        conditions.append(OHLCV.ts >= ts_from)
    if ts_to is not None:
        conditions.append(OHLCV.ts <= ts_to)
    
    return and_(*conditions) if conditions else None


def _validate_and_transform_ohlcv_row(row: OHLCVRow) -> Optional[OHLCVRow]:
    """
    Валидация и трансформация строки OHLCV данных.
    
    Returns:
        Валидированный словарь или None при ошибке
    """
    required_fields = {"source", "asset", "tf", "ts", "open", "high", "low", "close", "volume"}
    
    # Проверка наличия всех обязательных полей
    if not required_fields.issubset(row.keys()):
        logger.warning(f"Пропущена строка с отсутствующими полями: {set(row.keys()) - required_fields}")
        return None
    
    try:
        # Преобразование и валидация типов
        transformed = {
            "source": str(row["source"]).strip(),
            "asset": str(row["asset"]).upper().strip(),
            "tf": str(row["tf"]).strip(),
            "ts": int(row["ts"]),
            "open": float(Decimal(str(row["open"]))),  # Decimal для точности
            "high": float(Decimal(str(row["high"]))),
            "low": float(Decimal(str(row["low"]))),
            "close": float(Decimal(str(row["close"]))),
            "volume": float(Decimal(str(row["volume"]))),
        }
        
        # Дополнительная валидация значений
        if transformed["ts"] <= 0:
            logger.warning(f"Некорректная временная метка: {transformed['ts']}")
            return None
            
        if any(price <= 0 for price in [transformed["open"], transformed["high"], 
                                       transformed["low"], transformed["close"]]):
            logger.warning(f"Обнаружена некорректная цена в строке: {transformed}")
            return None
            
        return transformed
        
    except (ValueError, TypeError, Decimal.InvalidOperation) as e:
        logger.warning(f"Ошибка преобразования данных: {e}, строка: {row}")
        return None


async def upsert_ohlcv_batch(
    session: AsyncSession, 
    rows: Iterable[OHLCVRow],
    auto_commit: bool = True
) -> int:
    """
    Вставляет/обновляет свечи батчами с использованием SQLAlchemy Core.
    
    Args:
        session: Асинхронная сессия БД
        rows: Итерируемый объект со словарями данных
        auto_commit: Автоматически коммитить транзакцию
        
    Returns:
        Количество успешно обработанных строк
    """
    processed = 0
    batch_buffer: List[OHLCVRow] = []
    
    async def _flush_batch() -> None:
        """Формирует и выполняет UPSERT для текущего батча."""
        nonlocal processed
        if not batch_buffer:
            return
            
        try:
            # Используем SQLAlchemy Core для UPSERT :cite[6]
            stmt = sqlite_insert(OHLCV).values(batch_buffer)
            stmt = stmt.on_conflict_do_update(
                index_elements=["source", "asset", "tf", "ts"],  # UNIQUE constraint
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                }
            )
            
            result = await session.execute(stmt)
            processed += len(batch_buffer)
            batch_buffer.clear()
            logger.debug(f"Успешно обработан батч из {len(batch_buffer)} строк")
            
        except SQLAlchemyError as e:
            logger.error(f"Ошибка при вставке батча: {e}")
            await session.rollback()
            raise
    
    try:
        # Обработка входных данных
        for row in rows:
            validated_row = _validate_and_transform_ohlcv_row(row)
            if validated_row is None:
                continue
                
            batch_buffer.append(validated_row)
            
            if len(batch_buffer) >= _BATCH_SIZE:
                await _flush_batch()
        
        # Финальный flush
        await _flush_batch()
        
        if auto_commit:
            await session.commit()
            
        logger.info(f"Всего обработано строк: {processed}")
        return processed
        
    except Exception as e:
        logger.error(f"Критическая ошибка в upsert_ohlcv_batch: {e}")
        await session.rollback()
        raise


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
    order: str = "asc",
) -> Sequence[OHLCV]:
    """
    Постраничная выборка по УНИКАЛЬНЫМ ts:
      1) собираем страницу меток времени (DISTINCT ts) с order/offset/limit;
      2) для каждого ts берём одну «каноническую» строку (минимальный rowid, SQLite);
      3) возвращаем ORM-объекты в требуемом порядке.
    Так мы синхронизируемся с count_ohlcv(), который считает DISTINCT(ts).
    """
    if order not in ("asc", "desc"):
        raise ValueError("Параметр order должен быть 'asc' или 'desc'")

    # Строим общие фильтры
    cond = _build_filters(
        source=source, asset=asset, tf=tf,
        ts_from=ts_from, ts_to=ts_to
    )

    # Таблица и сортировка
    t = OHLCV.__table__
    ts_order = t.c.ts.desc() if order == "desc" else t.c.ts.asc()

    try:
        # 1) Страница уникальных ts с нужным order/offset/limit
        ts_page_q = select(t.c.ts)
        if cond is not None:
            ts_page_q = ts_page_q.where(cond)
        ts_page_sub = (
            ts_page_q.group_by(t.c.ts)
            .order_by(ts_order)
            .offset(int(offset) if offset else 0)
            .limit(int(limit) if limit else None)
            .subquery()
        )

        # 2) Для каждого ts из страницы берём одну каноническую запись: MIN(rowid)
        #    rowid есть в SQLite у таблиц без INTEGER PRIMARY KEY явного столбца.
        rowid_col = literal_column("rowid")
        min_rowid_q = select(func.min(rowid_col).label("rid")).select_from(t)
        in_ts_page = t.c.ts.in_(select(ts_page_sub.c.ts))
        if cond is not None:
            min_rowid_q = min_rowid_q.where(and_(cond, in_ts_page))
        else:
            min_rowid_q = min_rowid_q.where(in_ts_page)
        min_rowid_sub = min_rowid_q.group_by(t.c.ts).subquery()

        # 3) Выбираем сами ORM-объекты по найденным rowid и сортируем по ts
        final_q = (
            select(OHLCV)
            .where(rowid_col.in_(select(min_rowid_sub.c.rid)))
            .order_by(ts_order)
        )

        result = await session.execute(final_q)
        return result.scalars().all()

    except SQLAlchemyError as e:
        logger.error(f"Ошибка при запросе OHLCV данных: {e}")
        raise

async def count_ohlcv(
    session: AsyncSession,
    *,
    source: Optional[str] = None,
    asset: Optional[str] = None,
    tf: Optional[str] = None,
    ts_from: Optional[int] = None,
    ts_to: Optional[int] = None,
) -> int:
    """
    Подсчет уникальных временных меток с учетом фильтров.
    
    Returns:
        Количество уникальных временных меток
    """
    conditions = _build_filters(
        source=source, asset=asset, tf=tf,
        ts_from=ts_from, ts_to=ts_to
    )
    
    try:
        # Используем COUNT(DISTINCT) для точного подсчета уникальных ts
        stmt = select(func.count(func.distinct(OHLCV.ts)))
        
        if conditions is not None:
            stmt = stmt.where(conditions)
            
        result = await session.execute(stmt)
        count = result.scalar_one()
        return int(count)
        
    except SQLAlchemyError as e:
        logger.error(f"Ошибка при подсчете OHLCV: {e}")
        raise


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
    Расширенная статистика по OHLCV данным.
    
    Returns:
        Словарь с минимальной, максимальной временными метками и количеством
    """
    conditions = _build_filters(
        source=source, asset=asset, tf=tf,
        ts_from=ts_from, ts_to=ts_to
    )
    
    try:
        # Выполняем все агрегатные запросы в одной транзакции
        min_stmt = select(func.min(OHLCV.ts))
        max_stmt = select(func.max(OHLCV.ts)) 
        count_stmt = select(func.count(func.distinct(OHLCV.ts)))
        
        if conditions is not None:
            min_stmt = min_stmt.where(conditions)
            max_stmt = max_stmt.where(conditions)
            count_stmt = count_stmt.where(conditions)
        
        min_result = await session.execute(min_stmt)
        max_result = await session.execute(max_stmt)
        count_result = await session.execute(count_stmt)
        
        min_ts = min_result.scalar_one_or_none()
        max_ts = max_result.scalar_one_or_none()
        count = count_result.scalar_one()
        
        return {
            "min_ts": int(min_ts) if min_ts is not None else None,
            "max_ts": int(max_ts) if max_ts is not None else None,
            "count": int(count),
            "time_range": max_ts - min_ts if min_ts and max_ts else None,
        }
        
    except SQLAlchemyError as e:
        logger.error(f"Ошибка при получении статистики OHLCV: {e}")
        raise


async def get_ohlcv_paginated(
    session: AsyncSession,
    *,
    source: Optional[str] = None,
    asset: Optional[str] = None, 
    tf: Optional[str] = None,
    limit: int = 1000,
    offset: int = 0,
    order: str = "desc",
) -> Dict[str, Any]:
    """
    Универсальный метод для постраничного получения данных с метаинформацией.
    
    Returns:
        Словарь с данными, пагинацией и общей статистикой
    """
    # Получаем данные
    candles = await query_ohlcv(
        session=session,
        source=source,
        asset=asset,
        tf=tf,
        limit=limit,
        offset=offset,
        order=order
    )
    
    # Получаем общее количество
    total_count = await count_ohlcv(
        session=session,
        source=source, 
        asset=asset,
        tf=tf
    )
    
    # Рассчитываем next_offset для пагинации
    next_offset = offset + limit
    has_next_page = next_offset < total_count
    
    return {
        "candles": candles,
        "pagination": {
            "total": total_count,
            "limit": limit,
            "offset": offset,
            "next_offset": next_offset if has_next_page else None,
            "has_next_page": has_next_page,
            "total_pages": (total_count + limit - 1) // limit if limit > 0 else 0
        },
        "metadata": {
            "source": source,
            "asset": asset,
            "timeframe": tf,
            "retrieved_count": len(candles)
        }
    }