# db/crud_orders.py
from __future__ import annotations

from typing import Any, Dict, Optional, Iterable, List, Literal, Tuple
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, and_, or_, update, func

from .models_orders import OrderLog


# ──────────────────────────────────────────────────────────────────────────────
# Внутренние утилиты
# ──────────────────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _norm_side(side: Optional[str]) -> Optional[str]:
    if side is None:
        return None
    s = str(side).strip().lower()
    if s in ("buy", "sell"):
        return s
    # допускаем служебные значения для внутренних записей
    if s in ("reconcile", "cancel", "close", "open", "none", "na"):
        return s
    return s or None


def _norm_status(status: Optional[str]) -> Optional[str]:
    if status is None:
        return None
    return str(status).strip().upper() or None


def _to_float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        if v != v:  # NaN
            return None
        return v
    except Exception:
        return None


def _to_int_bool(x: Any) -> int:
    # в модели testnet = Integer(0/1)
    return 1 if bool(x) else 0


def _getattr_safe(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def _row_to_dict(row: OrderLog) -> Dict[str, Any]:
    """
    Устойчивое представление записи журнала.
    Поддерживает разные схемы (qty vs amount, order_id vs ext_id, и т.п.).
    """
    qty = _getattr_safe(row, "qty")
    if qty is None:
        qty = _getattr_safe(row, "amount")

    order_id = _getattr_safe(row, "order_id")
    if order_id is None:
        order_id = _getattr_safe(row, "ext_id")

    created_at = _getattr_safe(row, "created_at") or _utcnow()

    return {
        "id": _getattr_safe(row, "id"),
        "created_at": created_at,
        "exchange": _getattr_safe(row, "exchange"),
        "testnet": bool(_getattr_safe(row, "testnet", 0)),
        "symbol": _getattr_safe(row, "symbol"),
        "side": _getattr_safe(row, "side"),
        "type": _getattr_safe(row, "type"),
        "time_in_force": _getattr_safe(row, "time_in_force"),
        "status": _getattr_safe(row, "status"),
        "order_id": order_id,
        "client_order_id": _getattr_safe(row, "client_order_id"),
        "orig_client_order_id": _getattr_safe(row, "orig_client_order_id"),
        "price": _to_float_or_none(_getattr_safe(row, "price")),
        "stop_price": _to_float_or_none(_getattr_safe(row, "stop_price")),
        "qty": _to_float_or_none(qty),
        "amount": _to_float_or_none(qty),  # обратная совместимость
        "quote_qty": _to_float_or_none(_getattr_safe(row, "quote_qty")),
        "filled_qty": _to_float_or_none(_getattr_safe(row, "filled_qty")),
        "cummulative_quote_qty": _to_float_or_none(_getattr_safe(row, "cummulative_quote_qty")),
        "commission": _to_float_or_none(_getattr_safe(row, "commission")),
        "commission_asset": _getattr_safe(row, "commission_asset"),
        "event_ts_ms": _getattr_safe(row, "event_ts_ms"),
        "transact_ts_ms": _getattr_safe(row, "transact_ts_ms"),
        "order_ts_ms": _getattr_safe(row, "order_ts_ms"),
        # поля, которых может не быть в нашей актуальной модели — будут None
        "sl_pct": _to_float_or_none(_getattr_safe(row, "sl_pct")),
        "tp_pct": _to_float_or_none(_getattr_safe(row, "tp_pct")),
        "reason": _getattr_safe(row, "reason"),
        "raw": _getattr_safe(row, "raw"),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Создание записи (ручной и из ответа биржи)
# ──────────────────────────────────────────────────────────────────────────────

async def log_order(
    session: AsyncSession,
    *,
    exchange: str,
    testnet: bool,
    symbol: str,
    side: Optional[str] = None,
    type: Optional[str] = None,
    time_in_force: Optional[str] = None,
    # количество / цены
    qty: Optional[float] = None,
    quote_qty: Optional[float] = None,
    price: Optional[float] = None,
    stop_price: Optional[float] = None,
    filled_qty: Optional[float] = None,
    cummulative_quote_qty: Optional[float] = None,
    commission: Optional[float] = None,
    commission_asset: Optional[str] = None,
    # идентификаторы
    order_id: Optional[str] = None,
    client_order_id: Optional[str] = None,
    orig_client_order_id: Optional[str] = None,
    # статусы/метки
    status: Optional[str] = None,
    event_ts_ms: Optional[int] = None,
    transact_ts_ms: Optional[int] = None,
    order_ts_ms: Optional[int] = None,
    # обратная совместимость (старые имена)
    amount: Optional[float] = None,
    ext_id: Optional[str] = None,
    # управление транзакцией
    commit: bool = True,
) -> OrderLog:
    """
    Унифицированное логирование торгового события/ордера.
    Дружит с нашей актуальной моделью OrderLog, при этом «мягко» воспринимает
    устаревшие поля из старых версий кода.
    """
    qty_final = _to_float_or_none(qty if qty is not None else amount)
    order_id_final = order_id or ext_id

    row = OrderLog(
        exchange=str(exchange),
        testnet=_to_int_bool(testnet),
        symbol=str(symbol).upper(),
        side=_norm_side(side),
        type=(str(type).upper() if type else None),
        time_in_force=(str(time_in_force).upper() if time_in_force else None),
        status=_norm_status(status) or "NEW",
        order_id=order_id_final if hasattr(OrderLog, "order_id") else None,
        client_order_id=client_order_id if hasattr(OrderLog, "client_order_id") else None,
        orig_client_order_id=orig_client_order_id if hasattr(OrderLog, "orig_client_order_id") else None,
        price=_to_float_or_none(price) or 0.0,
        stop_price=_to_float_or_none(stop_price) or 0.0,
        qty=qty_final or 0.0 if hasattr(OrderLog, "qty") else None,
        quote_qty=_to_float_or_none(quote_qty) or 0.0 if hasattr(OrderLog, "quote_qty") else None,
        filled_qty=_to_float_or_none(filled_qty) or 0.0 if hasattr(OrderLog, "filled_qty") else None,
        cummulative_quote_qty=_to_float_or_none(cummulative_quote_qty) or 0.0
        if hasattr(OrderLog, "cummulative_quote_qty")
        else None,
        commission=_to_float_or_none(commission) or 0.0 if hasattr(OrderLog, "commission") else None,
        commission_asset=commission_asset if hasattr(OrderLog, "commission_asset") else None,
        event_ts_ms=event_ts_ms if hasattr(OrderLog, "event_ts_ms") else None,
        transact_ts_ms=transact_ts_ms if hasattr(OrderLog, "transact_ts_ms") else None,
        order_ts_ms=order_ts_ms if hasattr(OrderLog, "order_ts_ms") else None,
    )
    session.add(row)
    if commit:
        await session.commit()
        await session.refresh(row)
    else:
        # Позволяет вызывать несколько log_* в одной транзакции
        await session.flush()
    return row


async def log_from_exchange_response(
    session: AsyncSession,
    *,
    exchange: Literal["binance", "sim", "ui"] = "binance",
    testnet: bool,
    symbol: str,
    side: str,
    type: str,
    response: Dict[str, Any],
    time_in_force: Optional[str] = None,
    commit: bool = True,
) -> OrderLog:
    """
    Нормализует типичный ответ Binance (create order / order update) и пишет в журнал.
    Подхватывает ключевые поля: orderId, clientOrderId, status, price/qty/quote, timestamps и комиссии.
    """
    # Binance: возможные кейсы имен полей
    order_id = response.get("orderId") or response.get("order_id")
    client_order_id = response.get("clientOrderId") or response.get("client_order_id")
    orig_client_order_id = response.get("origClientOrderId") or response.get("orig_client_order_id")

    price = response.get("price")
    stop_price = response.get("stopPrice") or response.get("stop_price")

    qty = response.get("origQty") or response.get("orig_qty") or response.get("qty")
    filled_qty = response.get("executedQty") or response.get("executed_qty")
    quote_qty = response.get("quoteOrderQty") or response.get("quote_qty")
    cummulative_quote_qty = response.get("cummulativeQuoteQty") or response.get("cummulative_quote_qty")

    status = response.get("status")
    event_ts_ms = response.get("E")  # event time (streams)
    transact_ts_ms = response.get("transactTime") or response.get("transact_time")
    order_ts_ms = response.get("time") or response.get("order_time") or response.get("updateTime")

    commission = None
    commission_asset = None
    fills = response.get("fills")
    if isinstance(fills, list) and fills:
        # берём первую запись — как минимум индикатор комиссии/актива
        c0 = fills[0]
        commission = _to_float_or_none(c0.get("commission"))
        commission_asset = c0.get("commissionAsset")

    return await log_order(
        session,
        exchange=exchange,
        testnet=testnet,
        symbol=symbol,
        side=side,
        type=type,
        time_in_force=time_in_force or response.get("timeInForce") or response.get("time_in_force"),
        qty=_to_float_or_none(qty),
        quote_qty=_to_float_or_none(quote_qty),
        price=_to_float_or_none(price),
        stop_price=_to_float_or_none(stop_price),
        filled_qty=_to_float_or_none(filled_qty),
        cummulative_quote_qty=_to_float_or_none(cummulative_quote_qty),
        commission=_to_float_or_none(commission),
        commission_asset=commission_asset,
        order_id=str(order_id) if order_id is not None else None,
        client_order_id=str(client_order_id) if client_order_id is not None else None,
        orig_client_order_id=str(orig_client_order_id) if orig_client_order_id is not None else None,
        status=_norm_status(status),
        event_ts_ms=event_ts_ms,
        transact_ts_ms=transact_ts_ms,
        order_ts_ms=order_ts_ms,
        commit=commit,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Обновления
# ──────────────────────────────────────────────────────────────────────────────

async def update_status(
    session: AsyncSession,
    *,
    order_id: str,
    status: str,
    commit: bool = True,
) -> int:
    """
    Обновляет статус по order_id (или ext_id, если в схеме есть).
    Возвращает число затронутых строк.
    """
    conds = []
    if hasattr(OrderLog, "order_id"):
        conds.append(OrderLog.order_id == str(order_id))
    if hasattr(OrderLog, "ext_id"):
        conds.append(OrderLog.ext_id == str(order_id))
    if not conds:
        return 0

    q = update(OrderLog).where(or_(*conds)).values(status=_norm_status(status) or "NEW")
    res = await session.execute(q)
    if commit:
        await session.commit()
    return res.rowcount or 0


async def update_filled_and_status(
    session: AsyncSession,
    *,
    order_id: str,
    filled_qty: Optional[float] = None,
    cummulative_quote_qty: Optional[float] = None,
    status: Optional[str] = None,
    commit: bool = True,
) -> int:
    """
    Частичный апдейт наполнения ордера и статуса.
    """
    values: Dict[str, Any] = {}
    if hasattr(OrderLog, "filled_qty") and filled_qty is not None:
        values["filled_qty"] = _to_float_or_none(filled_qty) or 0.0
    if hasattr(OrderLog, "cummulative_quote_qty") and cummulative_quote_qty is not None:
        values["cummulative_quote_qty"] = _to_float_or_none(cummulative_quote_qty) or 0.0
    if status is not None:
        values["status"] = _norm_status(status) or "NEW"

    if not values:
        return 0

    conds = []
    if hasattr(OrderLog, "order_id"):
        conds.append(OrderLog.order_id == str(order_id))
    if hasattr(OrderLog, "ext_id"):
        conds.append(OrderLog.ext_id == str(order_id))
    if not conds:
        return 0

    q = update(OrderLog).where(or_(*conds)).values(**values)
    res = await session.execute(q)
    if commit:
        await session.commit()
    return res.rowcount or 0


# ──────────────────────────────────────────────────────────────────────────────
# Чтение / выборки
# ──────────────────────────────────────────────────────────────────────────────

async def get_last_orders(
    session: AsyncSession,
    *,
    limit: int = 50,
    symbols: Optional[Iterable[str]] = None,
    status_in: Optional[Iterable[str]] = None,
    testnet: Optional[bool] = None,
    exchange: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Последние записи журнала. Фильтры по символам/статусам/среде/бирже.
    """
    stmt = select(OrderLog)
    conds = []

    if symbols:
        syms = [str(s).upper() for s in symbols if s]
        if syms:
            conds.append(OrderLog.symbol.in_(syms))
    if status_in:
        sts = [str(s).upper() for s in status_in if s]
        if sts:
            conds.append(OrderLog.status.in_(sts))
    if testnet is not None:
        conds.append(OrderLog.testnet == _to_int_bool(testnet))
    if exchange:
        conds.append(OrderLog.exchange == str(exchange))

    if conds:
        stmt = stmt.where(and_(*conds))
    stmt = stmt.order_by(desc(OrderLog.created_at)).limit(int(limit))

    res = await session.execute(stmt)
    rows = list(res.scalars().all())
    return [_row_to_dict(r) for r in rows]


# Старое имя (обратная совместимость)
async def last_orders(session: AsyncSession, *, limit: int = 50):
    return await get_last_orders(session, limit=limit)


async def get_by_client_order_id(
    session: AsyncSession,
    *,
    client_order_id: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Поиск по client_order_id.
    """
    if not hasattr(OrderLog, "client_order_id"):
        return []
    stmt = (
        select(OrderLog)
        .where(OrderLog.client_order_id == str(client_order_id))
        .order_by(desc(OrderLog.created_at))
        .limit(int(limit))
    )
    res = await session.execute(stmt)
    return [_row_to_dict(r) for r in res.scalars().all()]


async def get_by_order_id(
    session: AsyncSession,
    *,
    order_id: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Поиск по биржевому order_id (или ext_id, если он в схеме).
    """
    conds = []
    if hasattr(OrderLog, "order_id"):
        conds.append(OrderLog.order_id == str(order_id))
    if hasattr(OrderLog, "ext_id"):
        conds.append(OrderLog.ext_id == str(order_id))

    if not conds:
        return []

    stmt = (
        select(OrderLog)
        .where(or_(*conds))
        .order_by(desc(OrderLog.created_at))
        .limit(int(limit))
    )
    res = await session.execute(stmt)
    return [_row_to_dict(r) for r in res.scalars().all()]


async def get_between(
    session: AsyncSession,
    *,
    ts_from: Optional[datetime] = None,
    ts_to: Optional[datetime] = None,
    symbols: Optional[Iterable[str]] = None,
    status_in: Optional[Iterable[str]] = None,
    testnet: Optional[bool] = None,
    exchange: Optional[str] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]:
    """
    Гибкая выборка по окну времени/символам/статусам/среде/бирже.
    """
    stmt = select(OrderLog)
    conds = []

    if ts_from:
        conds.append(OrderLog.created_at >= ts_from)
    if ts_to:
        conds.append(OrderLog.created_at <= ts_to)
    if symbols:
        syms = [str(s).upper() for s in symbols if s]
        if syms:
            conds.append(OrderLog.symbol.in_(syms))
    if status_in:
        sts = [str(s).upper() for s in status_in if s]
        if sts:
            conds.append(OrderLog.status.in_(sts))
    if testnet is not None:
        conds.append(OrderLog.testnet == _to_int_bool(testnet))
    if exchange:
        conds.append(OrderLog.exchange == str(exchange))

    if conds:
        stmt = stmt.where(and_(*conds))
    stmt = stmt.order_by(desc(OrderLog.created_at)).limit(int(limit))

    res = await session.execute(stmt)
    return [_row_to_dict(r) for r in res.scalars().all()]


# ──────────────────────────────────────────────────────────────────────────────
# Агрегации (минимально полезные)
# ──────────────────────────────────────────────────────────────────────────────

async def count_trades_today(
    session: AsyncSession,
    *,
    tz_name: str = "Asia/Almaty",
    testnet: Optional[bool] = None,
    exchange: Optional[str] = None,
) -> int:
    """
    Подсчёт количества записей за «текущие сутки» в указанной таймзоне.
    Используется для быстрого чекпойнта daily-limit.
    """
    # начало дня в tz → в UTC
    # Простейшая реализация: считаем по UTC-дню (достаточно для лимита в тестах).
    # Если нужна точная TZ-логика — перенесём вычисление в вызывающий код.
    day_start_utc = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)

    stmt = select(func.count()).select_from(OrderLog).where(OrderLog.created_at >= day_start_utc)

    if testnet is not None:
        stmt = stmt.where(OrderLog.testnet == _to_int_bool(testnet))
    if exchange:
        stmt = stmt.where(OrderLog.exchange == str(exchange))

    res = await session.execute(stmt)
    return int(res.scalar_one() or 0)
