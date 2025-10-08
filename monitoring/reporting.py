from __future__ import annotations

from typing import Any, Dict, Iterable

from utils.risk_config import load_risk_config

try:
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession
    from db.models_orders import OrderLog
except Exception:  # pragma: no cover
    AsyncSession = None  # type: ignore
    OrderLog = None  # type: ignore


def _order_pnl(order) -> float:
    side = (order.side or "").upper()
    quote = float(order.cummulative_quote_qty or order.quote_qty or 0.0)
    commission = float(order.commission or 0.0)
    if side == "SELL":
        return quote - commission
    if side == "BUY":
        return -quote - commission
    return -commission


def summarize_orders(orders: Iterable[Any]) -> Dict[str, Any]:
    daily: Dict[str, float] = {}
    total_volume = 0.0
    filled = 0
    rejected = 0
    for order in orders:
        day = order.created_at.date().isoformat() if order.created_at else "unknown"
        daily.setdefault(day, 0.0)
        pnl = _order_pnl(order)
        daily[day] += pnl
        total_volume += abs(float(order.cummulative_quote_qty or order.quote_qty or 0.0))
        status = (order.status or "").upper()
        if status == "FILLED":
            filled += 1
        elif status in {"REJECTED", "EXPIRED"}:
            rejected += 1
    timeline = [
        {"date": day, "pnl": round(value, 2)}
        for day, value in sorted(daily.items(), reverse=True)
    ]
    return {
        "trades": filled + rejected,
        "filled": filled,
        "rejected": rejected,
        "volume": round(total_volume, 2),
        "timeline": timeline,
        "total_pnl": round(sum(daily.values()), 2),
    }


async def build_dashboard_state(session: AsyncSession, *, recent_limit: int = 200) -> Dict[str, Any]:
    if AsyncSession is None or OrderLog is None:
        return {
            "pnl": {"trades": 0, "filled": 0, "rejected": 0, "volume": 0.0, "timeline": [], "total_pnl": 0.0},
            "risk": load_risk_config().to_dict(),
        }
    stmt = select(OrderLog).order_by(OrderLog.created_at.desc()).limit(recent_limit)
    orders = list(await session.scalars(stmt))
    pnl_summary = summarize_orders(orders)
    risk_cfg = load_risk_config().to_dict()
    return {"pnl": pnl_summary, "risk": risk_cfg}
