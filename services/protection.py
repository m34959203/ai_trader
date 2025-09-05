from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Literal

from db.session import get_session
from db import crud_orders

from executors.api_binance import BinanceExecutor

LOG = logging.getLogger("ai_trader.protection")

async def _select_executor(mode: Literal["binance","sim"], testnet: bool) -> BinanceExecutor:
    # из существующей логики — создаём такой же инстанс
    return BinanceExecutor(testnet=testnet)

async def _pending_entries(limit: int = 300) -> List[Dict[str, Any]]:
    """
    Возвращает последние записи из журнала, где указано protection_pending=True.
    Эти записи пишутся в raw.response внутри /exec/open.
    """
    async with get_session() as session:
        rows = await crud_orders.get_last_orders(session, limit=limit)  # уже есть в проекте
    out: List[Dict[str, Any]] = []
    for r in rows:
        raw = r.get("raw") or {}
        resp = raw.get("response") or {}
        if resp.get("protection_pending") is True:
            out.append(r)
    return out

async def protection_monitor(
    *,
    mode: Literal["binance","sim"] = "binance",
    testnet: bool = True,
    interval_sec: int = 30,
) -> None:
    """
    Раз в interval_sec:
      1) ищем в журнале pending-заказы (LIMIT entry без защиты),
      2) у Binance узнаём статус ордера; если FILLED — ставим SL/TP (или OCO),
      3) логируем результат в order_log.
    """
    LOG.info("Protection monitor: start (mode=%s testnet=%s interval=%ss)", mode, testnet, interval_sec)
    ex = await _select_executor(mode, testnet)

    try:
        while True:
            try:
                pend = await _pending_entries(limit=300)
                if not pend:
                    await asyncio.sleep(interval_sec)
                    continue

                for row in pend:
                    sym = str(row.get("symbol")).upper()
                    raw = row.get("raw") or {}
                    req = (raw.get("request") or {})
                    resp = (raw.get("response") or {})
                    entry_order_id = resp.get("order_id") or resp.get("raw", {}).get("orderId") or resp.get("clientOrderId") or req.get("client_order_id")
                    side = str(row.get("side") or req.get("side") or "buy").lower()

                    # узнаём фактический статус и исполненный объём
                    try:
                        ord_info = await ex.client.get_order(symbol=sym, origClientOrderId=str(entry_order_id))  # по clientOrderId надёжно
                    except Exception:
                        # fallback: если clientOrderId не подошёл, пробуем по orderId
                        try:
                            oid = int(entry_order_id) if str(entry_order_id).isdigit() else None
                            ord_info = await ex.client.get_order(symbol=sym, orderId=oid) if oid else {}
                        except Exception as e2:
                            LOG.warning("get_order failed for %s [%s]: %r", sym, entry_order_id, e2)
                            continue

                    status = (ord_info.get("status") or "").upper()
                    filled = float(ord_info.get("executedQty") or 0.0)
                    if status not in ("FILLED", "PARTIALLY_FILLED") or filled <= 0:
                        continue  # ждём исполнения

                    # считаем цены защиты — берём из raw.request, где мы уже положили sl_price/sl_pct/tp_price/tp_pct
                    sl_price = req.get("sl_price")
                    tp_price = req.get("tp_price")
                    if sl_price is None and (req.get("sl_pct") is not None):
                        # часть исполнена — прикидываем от цены ордера, либо от последней цены
                        px = float(ord_info.get("price") or req.get("price") or 0.0)
                        if px > 0:
                            sl_price = px * (1.0 - float(req["sl_pct"])) if side == "buy" else px * (1.0 + float(req["sl_pct"]))
                    if tp_price is None and (req.get("tp_pct") is not None):
                        px = float(ord_info.get("price") or req.get("price") or 0.0)
                        if px > 0:
                            tp_price = px * (1.0 + float(req["tp_pct"])) if side == "buy" else px * (1.0 - float(req["tp_pct"]))

                    # ставим защиту на ИСПОЛНЁННЫЙ объём
                    try:
                        prot = await ex.place_protection_orders(
                            symbol=sym, side=side, qty=float(filled), sl_price=sl_price, tp_price=tp_price
                        )
                    except Exception as pe:
                        LOG.error("place_protection failed %s: %r", sym, pe)
                        prot = {"error": str(pe)}

                    # пишем служебную запись в журнал
                    async with get_session() as session:
                        try:
                            await crud_orders.log_order(
                                session=session,
                                exchange=("binance" if mode=="binance" else "sim"),
                                testnet=bool(testnet or mode=="sim"),
                                symbol=sym,
                                side=None,
                                type=None,
                                qty=float(filled),
                                price=None,
                                status="RECONCILE" if prot.get("error") else "NEW",
                                order_id=str(entry_order_id),
                                client_order_id=None,
                                reason="place_protection",
                                raw={"from_entry": row, "protection": prot, "ts_ms": int(time.time()*1000)},
                            )
                        except Exception as le:
                            LOG.warning("order log failed (protection): %r", le)

                await asyncio.sleep(interval_sec)

            except Exception as loop_e:
                LOG.warning("Protection monitor loop error: %r", loop_e)
                await asyncio.sleep(interval_sec)
    finally:
        try:
            await ex.close()
        except Exception:
            pass
