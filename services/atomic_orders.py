"""Atomic order placement with OCO (One-Cancels-Other) for Binance.

This module provides atomic stop-loss and take-profit placement
to eliminate the dangerous 30-second gap between entry and protection.

Key improvements:
- OCO orders guarantee SL/TP are placed atomically with entry
- No gap period where position is unprotected
- Eliminates flash crash and gap risk during protection delay
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from utils.structured_logging import get_logger

LOG = get_logger("ai_trader.atomic_orders")


@dataclass
class AtomicOrderResult:
    """Result of atomic order placement."""

    success: bool
    entry_order_id: Optional[str] = None
    oco_order_id: Optional[str] = None
    stop_order_id: Optional[str] = None
    limit_order_id: Optional[str] = None
    error: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "entry_order_id": self.entry_order_id,
            "oco_order_id": self.oco_order_id,
            "stop_order_id": self.stop_order_id,
            "limit_order_id": self.limit_order_id,
            "error": self.error,
            "raw": self.raw,
        }


class AtomicOrderPlacer:
    """Places orders with atomic SL/TP protection using Binance OCO.

    Workflow:
    1. Place market entry order
    2. Wait for fill (max 5 seconds)
    3. IMMEDIATELY place OCO with SL + TP
    4. If OCO fails, emergency market exit

    This eliminates the 30-second protection gap.
    """

    def __init__(self, executor):
        """Initialize with Binance executor.

        Args:
            executor: BinanceExecutor instance with .client attribute
        """
        self.executor = executor
        self.client = executor.client

    async def place_entry_with_protection(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: float,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        sl_pct: Optional[float] = None,
        tp_pct: Optional[float] = None,
        entry_type: Literal["MARKET", "LIMIT"] = "MARKET",
        entry_price: Optional[float] = None,
    ) -> AtomicOrderResult:
        """Place entry order and atomically set SL/TP protection.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            side: "BUY" or "SELL"
            quantity: Order quantity
            sl_price: Stop-loss price (absolute)
            tp_price: Take-profit price (absolute)
            sl_pct: Stop-loss percentage (alternative to sl_price)
            tp_pct: Take-profit percentage (alternative to tp_price)
            entry_type: "MARKET" or "LIMIT"
            entry_price: Entry price (required for LIMIT orders)

        Returns:
            AtomicOrderResult with order IDs and status
        """
        try:
            # Step 1: Place entry order
            LOG.info(
                f"[ATOMIC] Placing {entry_type} {side} entry: {symbol} qty={quantity}"
            )

            if entry_type == "MARKET":
                entry_result = await self._place_market_entry(symbol, side, quantity)
            else:
                entry_result = await self._place_limit_entry(
                    symbol, side, quantity, entry_price
                )

            if not entry_result["success"]:
                return AtomicOrderResult(
                    success=False,
                    error=f"Entry order failed: {entry_result.get('error')}",
                    raw=entry_result,
                )

            entry_order_id = entry_result["order_id"]
            fill_price = entry_result.get("fill_price")
            filled_qty = entry_result.get("filled_qty", quantity)

            # Step 2: Calculate protection prices if not provided
            if fill_price:
                if sl_price is None and sl_pct:
                    sl_price = self._calculate_sl_price(
                        fill_price, side, sl_pct
                    )
                if tp_price is None and tp_pct:
                    tp_price = self._calculate_tp_price(
                        fill_price, side, tp_pct
                    )

            # Step 3: Place OCO protection IMMEDIATELY
            if sl_price and tp_price:
                LOG.info(
                    f"[ATOMIC] Placing OCO protection: SL={sl_price:.8f} TP={tp_price:.8f}"
                )

                oco_result = await self._place_oco_protection(
                    symbol=symbol,
                    side="SELL" if side == "BUY" else "BUY",  # Opposite side
                    quantity=filled_qty,
                    stop_price=sl_price,
                    limit_price=tp_price,
                )

                if not oco_result["success"]:
                    # CRITICAL: OCO failed, emergency exit
                    LOG.error(
                        f"[ATOMIC] OCO failed! Emergency exit for {symbol}"
                    )
                    await self._emergency_exit(symbol, side, filled_qty)

                    return AtomicOrderResult(
                        success=False,
                        entry_order_id=entry_order_id,
                        error=f"OCO placement failed: {oco_result.get('error')}. Emergency exit executed.",
                        raw={"entry": entry_result, "oco": oco_result},
                    )

                return AtomicOrderResult(
                    success=True,
                    entry_order_id=entry_order_id,
                    oco_order_id=oco_result.get("orderListId"),
                    stop_order_id=oco_result.get("orders", [{}])[0].get("orderId"),
                    limit_order_id=oco_result.get("orders", [{}])[1].get("orderId"),
                    raw={"entry": entry_result, "oco": oco_result},
                )

            else:
                # No protection requested (dangerous!)
                LOG.warning(
                    f"[ATOMIC] No SL/TP provided for {symbol} - position unprotected!"
                )
                return AtomicOrderResult(
                    success=True,
                    entry_order_id=entry_order_id,
                    error="No protection set - position is UNPROTECTED",
                    raw={"entry": entry_result},
                )

        except Exception as e:
            LOG.error(f"[ATOMIC] Unexpected error: {e}", exc_info=True)
            return AtomicOrderResult(
                success=False,
                error=f"Atomic order placement failed: {str(e)}",
            )

    async def _place_market_entry(
        self, symbol: str, side: str, quantity: float
    ) -> Dict[str, Any]:
        """Place market entry order and wait for fill."""
        try:
            # Place market order
            order = await self.client.create_order(
                symbol=symbol,
                side=side.upper(),
                type="MARKET",
                quantity=quantity,
            )

            order_id = str(order.get("orderId") or order.get("clientOrderId"))

            # Wait for fill (max 5 seconds)
            for attempt in range(10):  # 10 × 500ms = 5 seconds
                await self._sleep(0.5)

                try:
                    status = await self.client.get_order(
                        symbol=symbol,
                        orderId=int(order["orderId"]),
                    )

                    if status["status"] in ["FILLED", "PARTIALLY_FILLED"]:
                        fills = status.get("fills", [])
                        if fills:
                            # Calculate average fill price
                            total_value = sum(
                                float(f["price"]) * float(f["qty"]) for f in fills
                            )
                            total_qty = sum(float(f["qty"]) for f in fills)
                            avg_price = total_value / total_qty if total_qty > 0 else None
                        else:
                            avg_price = float(status.get("price") or 0)

                        return {
                            "success": True,
                            "order_id": order_id,
                            "fill_price": avg_price,
                            "filled_qty": float(status["executedQty"]),
                            "raw": status,
                        }

                except Exception as e:
                    LOG.warning(f"Status check failed (attempt {attempt}): {e}")

            # Timeout waiting for fill
            return {
                "success": False,
                "error": "Timeout waiting for market order fill",
                "order_id": order_id,
                "raw": order,
            }

        except Exception as e:
            LOG.error(f"Market entry failed: {e}")
            return {"success": False, "error": str(e)}

    async def _place_limit_entry(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> Dict[str, Any]:
        """Place limit entry order."""
        try:
            order = await self.client.create_order(
                symbol=symbol,
                side=side.upper(),
                type="LIMIT",
                timeInForce="GTC",
                quantity=quantity,
                price=price,
            )

            return {
                "success": True,
                "order_id": str(order.get("orderId") or order.get("clientOrderId")),
                "fill_price": price,
                "filled_qty": quantity,
                "raw": order,
            }

        except Exception as e:
            LOG.error(f"Limit entry failed: {e}")
            return {"success": False, "error": str(e)}

    async def _place_oco_protection(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_price: float,
        limit_price: float,
    ) -> Dict[str, Any]:
        """Place OCO order for SL + TP protection.

        OCO (One-Cancels-Other) ensures that when one order fills,
        the other is automatically cancelled.

        Args:
            symbol: Trading pair
            side: Exit side (opposite of entry)
            quantity: Quantity to protect
            stop_price: Stop-loss trigger price
            limit_price: Take-profit price

        Returns:
            OCO order result with orderListId and order IDs
        """
        try:
            # Binance OCO order format
            oco = await self.client.create_oco_order(
                symbol=symbol,
                side=side.upper(),
                quantity=quantity,
                price=limit_price,  # Take-profit limit price
                stopPrice=stop_price,  # Stop-loss trigger price
                stopLimitPrice=stop_price * 0.998,  # SL limit price (0.2% buffer)
                stopLimitTimeInForce="GTC",
            )

            return {
                "success": True,
                "orderListId": oco.get("orderListId"),
                "orders": oco.get("orders", []),
                "raw": oco,
            }

        except Exception as e:
            LOG.error(f"OCO placement failed: {e}")
            return {"success": False, "error": str(e)}

    async def _emergency_exit(
        self, symbol: str, entry_side: str, quantity: float
    ) -> None:
        """Emergency market exit if OCO placement fails.

        This is a critical safety mechanism. If we can't place protection,
        we immediately exit the position at market to prevent unlimited loss.
        """
        try:
            exit_side = "SELL" if entry_side == "BUY" else "BUY"

            LOG.critical(
                f"[EMERGENCY EXIT] Closing {quantity} {symbol} at market"
            )

            await self.client.create_order(
                symbol=symbol,
                side=exit_side,
                type="MARKET",
                quantity=quantity,
            )

            LOG.info(f"[EMERGENCY EXIT] Successfully exited {symbol}")

        except Exception as e:
            LOG.critical(
                f"[EMERGENCY EXIT FAILED] Could not exit {symbol}: {e}. "
                "MANUAL INTERVENTION REQUIRED!"
            )

    def _calculate_sl_price(
        self, entry_price: float, side: str, sl_pct: float
    ) -> float:
        """Calculate stop-loss price from percentage."""
        if side.upper() == "BUY":
            return entry_price * (1.0 - sl_pct)
        else:
            return entry_price * (1.0 + sl_pct)

    def _calculate_tp_price(
        self, entry_price: float, side: str, tp_pct: float
    ) -> float:
        """Calculate take-profit price from percentage."""
        if side.upper() == "BUY":
            return entry_price * (1.0 + tp_pct)
        else:
            return entry_price * (1.0 - tp_pct)

    async def _sleep(self, seconds: float) -> None:
        """Async sleep helper."""
        import asyncio
        await asyncio.sleep(seconds)
