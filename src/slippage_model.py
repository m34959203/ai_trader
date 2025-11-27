"""Realistic slippage and transaction cost modeling for backtesting.

This module provides professional-grade slippage estimation based on:
- Market volatility (ATR)
- Order size relative to average volume
- Bid-ask spread estimation
- Market impact from large orders

References:
- "Algorithmic Trading" by Ernest P. Chan (Chapter 6)
- "Advances in Financial Machine Learning" by Marcos López de Prado
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


@dataclass
class SlippageConfig:
    """Configuration for slippage model."""

    # Base slippage (minimum)
    base_slippage_bps: float = 5.0  # 0.05% = 5 basis points

    # Volatility impact
    volatility_multiplier: float = 2.0  # ATR% × multiplier

    # Volume impact
    volume_impact_multiplier: float = 0.5  # (qty/volume) × multiplier

    # Spread estimation
    min_spread_bps: float = 2.0  # Minimum bid-ask spread
    max_spread_bps: float = 50.0  # Maximum bid-ask spread (illiquid)

    # Market impact (for large orders)
    market_impact_exponent: float = 0.6  # Non-linear impact

    # Emergency slippage (flash crashes, gaps)
    gap_slippage_bps: float = 100.0  # 1% for gap events

    def to_dict(self):
        return {
            "base_slippage_bps": self.base_slippage_bps,
            "volatility_multiplier": self.volatility_multiplier,
            "volume_impact_multiplier": self.volume_impact_multiplier,
            "min_spread_bps": self.min_spread_bps,
            "max_spread_bps": self.max_spread_bps,
            "market_impact_exponent": self.market_impact_exponent,
            "gap_slippage_bps": self.gap_slippage_bps,
        }


class SlippageModel:
    """Calculate realistic slippage for backtesting.

    Slippage components:
    1. Base slippage: Fixed minimum cost (crossing spread)
    2. Volatility impact: ATR-based additional cost
    3. Volume impact: Order size relative to liquidity
    4. Market impact: Non-linear for large orders
    5. Special events: Gaps, halts, flash crashes

    Example:
        >>> model = SlippageModel()
        >>> slippage = model.calculate_slippage(
        ...     price=50000,
        ...     quantity=0.1,
        ...     avg_volume=100,
        ...     volatility=0.02,  # 2% ATR
        ...     side="buy"
        ... )
        >>> print(slippage)  # ~0.003 (0.3%)
    """

    def __init__(self, config: Optional[SlippageConfig] = None):
        self.config = config or SlippageConfig()

    def calculate_slippage(
        self,
        price: float,
        quantity: float,
        avg_volume: float,
        volatility: float,
        side: Literal["buy", "sell"] = "buy",
        is_market_order: bool = True,
        is_gap_event: bool = False,
    ) -> float:
        """Calculate total slippage as percentage.

        Args:
            price: Entry price
            quantity: Order quantity
            avg_volume: Average trading volume (same units as quantity)
            volatility: Market volatility (ATR as percentage, e.g., 0.02 = 2%)
            side: "buy" or "sell"
            is_market_order: True for market orders, False for limit
            is_gap_event: True if gap/halt/flash crash detected

        Returns:
            Slippage as percentage (e.g., 0.005 = 0.5%)
        """
        # Special case: gap events have huge slippage
        if is_gap_event:
            gap_slippage = self.config.gap_slippage_bps / 10000
            LOG.warning(
                f"Gap event detected - applying {self.config.gap_slippage_bps}bps slippage"
            )
            return gap_slippage

        # Component 1: Base slippage (spread crossing)
        base = self.config.base_slippage_bps / 10000

        # Component 2: Volatility impact
        vol_impact = volatility * self.config.volatility_multiplier

        # Component 3: Volume impact (liquidity)
        volume_ratio = quantity / avg_volume if avg_volume > 0 else 1.0
        volume_impact = (
            volume_ratio ** self.config.market_impact_exponent
        ) * self.config.volume_impact_multiplier

        # Component 4: Spread estimation
        spread = self._estimate_spread(volatility)

        # Total slippage
        total_slippage = base + vol_impact + volume_impact + spread

        # Market orders pay the spread, limit orders don't (but have execution risk)
        if not is_market_order:
            total_slippage *= 0.5  # Limit orders pay ~50% slippage

        # Slippage direction (buy = positive, sell = negative)
        direction = 1.0 if side.lower() == "buy" else -1.0

        return total_slippage * direction

    def calculate_fill_price(
        self,
        entry_price: float,
        quantity: float,
        avg_volume: float,
        volatility: float,
        side: Literal["buy", "sell"] = "buy",
        is_market_order: bool = True,
        is_gap_event: bool = False,
    ) -> float:
        """Calculate actual fill price including slippage.

        Args:
            entry_price: Ideal entry price (e.g., bar close)
            ... (same as calculate_slippage)

        Returns:
            Actual fill price after slippage
        """
        slippage_pct = self.calculate_slippage(
            price=entry_price,
            quantity=quantity,
            avg_volume=avg_volume,
            volatility=volatility,
            side=side,
            is_market_order=is_market_order,
            is_gap_event=is_gap_event,
        )

        fill_price = entry_price * (1.0 + slippage_pct)

        return fill_price

    def _estimate_spread(self, volatility: float) -> float:
        """Estimate bid-ask spread from volatility.

        Empirical formula: spread ≈ k × √(volatility)
        where k is calibrated to market (typically 0.5-2.0 for crypto).

        Args:
            volatility: Market volatility (ATR%)

        Returns:
            Estimated spread as percentage
        """
        # Simple model: spread increases with sqrt(volatility)
        spread_bps = (
            self.config.min_spread_bps + np.sqrt(volatility * 1000) * 2.0
        )

        # Cap at maximum
        spread_bps = min(spread_bps, self.config.max_spread_bps)

        return spread_bps / 10000

    def add_slippage_to_backtest(
        self,
        df: pd.DataFrame,
        quantity_col: str = "quantity",
        volume_col: str = "volume",
        atr_col: str = "atr",
        price_col: str = "close",
        side_col: str = "signal",  # 1 = buy, -1 = sell
    ) -> pd.DataFrame:
        """Add slippage columns to backtest DataFrame.

        Args:
            df: DataFrame with OHLCV and signals
            quantity_col: Column with order quantities
            volume_col: Column with trading volume
            atr_col: Column with ATR values
            price_col: Price column for fill price calculation
            side_col: Column with signals (1=buy, -1=sell, 0=flat)

        Returns:
            DataFrame with added columns:
            - slippage_pct: Slippage percentage
            - fill_price: Actual fill price
            - slippage_cost: Dollar cost of slippage
        """
        df = df.copy()

        # Calculate volatility as ATR percentage
        df["volatility"] = df[atr_col] / df[price_col]

        # Estimate average volume (rolling 20-period average)
        df["avg_volume"] = df[volume_col].rolling(20, min_periods=1).mean()

        # Detect gap events (open significantly different from previous close)
        df["gap_pct"] = abs(
            (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        )
        df["is_gap"] = df["gap_pct"] > 0.02  # 2% gap threshold

        # Calculate slippage for each trade
        slippages = []
        for idx, row in df.iterrows():
            if row.get(side_col, 0) == 0:
                # No trade
                slippages.append(0.0)
                continue

            side = "buy" if row[side_col] > 0 else "sell"

            slippage = self.calculate_slippage(
                price=row[price_col],
                quantity=row.get(quantity_col, 1.0),
                avg_volume=row["avg_volume"],
                volatility=row["volatility"],
                side=side,
                is_market_order=True,
                is_gap_event=row.get("is_gap", False),
            )

            slippages.append(slippage)

        df["slippage_pct"] = slippages

        # Calculate fill prices
        df["fill_price"] = df[price_col] * (1.0 + df["slippage_pct"])

        # Calculate dollar cost of slippage
        df["slippage_cost"] = (
            abs(df["slippage_pct"]) * df[price_col] * df.get(quantity_col, 1.0)
        )

        return df


# Logging setup
from utils.structured_logging import get_logger

LOG = get_logger("ai_trader.slippage")


# Factory function
def create_conservative_model() -> SlippageModel:
    """Create conservative slippage model (higher costs).

    Use for worst-case backtesting.
    """
    config = SlippageConfig(
        base_slippage_bps=8.0,
        volatility_multiplier=3.0,
        volume_impact_multiplier=0.8,
        gap_slippage_bps=150.0,
    )
    return SlippageModel(config)


def create_optimistic_model() -> SlippageModel:
    """Create optimistic slippage model (lower costs).

    Use for best-case backtesting.
    """
    config = SlippageConfig(
        base_slippage_bps=3.0,
        volatility_multiplier=1.0,
        volume_impact_multiplier=0.3,
        gap_slippage_bps=50.0,
    )
    return SlippageModel(config)


def create_realistic_model() -> SlippageModel:
    """Create realistic slippage model (balanced).

    Recommended for production backtesting.
    """
    return SlippageModel()  # Uses default config
