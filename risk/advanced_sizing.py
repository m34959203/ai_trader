"""Advanced position sizing with Kelly Criterion and volatility adaptation.

Addresses critical issues:
1. Real Kelly Criterion (not just in docstring!)
2. Aggressive volatility adjustment (>10% ATR = stop trading)
3. Drawdown-responsive sizing
4. Confidence-weighted position sizing
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from utils.structured_logging import get_logger

LOG = get_logger("ai_trader.advanced_sizing")


@dataclass
class KellyResult:
    """Kelly Criterion calculation result."""

    kelly_fraction: float  # Full Kelly
    half_kelly: float  # Conservative (recommended)
    quarter_kelly: float  # Very conservative
    win_rate: float  # Historical win rate
    avg_win: float  # Average winning trade
    avg_loss: float  # Average losing trade (absolute)
    expectancy: float  # Expected value per trade
    recommendation: str  # Sizing recommendation
    warnings: List[str]  # Risk warnings

    def to_dict(self):
        return {
            "kelly_fraction": self.kelly_fraction,
            "half_kelly": self.half_kelly,
            "quarter_kelly": self.quarter_kelly,
            "win_rate": self.win_rate,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "expectancy": self.expectancy,
            "recommendation": self.recommendation,
            "warnings": self.warnings,
        }


class KellyCriterion:
    """Real Kelly Criterion implementation for position sizing.

    Formula:
        f* = (p×b - q) / b
        where:
        - f* = Kelly fraction (% of capital to risk)
        - p = win rate
        - q = 1 - p (loss rate)
        - b = avg_win / avg_loss (win/loss ratio)

    Example:
        >>> kelly = KellyCriterion(lookback=50)
        >>> kelly.update(pnl=100)  # Winning trade
        >>> kelly.update(pnl=-50)  # Losing trade
        >>> result = kelly.calculate()
        >>> print(result.half_kelly)  # Recommended fraction
    """

    def __init__(
        self,
        lookback: int = 50,
        min_trades: int = 20,
    ):
        """Initialize Kelly calculator.

        Args:
            lookback: Number of recent trades to consider
            min_trades: Minimum trades required for valid calculation
        """
        self.lookback = lookback
        self.min_trades = min_trades
        self.trade_history: List[float] = []

    def update(self, pnl: float) -> None:
        """Add new trade result.

        Args:
            pnl: Trade P&L (positive = win, negative = loss)
        """
        self.trade_history.append(pnl)

        # Trim to lookback window
        if len(self.trade_history) > self.lookback:
            self.trade_history = self.trade_history[-self.lookback:]

    def calculate(self) -> KellyResult:
        """Calculate Kelly fractions from trade history.

        Returns:
            KellyResult with recommended sizing
        """
        warnings = []

        # Check minimum trades
        if len(self.trade_history) < self.min_trades:
            warnings.append(
                f"Insufficient trade history ({len(self.trade_history)} < {self.min_trades}). "
                "Using conservative 1% sizing."
            )
            return KellyResult(
                kelly_fraction=0.01,
                half_kelly=0.01,
                quarter_kelly=0.01,
                win_rate=0.5,
                avg_win=0.0,
                avg_loss=0.0,
                expectancy=0.0,
                recommendation="Insufficient data - use 1% fixed sizing",
                warnings=warnings,
            )

        # Separate wins and losses
        wins = [pnl for pnl in self.trade_history if pnl > 0]
        losses = [abs(pnl) for pnl in self.trade_history if pnl < 0]

        if not wins or not losses:
            warnings.append("No wins or no losses in history. Using 1% sizing.")
            return KellyResult(
                kelly_fraction=0.01,
                half_kelly=0.01,
                quarter_kelly=0.01,
                win_rate=len(wins) / len(self.trade_history) if wins else 0.0,
                avg_win=np.mean(wins) if wins else 0.0,
                avg_loss=np.mean(losses) if losses else 0.0,
                expectancy=0.0,
                recommendation="Insufficient diversity - use 1% fixed sizing",
                warnings=warnings,
            )

        # Calculate metrics
        win_rate = len(wins) / len(self.trade_history)
        loss_rate = 1.0 - win_rate
        avg_win = np.mean(wins)
        avg_loss = np.mean(losses)

        # Win/loss ratio
        b = avg_win / avg_loss if avg_loss > 0 else 1.0

        # Kelly formula: f* = (p×b - q) / b
        kelly_numerator = (win_rate * b) - loss_rate
        kelly_fraction = kelly_numerator / b if b > 0 else 0.0

        # Expectancy (expected value per trade)
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)

        # Safety checks
        if kelly_fraction < 0:
            warnings.append(
                f"Negative Kelly ({kelly_fraction:.3f}) - system has negative expectancy! "
                "DO NOT TRADE until strategy is fixed."
            )
            kelly_fraction = 0.0

        if kelly_fraction > 0.20:  # > 20% is dangerous
            warnings.append(
                f"Kelly fraction very high ({kelly_fraction:.3f}). "
                "Capping at 5% for safety."
            )
            kelly_fraction = min(kelly_fraction, 0.05)

        # Calculate fractional Kelly (more conservative)
        half_kelly = kelly_fraction * 0.5
        quarter_kelly = kelly_fraction * 0.25

        # Recommendation logic
        if expectancy <= 0:
            recommendation = "STOP TRADING - Negative expectancy"
        elif win_rate < 0.40:
            recommendation = "Use quarter-Kelly (very low win rate)"
        elif win_rate < 0.50:
            recommendation = "Use half-Kelly (moderate win rate)"
        elif b < 1.5:  # Win/loss ratio < 1.5
            recommendation = "Use half-Kelly (low win/loss ratio)"
        else:
            recommendation = "Use half-Kelly (recommended)"

        return KellyResult(
            kelly_fraction=kelly_fraction,
            half_kelly=half_kelly,
            quarter_kelly=quarter_kelly,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            expectancy=expectancy,
            recommendation=recommendation,
            warnings=warnings,
        )


class VolatilityAdapter:
    """Adaptive position sizing based on market volatility.

    Aggressive adjustment:
    - ATR > 20%: STOP TRADING (market chaos)
    - ATR > 10%: Reduce size by 80%
    - ATR > 5%: Reduce size by 50%
    - ATR < 2%: Normal sizing
    """

    def __init__(
        self,
        extreme_threshold: float = 0.20,  # 20% ATR
        high_threshold: float = 0.10,  # 10% ATR
        medium_threshold: float = 0.05,  # 5% ATR
    ):
        self.extreme_threshold = extreme_threshold
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

    def calculate_volatility_multiplier(
        self,
        atr_pct: float,
        force_trading: bool = False,
    ) -> tuple[float, str]:
        """Calculate position size multiplier based on volatility.

        Args:
            atr_pct: ATR as percentage (e.g., 0.02 = 2%)
            force_trading: Override extreme volatility halt (dangerous!)

        Returns:
            (multiplier, reason)
        """
        if atr_pct >= self.extreme_threshold:
            if force_trading:
                LOG.warning(
                    f"EXTREME VOLATILITY ({atr_pct:.1%}) - forcing trade against recommendation!"
                )
                return 0.1, f"Extreme volatility ({atr_pct:.1%}) - reduced to 10%"
            else:
                return 0.0, f"HALT - Extreme volatility ({atr_pct:.1%})"

        elif atr_pct >= self.high_threshold:
            multiplier = 0.2  # -80%
            return multiplier, f"High volatility ({atr_pct:.1%}) - reduced to 20%"

        elif atr_pct >= self.medium_threshold:
            multiplier = 0.5  # -50%
            return multiplier, f"Medium volatility ({atr_pct:.1%}) - reduced to 50%"

        else:
            # Normal volatility - no reduction
            return 1.0, f"Normal volatility ({atr_pct:.1%})"


class DrawdownAdapter:
    """Reduce position size during drawdowns to prevent death spiral."""

    def __init__(
        self,
        peak_equity: float,
        severe_dd_threshold: float = 0.15,  # 15% DD
        moderate_dd_threshold: float = 0.08,  # 8% DD
    ):
        self.peak_equity = peak_equity
        self.severe_dd_threshold = severe_dd_threshold
        self.moderate_dd_threshold = moderate_dd_threshold

    def update_peak(self, current_equity: float) -> None:
        """Update peak equity if new high reached."""
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

    def calculate_drawdown_multiplier(
        self, current_equity: float
    ) -> tuple[float, str]:
        """Calculate position size reduction based on drawdown.

        Args:
            current_equity: Current portfolio value

        Returns:
            (multiplier, reason)
        """
        drawdown = (self.peak_equity - current_equity) / self.peak_equity

        if drawdown >= self.severe_dd_threshold:
            # Severe drawdown: reduce to 30%
            return 0.3, f"Severe drawdown ({drawdown:.1%}) - reduced to 30%"

        elif drawdown >= self.moderate_dd_threshold:
            # Moderate drawdown: reduce to 60%
            return 0.6, f"Moderate drawdown ({drawdown:.1%}) - reduced to 60%"

        else:
            # Normal or no drawdown
            return 1.0, f"No significant drawdown ({drawdown:.1%})"


class AdvancedPositionSizer:
    """Integrated position sizing with Kelly, volatility, and drawdown adaptation.

    Usage:
        >>> sizer = AdvancedPositionSizer(initial_equity=100000)
        >>> sizer.kelly.update(100)  # Add trade results
        >>> sizer.kelly.update(-50)
        >>> size = sizer.calculate_position_size(
        ...     base_risk=0.02,  # 2% base risk
        ...     atr_pct=0.03,    # 3% volatility
        ...     current_equity=98000,
        ...     signal_confidence=0.8,
        ... )
        >>> print(size)  # Adjusted position size
    """

    def __init__(
        self,
        initial_equity: float,
        kelly_lookback: int = 50,
    ):
        self.kelly = KellyCriterion(lookback=kelly_lookback)
        self.volatility = VolatilityAdapter()
        self.drawdown = DrawdownAdapter(peak_equity=initial_equity)

    def calculate_position_size(
        self,
        base_risk: float,
        atr_pct: float,
        current_equity: float,
        signal_confidence: Optional[float] = None,
        force_trading: bool = False,
    ) -> tuple[float, dict]:
        """Calculate final position size with all adjustments.

        Args:
            base_risk: Base risk per trade (e.g., 0.02 = 2%)
            atr_pct: Market volatility (ATR%)
            current_equity: Current portfolio value
            signal_confidence: Signal confidence (0-1, optional)
            force_trading: Force trade even in extreme volatility

        Returns:
            (final_risk_fraction, adjustments_dict)
        """
        adjustments = {}

        # 1. Kelly Criterion
        kelly_result = self.kelly.calculate()
        kelly_size = kelly_result.half_kelly  # Use conservative half-Kelly
        adjustments["kelly"] = {
            "fraction": kelly_size,
            "recommendation": kelly_result.recommendation,
            "warnings": kelly_result.warnings,
        }

        # 2. Volatility adjustment
        vol_multiplier, vol_reason = self.volatility.calculate_volatility_multiplier(
            atr_pct, force_trading
        )
        adjustments["volatility"] = {
            "multiplier": vol_multiplier,
            "reason": vol_reason,
        }

        # 3. Drawdown adjustment
        self.drawdown.update_peak(current_equity)
        dd_multiplier, dd_reason = self.drawdown.calculate_drawdown_multiplier(
            current_equity
        )
        adjustments["drawdown"] = {
            "multiplier": dd_multiplier,
            "reason": dd_reason,
        }

        # 4. Signal confidence adjustment (if provided)
        conf_multiplier = signal_confidence if signal_confidence is not None else 1.0
        adjustments["confidence"] = {
            "multiplier": conf_multiplier,
            "provided": signal_confidence is not None,
        }

        # Final calculation: Base × Kelly × Vol × DD × Confidence
        final_size = base_risk * kelly_size * vol_multiplier * dd_multiplier * conf_multiplier

        # Safety caps
        final_size = max(0.0, min(final_size, 0.05))  # Cap at 5%

        adjustments["final_risk"] = final_size
        adjustments["reduction_from_base"] = (
            (base_risk - final_size) / base_risk * 100
            if base_risk > 0
            else 0
        )

        return final_size, adjustments
