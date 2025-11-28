"""Weekend and overnight gap protection for trading.

Addresses the critical risk: positions held over weekends or overnight
can gap significantly due to news/events when markets are closed.

Example disaster scenario:
- Friday 23:59: Long BTC at $50K, SL at $49K
- Weekend: Regulatory ban announced
- Monday 00:01: BTC opens at $45K (-10% gap)
- SL triggers at market → fills at $44.5K
- Loss: $5,500 instead of planned $1,000 (550% of planned risk!)

This module prevents such disasters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Literal, Tuple

from utils.structured_logging import get_logger

LOG = get_logger("ai_trader.gap_protection")


@dataclass
class GapProtectionConfig:
    """Configuration for gap protection."""

    # Weekend settings
    enable_weekend_reduction: bool = True
    weekend_size_multiplier: float = 0.5  # -50% on Fri evening / Sat / Sun
    weekend_start_day: int = 4  # Friday (0=Monday, 4=Friday)
    weekend_start_hour: int = 20  # 20:00 UTC (8 PM)

    # Overnight settings
    enable_overnight_reduction: bool = True
    overnight_size_cap: float = 0.015  # Max 1.5% risk overnight
    overnight_start_hour: int = 22  # 22:00 UTC (10 PM)
    overnight_end_hour: int = 6  # 06:00 UTC (6 AM)

    # Stop-loss widening
    enable_wide_stops: bool = True
    weekend_stop_multiplier: float = 2.0  # 2× wider stops for weekend
    overnight_stop_multiplier: float = 1.5  # 1.5× wider stops overnight

    # Emergency settings
    enable_friday_close_all: bool = False  # Close all positions Friday night
    friday_close_hour: int = 22  # Close at 22:00 UTC if enabled

    def to_dict(self):
        return {
            "enable_weekend_reduction": self.enable_weekend_reduction,
            "weekend_size_multiplier": self.weekend_size_multiplier,
            "weekend_start_day": self.weekend_start_day,
            "weekend_start_hour": self.weekend_start_hour,
            "enable_overnight_reduction": self.enable_overnight_reduction,
            "overnight_size_cap": self.overnight_size_cap,
            "overnight_start_hour": self.overnight_start_hour,
            "overnight_end_hour": self.overnight_end_hour,
            "enable_wide_stops": self.enable_wide_stops,
            "weekend_stop_multiplier": self.weekend_stop_multiplier,
            "overnight_stop_multiplier": self.overnight_stop_multiplier,
            "enable_friday_close_all": self.enable_friday_close_all,
            "friday_close_hour": self.friday_close_hour,
        }


@dataclass
class GapProtectionAdjustment:
    """Gap protection adjustments for a trade."""

    should_trade: bool  # False if should close all positions
    size_multiplier: float  # Multiply position size by this
    stop_multiplier: float  # Multiply stop distance by this
    reason: str  # Explanation of adjustment
    risk_period: Literal["normal", "overnight", "weekend"]

    def to_dict(self):
        return {
            "should_trade": self.should_trade,
            "size_multiplier": self.size_multiplier,
            "stop_multiplier": self.stop_multiplier,
            "reason": self.reason,
            "risk_period": self.risk_period,
        }


class GapProtector:
    """Protect positions from weekend and overnight gaps.

    Usage:
        >>> protector = GapProtector()
        >>> adjustment = protector.get_adjustment(datetime.now())
        >>> if adjustment.should_trade:
        ...     position_size *= adjustment.size_multiplier
        ...     stop_distance *= adjustment.stop_multiplier
        ... else:
        ...     # Close all positions (Friday night)
        ...     close_all_positions()
    """

    def __init__(self, config: GapProtectionConfig | None = None):
        """Initialize gap protector.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or GapProtectionConfig()

    def get_adjustment(
        self,
        current_time: datetime | None = None,
    ) -> GapProtectionAdjustment:
        """Calculate position size and stop adjustments for current time.

        Args:
            current_time: Time to check (defaults to now UTC)

        Returns:
            GapProtectionAdjustment with size/stop multipliers
        """
        if current_time is None:
            current_time = datetime.utcnow()

        # Check if Friday close-all is enabled and triggered
        if self._should_close_all(current_time):
            return GapProtectionAdjustment(
                should_trade=False,
                size_multiplier=0.0,
                stop_multiplier=1.0,
                reason=f"Friday close-all enabled: closing at {self.config.friday_close_hour}:00 UTC",
                risk_period="weekend",
            )

        # Check weekend period
        is_weekend, weekend_reason = self._is_weekend_period(current_time)
        if is_weekend and self.config.enable_weekend_reduction:
            return GapProtectionAdjustment(
                should_trade=True,
                size_multiplier=self.config.weekend_size_multiplier,
                stop_multiplier=self.config.weekend_stop_multiplier if self.config.enable_wide_stops else 1.0,
                reason=weekend_reason,
                risk_period="weekend",
            )

        # Check overnight period
        is_overnight, overnight_reason = self._is_overnight_period(current_time)
        if is_overnight and self.config.enable_overnight_reduction:
            return GapProtectionAdjustment(
                should_trade=True,
                size_multiplier=1.0,  # No size reduction, but cap applies
                stop_multiplier=self.config.overnight_stop_multiplier if self.config.enable_wide_stops else 1.0,
                reason=overnight_reason,
                risk_period="overnight",
            )

        # Normal trading hours
        return GapProtectionAdjustment(
            should_trade=True,
            size_multiplier=1.0,
            stop_multiplier=1.0,
            reason="Normal trading hours - no adjustments",
            risk_period="normal",
        )

    def adjust_position_size(
        self,
        base_size: float,
        current_time: datetime | None = None,
    ) -> Tuple[float, str]:
        """Adjust position size for gap risk.

        Args:
            base_size: Base position size (risk fraction, e.g., 0.02 = 2%)
            current_time: Time to check (defaults to now UTC)

        Returns:
            (adjusted_size, reason)
        """
        adjustment = self.get_adjustment(current_time)

        if not adjustment.should_trade:
            return 0.0, adjustment.reason

        adjusted_size = base_size * adjustment.size_multiplier

        # Apply overnight cap if in overnight period
        if adjustment.risk_period == "overnight":
            if adjusted_size > self.config.overnight_size_cap:
                adjusted_size = self.config.overnight_size_cap
                return adjusted_size, f"{adjustment.reason} (capped at {self.config.overnight_size_cap:.1%})"

        return adjusted_size, adjustment.reason

    def adjust_stop_distance(
        self,
        base_stop_distance: float,
        current_time: datetime | None = None,
    ) -> Tuple[float, str]:
        """Adjust stop-loss distance for gap risk.

        Wider stops account for potential gaps when market is closed.

        Args:
            base_stop_distance: Base stop distance (e.g., 0.02 = 2%)
            current_time: Time to check (defaults to now UTC)

        Returns:
            (adjusted_stop_distance, reason)
        """
        adjustment = self.get_adjustment(current_time)

        adjusted_stop = base_stop_distance * adjustment.stop_multiplier

        return adjusted_stop, adjustment.reason

    def _should_close_all(self, current_time: datetime) -> bool:
        """Check if should close all positions (Friday close-all mode).

        Args:
            current_time: Time to check

        Returns:
            True if should close all positions
        """
        if not self.config.enable_friday_close_all:
            return False

        is_friday = current_time.weekday() == 4  # Friday
        is_close_hour = current_time.hour >= self.config.friday_close_hour

        return is_friday and is_close_hour

    def _is_weekend_period(self, current_time: datetime) -> Tuple[bool, str]:
        """Check if current time is in weekend risk period.

        Args:
            current_time: Time to check

        Returns:
            (is_weekend, reason)
        """
        day = current_time.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun
        hour = current_time.hour

        # Saturday or Sunday (always weekend)
        if day in [5, 6]:
            return True, f"Weekend ({current_time.strftime('%A')})"

        # Friday after weekend_start_hour
        if day == self.config.weekend_start_day and hour >= self.config.weekend_start_hour:
            return True, f"Friday evening (after {self.config.weekend_start_hour}:00 UTC)"

        return False, ""

    def _is_overnight_period(self, current_time: datetime) -> Tuple[bool, str]:
        """Check if current time is in overnight risk period.

        Args:
            current_time: Time to check

        Returns:
            (is_overnight, reason)
        """
        hour = current_time.hour

        # Simple case: overnight_start > overnight_end (e.g., 22:00 - 06:00)
        if self.config.overnight_start_hour > self.config.overnight_end_hour:
            is_overnight = hour >= self.config.overnight_start_hour or hour < self.config.overnight_end_hour
        else:
            # Unusual case: overnight_start < overnight_end (e.g., 02:00 - 06:00)
            is_overnight = self.config.overnight_start_hour <= hour < self.config.overnight_end_hour

        if is_overnight:
            return True, f"Overnight hours ({self.config.overnight_start_hour}:00 - {self.config.overnight_end_hour}:00 UTC)"

        return False, ""


# Factory functions for common configurations
def create_conservative_protector() -> GapProtector:
    """Create gap protector with conservative settings.

    - 70% size reduction on weekends
    - 1% max overnight risk
    - 3× wider stops on weekends
    - Close all positions Friday 20:00 UTC
    """
    config = GapProtectionConfig(
        enable_weekend_reduction=True,
        weekend_size_multiplier=0.3,  # -70%
        weekend_start_day=4,
        weekend_start_hour=20,
        enable_overnight_reduction=True,
        overnight_size_cap=0.01,  # 1% max
        overnight_start_hour=22,
        overnight_end_hour=6,
        enable_wide_stops=True,
        weekend_stop_multiplier=3.0,  # 3× wider
        overnight_stop_multiplier=2.0,  # 2× wider
        enable_friday_close_all=True,  # VERY CONSERVATIVE
        friday_close_hour=20,
    )
    return GapProtector(config)


def create_aggressive_protector() -> GapProtector:
    """Create gap protector with aggressive settings.

    - 30% size reduction on weekends
    - 2.5% max overnight risk
    - No stop widening
    - No Friday close-all
    """
    config = GapProtectionConfig(
        enable_weekend_reduction=True,
        weekend_size_multiplier=0.7,  # -30%
        weekend_start_day=4,
        weekend_start_hour=22,
        enable_overnight_reduction=True,
        overnight_size_cap=0.025,  # 2.5% max
        overnight_start_hour=23,
        overnight_end_hour=5,
        enable_wide_stops=False,  # No stop widening
        weekend_stop_multiplier=1.0,
        overnight_stop_multiplier=1.0,
        enable_friday_close_all=False,  # Keep positions open
        friday_close_hour=22,
    )
    return GapProtector(config)


def create_balanced_protector() -> GapProtector:
    """Create gap protector with balanced settings (recommended).

    - 50% size reduction on weekends
    - 1.5% max overnight risk
    - 2× wider stops on weekends, 1.5× overnight
    - No Friday close-all (but reduced size)
    """
    return GapProtector()  # Uses default config which is balanced
