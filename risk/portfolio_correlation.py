"""Portfolio correlation and risk aggregation for multi-position trading.

Addresses the critical issue: "5 crypto positions ≠ 5% risk"
When all assets are correlated 0.9, effective risk is much higher.

Key features:
- Rolling correlation matrix (90-day default)
- Effective portfolio risk calculation
- Sector exposure limits
- Correlation-adjusted position sizing
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.structured_logging import get_logger

LOG = get_logger("ai_trader.portfolio_correlation")


@dataclass
class PortfolioRisk:
    """Aggregated portfolio risk metrics."""

    total_positions: int
    individual_risk_sum: float  # Simple sum of position risks
    effective_risk: float  # Correlation-adjusted risk
    correlation_factor: float  # effective / individual
    max_correlated_exposure: float  # Max risk in single correlated group
    sector_exposures: Dict[str, float]  # Risk by sector
    warnings: List[str]  # Risk warnings

    def to_dict(self) -> Dict:
        return {
            "total_positions": self.total_positions,
            "individual_risk_sum": self.individual_risk_sum,
            "effective_risk": self.effective_risk,
            "correlation_factor": self.correlation_factor,
            "max_correlated_exposure": self.max_correlated_exposure,
            "sector_exposures": self.sector_exposures,
            "warnings": self.warnings,
        }


@dataclass
class Position:
    """Open position for portfolio analysis."""

    symbol: str
    side: str  # "long" or "short"
    quantity: float
    entry_price: float
    stop_loss: float
    risk_fraction: float  # % of portfolio at risk
    sector: Optional[str] = None  # e.g., "crypto", "tech", "commodity"


class PortfolioCorrelationTracker:
    """Track correlations and calculate true portfolio risk.

    Example:
        >>> tracker = PortfolioCorrelationTracker()
        >>> tracker.update_prices({
        ...     "BTCUSDT": [50000, 51000, 49000, ...],
        ...     "ETHUSDT": [3000, 3100, 2950, ...],
        ... })
        >>> risk = tracker.calculate_portfolio_risk([
        ...     Position("BTCUSDT", "long", 0.1, 50000, 49000, 0.01),
        ...     Position("ETHUSDT", "long", 1.0, 3000, 2950, 0.01),
        ... ])
        >>> print(risk.effective_risk)  # 0.019 (higher than 0.02 simple sum)
    """

    def __init__(
        self,
        window: int = 90,  # Days for correlation calculation
        min_periods: int = 30,  # Minimum periods required
        sector_limits: Optional[Dict[str, float]] = None,
    ):
        """Initialize correlation tracker.

        Args:
            window: Rolling window for correlation (days)
            min_periods: Minimum periods required for valid correlation
            sector_limits: Max risk per sector (e.g., {"crypto": 0.10})
        """
        self.window = window
        self.min_periods = min_periods
        self.sector_limits = sector_limits or {
            "crypto": 0.15,  # Max 15% in crypto
            "equity": 0.20,  # Max 20% in stocks
            "commodity": 0.10,  # Max 10% in commodities
            "forex": 0.10,  # Max 10% in currencies
        }

        # Price history storage
        self.price_history: Dict[str, List[float]] = {}
        self.timestamps: List[float] = []

        # Cached correlation matrix
        self._correlation_matrix: Optional[pd.DataFrame] = None
        self._last_update: float = 0

    def update_prices(
        self, prices: Dict[str, float], timestamp: Optional[float] = None
    ) -> None:
        """Update price history for correlation calculation.

        Args:
            prices: Dict of {symbol: price}
            timestamp: Unix timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = time.time()

        # Initialize storage for new symbols
        for symbol in prices:
            if symbol not in self.price_history:
                self.price_history[symbol] = []

        # Append new prices
        self.timestamps.append(timestamp)
        for symbol, price in prices.items():
            self.price_history[symbol].append(price)

        # Trim to window size
        max_length = self.window + 10  # Keep some buffer
        if len(self.timestamps) > max_length:
            trim = len(self.timestamps) - max_length
            self.timestamps = self.timestamps[trim:]
            for symbol in self.price_history:
                self.price_history[symbol] = self.price_history[symbol][trim:]

        # Invalidate cached correlation
        self._correlation_matrix = None

    def get_correlation_matrix(
        self, symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Calculate correlation matrix for given symbols.

        Args:
            symbols: List of symbols (default: all tracked symbols)

        Returns:
            Correlation matrix as DataFrame
        """
        if symbols is None:
            symbols = list(self.price_history.keys())

        # Use cached matrix if available and recent
        if (
            self._correlation_matrix is not None
            and time.time() - self._last_update < 3600  # 1 hour cache
            and all(s in self._correlation_matrix.columns for s in symbols)
        ):
            return self._correlation_matrix.loc[symbols, symbols]

        # Build price DataFrame
        price_data = {}
        for symbol in symbols:
            if symbol in self.price_history and len(self.price_history[symbol]) >= self.min_periods:
                price_data[symbol] = self.price_history[symbol]

        if not price_data:
            # No data available - return identity matrix
            LOG.warning("No price data for correlation - using identity matrix")
            return pd.DataFrame(
                np.eye(len(symbols)), index=symbols, columns=symbols
            )

        df = pd.DataFrame(price_data)

        # Calculate returns
        returns = df.pct_change().dropna()

        if len(returns) < self.min_periods:
            LOG.warning(
                f"Insufficient data ({len(returns)} < {self.min_periods}) - using identity matrix"
            )
            return pd.DataFrame(
                np.eye(len(symbols)), index=symbols, columns=symbols
            )

        # Calculate correlation matrix
        corr_matrix = returns.corr()

        # Fill NaN with 0 (uncorrelated)
        corr_matrix = corr_matrix.fillna(0)

        # Cache result
        self._correlation_matrix = corr_matrix
        self._last_update = time.time()

        return corr_matrix

    def calculate_portfolio_risk(
        self, positions: List[Position]
    ) -> PortfolioRisk:
        """Calculate effective portfolio risk considering correlations.

        Args:
            positions: List of open positions

        Returns:
            PortfolioRisk with correlation-adjusted metrics
        """
        if not positions:
            return PortfolioRisk(
                total_positions=0,
                individual_risk_sum=0.0,
                effective_risk=0.0,
                correlation_factor=1.0,
                max_correlated_exposure=0.0,
                sector_exposures={},
                warnings=[],
            )

        # Extract symbols and risk fractions
        symbols = [p.symbol for p in positions]
        risks = np.array([p.risk_fraction for p in positions])

        # Simple sum
        individual_risk_sum = float(risks.sum())

        # Get correlation matrix
        corr_matrix = self.get_correlation_matrix(symbols)

        # Calculate effective risk: sqrt(w^T * Σ * w)
        # where w = risk vector, Σ = correlation matrix
        try:
            effective_risk = np.sqrt(
                risks @ corr_matrix.values @ risks
            )
        except Exception as e:
            LOG.error(f"Effective risk calculation failed: {e}")
            effective_risk = individual_risk_sum  # Fallback

        correlation_factor = (
            effective_risk / individual_risk_sum
            if individual_risk_sum > 0
            else 1.0
        )

        # Sector exposure analysis
        sector_exposures = {}
        for pos in positions:
            sector = pos.sector or self._infer_sector(pos.symbol)
            sector_exposures[sector] = (
                sector_exposures.get(sector, 0.0) + pos.risk_fraction
            )

        # Find max correlated group
        max_correlated = self._find_max_correlated_group(
            positions, corr_matrix
        )

        # Generate warnings
        warnings = []

        # Warning 1: High correlation factor
        if correlation_factor > 1.3:
            warnings.append(
                f"High correlation factor: {correlation_factor:.2f}. "
                f"Effective risk ({effective_risk:.1%}) >> individual sum ({individual_risk_sum:.1%})"
            )

        # Warning 2: Sector limit violations
        for sector, exposure in sector_exposures.items():
            limit = self.sector_limits.get(sector, 0.20)
            if exposure > limit:
                warnings.append(
                    f"Sector limit exceeded: {sector} at {exposure:.1%} (limit {limit:.1%})"
                )

        # Warning 3: Highly correlated positions
        if max_correlated > individual_risk_sum * 0.7:
            warnings.append(
                f"High correlated group risk: {max_correlated:.1%} "
                f"(>70% of individual sum)"
            )

        return PortfolioRisk(
            total_positions=len(positions),
            individual_risk_sum=individual_risk_sum,
            effective_risk=float(effective_risk),
            correlation_factor=correlation_factor,
            max_correlated_exposure=max_correlated,
            sector_exposures=sector_exposures,
            warnings=warnings,
        )

    def calculate_correlation_adjusted_size(
        self,
        new_position: Position,
        existing_positions: List[Position],
        target_portfolio_risk: float = 0.06,  # 6% max portfolio risk
    ) -> Tuple[float, List[str]]:
        """Calculate position size adjusted for portfolio correlation.

        Args:
            new_position: Proposed new position
            existing_positions: Current open positions
            target_portfolio_risk: Max allowed portfolio risk

        Returns:
            (adjusted_risk_fraction, warnings)
        """
        # Calculate current portfolio risk
        current_risk = self.calculate_portfolio_risk(existing_positions)

        # Simulate adding new position
        simulated_positions = existing_positions + [new_position]
        simulated_risk = self.calculate_portfolio_risk(simulated_positions)

        # Check if new position would exceed limits
        warnings = []

        # Portfolio risk limit
        if simulated_risk.effective_risk > target_portfolio_risk:
            # Scale down new position
            excess = simulated_risk.effective_risk - target_portfolio_risk
            reduction_factor = 1.0 - (excess / new_position.risk_fraction)
            reduction_factor = max(0.1, min(1.0, reduction_factor))

            adjusted_risk = new_position.risk_fraction * reduction_factor

            warnings.append(
                f"Position size reduced by {(1 - reduction_factor) * 100:.1f}% "
                f"to stay within portfolio risk limit ({target_portfolio_risk:.1%})"
            )

            return adjusted_risk, warnings

        # Sector limit check
        new_sector = new_position.sector or self._infer_sector(
            new_position.symbol
        )
        sector_limit = self.sector_limits.get(new_sector, 0.20)
        new_sector_exposure = simulated_risk.sector_exposures.get(
            new_sector, 0.0
        )

        if new_sector_exposure > sector_limit:
            excess = new_sector_exposure - sector_limit
            reduction_factor = 1.0 - (excess / new_position.risk_fraction)
            reduction_factor = max(0.1, min(1.0, reduction_factor))

            adjusted_risk = new_position.risk_fraction * reduction_factor

            warnings.append(
                f"Position size reduced by {(1 - reduction_factor) * 100:.1f}% "
                f"due to {new_sector} sector limit ({sector_limit:.1%})"
            )

            return adjusted_risk, warnings

        # No adjustment needed
        return new_position.risk_fraction, []

    def _find_max_correlated_group(
        self, positions: List[Position], corr_matrix: pd.DataFrame
    ) -> float:
        """Find maximum risk in highly correlated position group.

        Args:
            positions: Open positions
            corr_matrix: Correlation matrix

        Returns:
            Max correlated group risk
        """
        # Find pairs with correlation > 0.7
        high_corr_threshold = 0.7

        groups = []
        used_symbols = set()

        for i, pos1 in enumerate(positions):
            if pos1.symbol in used_symbols:
                continue

            group = [pos1]
            for j, pos2 in enumerate(positions[i + 1 :]):
                if pos2.symbol in used_symbols:
                    continue

                try:
                    corr = corr_matrix.loc[pos1.symbol, pos2.symbol]
                    if corr > high_corr_threshold:
                        group.append(pos2)
                        used_symbols.add(pos2.symbol)
                except KeyError:
                    pass

            if len(group) > 1:
                groups.append(group)
                used_symbols.add(pos1.symbol)

        # Calculate risk for each group
        if not groups:
            return 0.0

        max_group_risk = max(
            sum(p.risk_fraction for p in group) for group in groups
        )

        return max_group_risk

    def _infer_sector(self, symbol: str) -> str:
        """Infer sector from symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Sector name
        """
        symbol_upper = symbol.upper()

        # Crypto patterns
        crypto_suffixes = ["USDT", "BUSD", "USD", "BTC", "ETH"]
        if any(symbol_upper.endswith(s) for s in crypto_suffixes):
            return "crypto"

        # Forex patterns
        forex_pairs = [
            "EUR", "USD", "GBP", "JPY", "CHF", "AUD", "CAD", "NZD"
        ]
        if any(symbol_upper.startswith(c) and symbol_upper[3:].startswith(c2)
               for c in forex_pairs for c2 in forex_pairs):
            return "forex"

        # Commodity patterns
        commodities = ["XAU", "XAG", "OIL", "GAS", "GOLD", "SILVER"]
        if any(c in symbol_upper for c in commodities):
            return "commodity"

        # Default to equity
        return "equity"
