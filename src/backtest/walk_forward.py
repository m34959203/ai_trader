"""Walk-forward testing framework for strategy validation.

Walk-forward testing is the gold standard for detecting overfitting in trading strategies.
Unlike simple backtesting which uses all historical data at once, walk-forward:

1. Trains on historical window (e.g., 12 months)
2. Tests on future window (e.g., 3 months)
3. Rolls window forward
4. Repeats until all data is covered

This reveals if strategy parameters that worked in past still work in future.

Example:
    Train on 2020-2021 → Test on Q1 2022
    Train on 2020-2021 + Q1 2022 → Test on Q2 2022
    Train on 2020-2021 + Q1-Q2 2022 → Test on Q3 2022
    ...

If test performance significantly worse than train → overfitting detected!
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.structured_logging import get_logger

LOG = get_logger("ai_trader.walk_forward")


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward testing."""

    train_window_days: int = 365  # 1 year training window
    test_window_days: int = 90  # 3 months testing window
    step_days: int = 30  # Move forward 1 month each iteration
    min_train_samples: int = 100  # Minimum bars for training
    min_test_samples: int = 20  # Minimum bars for testing

    # Optimization settings
    optimize_on_train: bool = True  # Optimize parameters on training data
    use_anchored_walk: bool = False  # True = expanding window, False = rolling window

    def to_dict(self):
        return {
            "train_window_days": self.train_window_days,
            "test_window_days": self.test_window_days,
            "step_days": self.step_days,
            "min_train_samples": self.min_train_samples,
            "min_test_samples": self.min_test_samples,
            "optimize_on_train": self.optimize_on_train,
            "use_anchored_walk": self.use_anchored_walk,
        }


@dataclass
class WalkForwardResult:
    """Result of a single walk-forward iteration."""

    iteration: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime

    # Optimized parameters (if optimization enabled)
    optimized_params: Dict[str, Any] = field(default_factory=dict)

    # Training metrics
    train_sharpe: float = 0.0
    train_returns: float = 0.0
    train_max_dd: float = 0.0
    train_win_rate: float = 0.0

    # Testing metrics (out-of-sample)
    test_sharpe: float = 0.0
    test_returns: float = 0.0
    test_max_dd: float = 0.0
    test_win_rate: float = 0.0

    # Degradation metrics
    sharpe_degradation: float = 0.0  # (test - train) / train
    returns_degradation: float = 0.0

    # Trade counts
    train_trades: int = 0
    test_trades: int = 0

    def to_dict(self):
        return {
            "iteration": self.iteration,
            "train_start": self.train_start.isoformat() if isinstance(self.train_start, datetime) else str(self.train_start),
            "train_end": self.train_end.isoformat() if isinstance(self.train_end, datetime) else str(self.train_end),
            "test_start": self.test_start.isoformat() if isinstance(self.test_start, datetime) else str(self.test_start),
            "test_end": self.test_end.isoformat() if isinstance(self.test_end, datetime) else str(self.test_end),
            "optimized_params": self.optimized_params,
            "train_sharpe": self.train_sharpe,
            "train_returns": self.train_returns,
            "train_max_dd": self.train_max_dd,
            "train_win_rate": self.train_win_rate,
            "test_sharpe": self.test_sharpe,
            "test_returns": self.test_returns,
            "test_max_dd": self.test_max_dd,
            "test_win_rate": self.test_win_rate,
            "sharpe_degradation": self.sharpe_degradation,
            "returns_degradation": self.returns_degradation,
            "train_trades": self.train_trades,
            "test_trades": self.test_trades,
        }


@dataclass
class WalkForwardSummary:
    """Aggregate results from all walk-forward iterations."""

    total_iterations: int
    config: WalkForwardConfig

    # Aggregate metrics
    avg_test_sharpe: float = 0.0
    avg_test_returns: float = 0.0
    avg_test_max_dd: float = 0.0
    avg_test_win_rate: float = 0.0

    # Degradation analysis
    avg_sharpe_degradation: float = 0.0
    avg_returns_degradation: float = 0.0

    # Consistency metrics
    positive_test_periods: int = 0  # Periods with positive returns
    sharpe_above_1_periods: int = 0  # Periods with Sharpe > 1

    # Overfitting detection
    overfitting_detected: bool = False
    overfitting_reason: str = ""

    # All iteration results
    iterations: List[WalkForwardResult] = field(default_factory=list)

    def to_dict(self):
        return {
            "total_iterations": self.total_iterations,
            "config": self.config.to_dict(),
            "avg_test_sharpe": self.avg_test_sharpe,
            "avg_test_returns": self.avg_test_returns,
            "avg_test_max_dd": self.avg_test_max_dd,
            "avg_test_win_rate": self.avg_test_win_rate,
            "avg_sharpe_degradation": self.avg_sharpe_degradation,
            "avg_returns_degradation": self.avg_returns_degradation,
            "positive_test_periods": self.positive_test_periods,
            "sharpe_above_1_periods": self.sharpe_above_1_periods,
            "overfitting_detected": self.overfitting_detected,
            "overfitting_reason": self.overfitting_reason,
            "iterations": [it.to_dict() for it in self.iterations],
        }


class WalkForwardTester:
    """Walk-forward testing framework for trading strategies.

    Example:
        >>> def backtest_func(df_train, df_test, params):
        ...     # Run backtest with params
        ...     return {
        ...         'sharpe': 1.5,
        ...         'returns': 0.15,
        ...         'max_dd': -0.08,
        ...         'win_rate': 0.58,
        ...         'trades': 50,
        ...     }
        >>>
        >>> def optimize_func(df_train):
        ...     # Optimize parameters on training data
        ...     return {'ema_fast': 10, 'ema_slow': 20}
        >>>
        >>> tester = WalkForwardTester(
        ...     backtest_func=backtest_func,
        ...     optimize_func=optimize_func,
        ... )
        >>> summary = tester.run(df_historical)
        >>> print(f"Overfitting detected: {summary.overfitting_detected}")
    """

    def __init__(
        self,
        backtest_func: Callable[[pd.DataFrame, pd.DataFrame, Dict], Dict],
        optimize_func: Optional[Callable[[pd.DataFrame], Dict]] = None,
        config: Optional[WalkForwardConfig] = None,
    ):
        """Initialize walk-forward tester.

        Args:
            backtest_func: Function that runs backtest
                Args: (df_train, df_test, params)
                Returns: dict with keys: sharpe, returns, max_dd, win_rate, trades
            optimize_func: Optional function to optimize parameters on training data
                Args: (df_train)
                Returns: dict with optimized parameters
            config: Walk-forward configuration
        """
        self.backtest_func = backtest_func
        self.optimize_func = optimize_func
        self.config = config or WalkForwardConfig()

    def run(
        self,
        df: pd.DataFrame,
        base_params: Optional[Dict[str, Any]] = None,
    ) -> WalkForwardSummary:
        """Run walk-forward test on historical data.

        Args:
            df: Historical OHLCV data with DatetimeIndex
            base_params: Base parameters (used if no optimization)

        Returns:
            WalkForwardSummary with all results
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have DatetimeIndex for walk-forward testing")

        base_params = base_params or {}
        results: List[WalkForwardResult] = []

        # Generate walk-forward windows
        windows = self._generate_windows(df)

        LOG.info(f"Running walk-forward test with {len(windows)} iterations")

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            LOG.info(
                f"Iteration {i+1}/{len(windows)}: "
                f"Train {train_start.date()} to {train_end.date()}, "
                f"Test {test_start.date()} to {test_end.date()}"
            )

            # Split data
            df_train = df[train_start:train_end]
            df_test = df[test_start:test_end]

            # Validate minimum samples
            if len(df_train) < self.config.min_train_samples:
                LOG.warning(f"Insufficient training samples ({len(df_train)}), skipping iteration")
                continue
            if len(df_test) < self.config.min_test_samples:
                LOG.warning(f"Insufficient test samples ({len(df_test)}), skipping iteration")
                continue

            # Optimize parameters on training data
            if self.config.optimize_on_train and self.optimize_func:
                try:
                    optimized_params = self.optimize_func(df_train)
                except Exception as e:
                    LOG.error(f"Optimization failed: {e}")
                    optimized_params = base_params
            else:
                optimized_params = base_params

            # Run backtest on training data
            try:
                train_metrics = self.backtest_func(df_train, df_train, optimized_params)
            except Exception as e:
                LOG.error(f"Training backtest failed: {e}")
                continue

            # Run backtest on test data (out-of-sample)
            try:
                test_metrics = self.backtest_func(df_train, df_test, optimized_params)
            except Exception as e:
                LOG.error(f"Test backtest failed: {e}")
                continue

            # Calculate degradation
            sharpe_deg = (
                (test_metrics['sharpe'] - train_metrics['sharpe']) / abs(train_metrics['sharpe'])
                if train_metrics['sharpe'] != 0 else 0.0
            )
            returns_deg = (
                (test_metrics['returns'] - train_metrics['returns']) / abs(train_metrics['returns'])
                if train_metrics['returns'] != 0 else 0.0
            )

            # Store result
            result = WalkForwardResult(
                iteration=i + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                optimized_params=optimized_params,
                train_sharpe=train_metrics.get('sharpe', 0.0),
                train_returns=train_metrics.get('returns', 0.0),
                train_max_dd=train_metrics.get('max_dd', 0.0),
                train_win_rate=train_metrics.get('win_rate', 0.0),
                test_sharpe=test_metrics.get('sharpe', 0.0),
                test_returns=test_metrics.get('returns', 0.0),
                test_max_dd=test_metrics.get('max_dd', 0.0),
                test_win_rate=test_metrics.get('win_rate', 0.0),
                sharpe_degradation=sharpe_deg,
                returns_degradation=returns_deg,
                train_trades=train_metrics.get('trades', 0),
                test_trades=test_metrics.get('trades', 0),
            )
            results.append(result)

        # Generate summary
        summary = self._generate_summary(results)

        return summary

    def _generate_windows(
        self,
        df: pd.DataFrame,
    ) -> List[tuple[datetime, datetime, datetime, datetime]]:
        """Generate train/test windows for walk-forward.

        Args:
            df: DataFrame with DatetimeIndex

        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        windows = []

        start_date = df.index[0]
        end_date = df.index[-1]

        current_date = start_date

        while True:
            # Calculate window boundaries
            train_start = current_date

            if self.config.use_anchored_walk:
                # Anchored (expanding window): always start from beginning
                train_start = start_date

            train_end = current_date + timedelta(days=self.config.train_window_days)
            test_start = train_end
            test_end = test_start + timedelta(days=self.config.test_window_days)

            # Check if we have enough data
            if test_end > end_date:
                break

            windows.append((train_start, train_end, test_start, test_end))

            # Move forward
            current_date += timedelta(days=self.config.step_days)

        return windows

    def _generate_summary(
        self,
        results: List[WalkForwardResult],
    ) -> WalkForwardSummary:
        """Generate aggregate summary from iteration results.

        Args:
            results: List of iteration results

        Returns:
            WalkForwardSummary
        """
        if not results:
            return WalkForwardSummary(
                total_iterations=0,
                config=self.config,
                overfitting_detected=True,
                overfitting_reason="No valid iterations completed",
            )

        # Calculate averages
        avg_test_sharpe = np.mean([r.test_sharpe for r in results])
        avg_test_returns = np.mean([r.test_returns for r in results])
        avg_test_max_dd = np.mean([r.test_max_dd for r in results])
        avg_test_win_rate = np.mean([r.test_win_rate for r in results])

        avg_sharpe_deg = np.mean([r.sharpe_degradation for r in results])
        avg_returns_deg = np.mean([r.returns_degradation for r in results])

        # Count positive periods
        positive_periods = sum(1 for r in results if r.test_returns > 0)
        sharpe_above_1 = sum(1 for r in results if r.test_sharpe > 1.0)

        # Detect overfitting
        overfitting, reason = self._detect_overfitting(results, avg_sharpe_deg, avg_returns_deg)

        summary = WalkForwardSummary(
            total_iterations=len(results),
            config=self.config,
            avg_test_sharpe=avg_test_sharpe,
            avg_test_returns=avg_test_returns,
            avg_test_max_dd=avg_test_max_dd,
            avg_test_win_rate=avg_test_win_rate,
            avg_sharpe_degradation=avg_sharpe_deg,
            avg_returns_degradation=avg_returns_deg,
            positive_test_periods=positive_periods,
            sharpe_above_1_periods=sharpe_above_1,
            overfitting_detected=overfitting,
            overfitting_reason=reason,
            iterations=results,
        )

        return summary

    def _detect_overfitting(
        self,
        results: List[WalkForwardResult],
        avg_sharpe_deg: float,
        avg_returns_deg: float,
    ) -> tuple[bool, str]:
        """Detect if strategy shows signs of overfitting.

        Args:
            results: Iteration results
            avg_sharpe_deg: Average Sharpe degradation
            avg_returns_deg: Average returns degradation

        Returns:
            (is_overfitting, reason)
        """
        # Check 1: Severe degradation
        if avg_sharpe_deg < -0.50:  # Test Sharpe 50% worse than train
            return True, f"Severe Sharpe degradation: {avg_sharpe_deg:.1%}"

        if avg_returns_deg < -0.50:
            return True, f"Severe returns degradation: {avg_returns_deg:.1%}"

        # Check 2: Majority of test periods negative
        positive_ratio = sum(1 for r in results if r.test_returns > 0) / len(results)
        if positive_ratio < 0.40:  # Less than 40% positive periods
            return True, f"Only {positive_ratio:.0%} of test periods profitable"

        # Check 3: Test Sharpe consistently poor
        avg_test_sharpe = np.mean([r.test_sharpe for r in results])
        if avg_test_sharpe < 0.5:
            return True, f"Low average test Sharpe: {avg_test_sharpe:.2f}"

        # Check 4: High variance in test results (inconsistent)
        test_returns_std = np.std([r.test_returns for r in results])
        test_returns_mean = np.mean([r.test_returns for r in results])
        if test_returns_mean != 0:
            cv = abs(test_returns_std / test_returns_mean)
            if cv > 2.0:  # Coefficient of variation > 2
                return True, f"Highly inconsistent test results (CV={cv:.2f})"

        # No overfitting detected
        return False, "Walk-forward validation passed"
