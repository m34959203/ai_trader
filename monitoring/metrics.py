"""Comprehensive metrics collection system for trading platform.

Tracks:
- Trade execution metrics
- System performance
- ML model performance
- Risk management metrics
- API performance
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from collections import defaultdict, deque
import threading

from utils.structured_logging import get_logger

LOG = get_logger("ai_trader.metrics")


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricStats:
    """Statistical summary of a metric."""
    count: int = 0
    sum: float = 0.0
    min: float = float('inf')
    max: float = float('-inf')
    last: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        self.sum += value
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        self.last = value

    @property
    def mean(self) -> float:
        return self.sum / self.count if self.count > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "count": self.count,
            "sum": self.sum,
            "min": self.min if self.min != float('inf') else 0.0,
            "max": self.max if self.max != float('-inf') else 0.0,
            "mean": self.mean,
            "last": self.last,
        }


class MetricsCollector:
    """Thread-safe metrics collector with retention and aggregation."""

    def __init__(self, retention_seconds: int = 3600):
        """Initialize metrics collector.

        Args:
            retention_seconds: How long to keep raw metric points
        """
        self.retention_seconds = retention_seconds
        self._lock = threading.RLock()

        # Raw metric points (for time series)
        self._metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))

        # Aggregated stats (for current period)
        self._stats: Dict[str, MetricStats] = defaultdict(MetricStats)

        # System info
        self._start_time = time.time()
        self._info: Dict[str, Any] = {
            "version": "1.0.0",
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

    def record(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value.

        Args:
            name: Metric name (e.g., "trade.execution_latency_ms")
            value: Metric value
            tags: Optional tags for grouping (e.g., {"symbol": "BTCUSDT"})
        """
        now = time.time()
        point = MetricPoint(timestamp=now, value=value, tags=tags or {})

        with self._lock:
            self._metrics[name].append(point)
            self._stats[name].update(value)

            # Clean old points
            self._clean_old_points(name, now)

    def increment(self, name: str, delta: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
        """Increment a counter metric.

        Args:
            name: Counter name (e.g., "trade.count")
            delta: Amount to increment (default 1.0)
            tags: Optional tags
        """
        current = self._stats[name].sum
        self.record(name, current + delta, tags)

    def _clean_old_points(self, name: str, now: float) -> None:
        """Remove metric points older than retention period."""
        cutoff = now - self.retention_seconds
        points = self._metrics[name]

        while points and points[0].timestamp < cutoff:
            points.popleft()

    def get_stats(self, name: str) -> Dict[str, Any]:
        """Get aggregated statistics for a metric.

        Args:
            name: Metric name

        Returns:
            Dict with count, sum, min, max, mean, last
        """
        with self._lock:
            return self._stats[name].to_dict()

    def get_time_series(
        self,
        name: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[MetricPoint]:
        """Get time series data for a metric.

        Args:
            name: Metric name
            start_time: Unix timestamp (default: all available)
            end_time: Unix timestamp (default: now)

        Returns:
            List of metric points in time range
        """
        with self._lock:
            points = list(self._metrics[name])

        if start_time:
            points = [p for p in points if p.timestamp >= start_time]
        if end_time:
            points = [p for p in points if p.timestamp <= end_time]

        return points

    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all current metric statistics.

        Returns:
            Dict mapping metric names to their stats
        """
        with self._lock:
            return {name: stats.to_dict() for name, stats in self._stats.items()}

    def get_percentiles(
        self,
        name: str,
        percentiles: List[float] = [0.5, 0.95, 0.99],
    ) -> Dict[str, float]:
        """Calculate percentiles for a metric.

        Args:
            name: Metric name
            percentiles: List of percentiles to calculate (0.0 to 1.0)

        Returns:
            Dict mapping percentile to value (e.g., {"p50": 123.4})
        """
        with self._lock:
            points = list(self._metrics[name])

        if not points:
            return {f"p{int(p*100)}": 0.0 for p in percentiles}

        values = sorted([p.value for p in points])
        result = {}

        for p in percentiles:
            idx = int(len(values) * p)
            idx = min(idx, len(values) - 1)
            result[f"p{int(p*100)}"] = values[idx]

        return result

    def reset_stats(self) -> None:
        """Reset all aggregated statistics (keeps raw points)."""
        with self._lock:
            self._stats.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary.

        Returns:
            Dict with system info and all metrics
        """
        uptime = time.time() - self._start_time

        return {
            "info": {
                **self._info,
                "uptime_seconds": uptime,
                "uptime_human": f"{uptime/3600:.1f}h",
            },
            "metrics": self.get_all_metrics(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# Global singleton
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector(retention_seconds=3600)
    return _metrics_collector


def record_metric(name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
    """Record a metric (convenience function)."""
    get_metrics_collector().record(name, value, tags)


def increment_counter(name: str, delta: float = 1.0, tags: Optional[Dict[str, str]] = None) -> None:
    """Increment a counter (convenience function)."""
    get_metrics_collector().increment(name, delta, tags)


# Metric name constants
class MetricNames:
    """Standard metric names for consistency."""

    # Trade metrics
    TRADE_COUNT = "trade.count"
    TRADE_EXECUTION_LATENCY_MS = "trade.execution_latency_ms"
    TRADE_SLIPPAGE_BPS = "trade.slippage_bps"
    TRADE_PNL = "trade.pnl"
    TRADE_VOLUME = "trade.volume"

    # Signal metrics
    SIGNAL_GENERATION_TIME_MS = "signal.generation_time_ms"
    SIGNAL_COUNT = "signal.count"
    SIGNAL_CONFIDENCE = "signal.confidence"

    # ML metrics
    ML_INFERENCE_TIME_MS = "ml.inference_time_ms"
    ML_PREDICTION_CONFIDENCE = "ml.prediction_confidence"
    ML_FAILURE_COUNT = "ml.failure_count"

    # Risk metrics
    RISK_CHECK_TIME_MS = "risk.check_time_ms"
    RISK_POSITION_SIZE_ADJUSTED = "risk.position_size_adjusted"
    RISK_TRADE_BLOCKED_COUNT = "risk.trade_blocked_count"
    RISK_KELLY_FRACTION = "risk.kelly_fraction"
    RISK_CORRELATION_FACTOR = "risk.correlation_factor"

    # System metrics
    SYSTEM_REQUEST_COUNT = "system.request_count"
    SYSTEM_REQUEST_LATENCY_MS = "system.request_latency_ms"
    SYSTEM_ERROR_COUNT = "system.error_count"
    SYSTEM_DB_QUERY_TIME_MS = "system.db_query_time_ms"

    # Portfolio metrics
    PORTFOLIO_EQUITY = "portfolio.equity"
    PORTFOLIO_PNL_DAILY = "portfolio.pnl_daily"
    PORTFOLIO_POSITION_COUNT = "portfolio.position_count"
    PORTFOLIO_EXPOSURE = "portfolio.exposure"
