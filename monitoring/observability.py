"""Prometheus integration and SLO tracking helpers for Stage 3."""
from __future__ import annotations

import threading
from typing import Dict, Iterable, Optional

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from monitoring.slo import SLO, SLOTracker


class ObservabilityHub:
    """Central registry for Prometheus metrics and SLO progress."""

    def __init__(self, slos: Optional[Iterable[SLO]] = None) -> None:
        default_slos = list(slos or [
            SLO(name="order_success", target=0.98, window=500),
            SLO(name="latency_under_2s", target=0.95, window=500),
            SLO(name="executor_availability", target=0.99, window=200),
        ])
        self._tracker = SLOTracker(default_slos)
        self._registry = CollectorRegistry()
        self._order_latency = Histogram(
            "ai_trader_order_latency_seconds",
            "Latency of order placement attempts",
            labelnames=("symbol",),
            buckets=(0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, float("inf")),
            registry=self._registry,
        )
        self._orders_total = Counter(
            "ai_trader_orders_total",
            "Total orders routed by status",
            labelnames=("status",),
            registry=self._registry,
        )
        self._failovers = Counter(
            "ai_trader_executor_failovers_total",
            "Executor failovers grouped by context and recovery result",
            labelnames=("context", "recovered"),
            registry=self._registry,
        )
        self._slo_actual = Gauge(
            "ai_trader_slo_compliance_ratio",
            "Current SLO compliance ratio",
            labelnames=("slo",),
            registry=self._registry,
        )
        self._slo_target = Gauge(
            "ai_trader_slo_target_ratio",
            "Target SLO ratio",
            labelnames=("slo",),
            registry=self._registry,
        )
        self._lock = threading.Lock()
        self._refresh_slo_gauges()

    @property
    def prometheus_content_type(self) -> str:
        return CONTENT_TYPE_LATEST

    def _refresh_slo_gauges(self) -> None:
        report = self._tracker.report()
        for name, stats in report.items():
            self._slo_actual.labels(name).set(stats["actual"])
            self._slo_target.labels(name).set(stats["target"])

    def record_order(self, *, symbol: str, success: bool, latency: float) -> None:
        """Record a trading attempt for success SLOs and Prometheus metrics."""

        symbol_label = symbol or "UNKNOWN"
        latency_value = max(float(latency), 0.0)
        status = "success" if success else "failure"
        with self._lock:
            self._orders_total.labels(status=status).inc()
            self._order_latency.labels(symbol_label).observe(latency_value)
            self._tracker.record("order_success", success)
            self._tracker.record("latency_under_2s", latency_value <= 2.0)
            self._refresh_slo_gauges()

    def record_failover(self, *, context: str, recovered: bool) -> None:
        with self._lock:
            self._failovers.labels(context=context, recovered=str(bool(recovered)).lower()).inc()
            self._tracker.record("executor_availability", recovered)
            self._refresh_slo_gauges()

    def export_prometheus(self) -> bytes:
        with self._lock:
            return generate_latest(self._registry)

    def slo_report(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return self._tracker.report()


OBSERVABILITY = ObservabilityHub()
