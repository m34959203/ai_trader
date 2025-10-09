"""Prometheus integration and SLO tracking helpers for Stage 3."""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from monitoring.slo import SLO, SLOTracker


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class BrokerLatencySample:
    broker: str
    operation: str
    status: str
    latency_seconds: float
    observed_at: str


@dataclass(frozen=True)
class ThreadFailureSample:
    thread: str
    stage: str
    recovered: bool
    reason: Optional[str]
    observed_at: str


@dataclass(frozen=True)
class MarketDataQualitySample:
    source: str
    symbol: str
    interval: str
    score: float
    staleness_seconds: float
    missing_bars: int
    observed_at: str


class ObservabilityHub:
    """Central registry for Prometheus metrics and SLO progress."""

    def __init__(self, slos: Optional[Iterable[SLO]] = None) -> None:
        default_slos = list(slos or [
            SLO(name="order_success", target=0.98, window=500),
            SLO(name="latency_under_2s", target=0.95, window=500),
            SLO(name="executor_availability", target=0.99, window=200),
            SLO(name="broker_latency_under_1s", target=0.97, window=500),
            SLO(name="market_data_quality_score", target=0.95, window=500),
            SLO(name="market_data_freshness", target=0.98, window=500),
            SLO(name="stream_thread_recovery", target=0.99, window=200),
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
        self._broker_latency = Histogram(
            "ai_trader_broker_latency_seconds",
            "Round-trip latency for broker API calls",
            labelnames=("broker", "operation", "status"),
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, float("inf")),
            registry=self._registry,
        )
        self._thread_failures = Counter(
            "ai_trader_thread_failures_total",
            "Background thread or task failures by stage and recovery status",
            labelnames=("thread", "stage", "recovered"),
            registry=self._registry,
        )
        self._market_data_quality = Gauge(
            "ai_trader_market_data_quality",
            "Market data quality metrics (score/staleness/missing bars)",
            labelnames=("source", "symbol", "interval", "metric"),
            registry=self._registry,
        )
        self._lock = threading.Lock()
        self._broker_latency_events: Dict[Tuple[str, str, str], BrokerLatencySample] = {}
        self._thread_failure_events: Dict[Tuple[str, str], ThreadFailureSample] = {}
        self._market_data_events: Dict[Tuple[str, str, str], MarketDataQualitySample] = {}
        self._broker_latency_threshold = 1.0
        self._market_quality_threshold = 0.95
        self._market_freshness_threshold = 5.0
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

    def record_broker_latency(
        self,
        *,
        broker: str,
        operation: str,
        latency: float,
        status: str,
    ) -> None:
        broker_label = broker or "unknown"
        op_label = operation or "unknown"
        status_label = status or "unknown"
        latency_value = max(float(latency), 0.0)
        sample = BrokerLatencySample(
            broker=broker_label,
            operation=op_label,
            status=status_label,
            latency_seconds=latency_value,
            observed_at=_utcnow(),
        )
        with self._lock:
            self._broker_latency.labels(broker_label, op_label, status_label).observe(latency_value)
            self._broker_latency_events[(broker_label, op_label, status_label)] = sample
            self._tracker.record(
                "broker_latency_under_1s",
                bool(status_label == "success") and latency_value <= self._broker_latency_threshold,
            )
            self._refresh_slo_gauges()

    def record_thread_failure(
        self,
        *,
        thread: str,
        stage: str,
        recovered: bool,
        reason: Optional[str] = None,
    ) -> None:
        thread_label = thread or "unknown"
        stage_label = stage or "runtime"
        recovered_label = str(bool(recovered)).lower()
        sample = ThreadFailureSample(
            thread=thread_label,
            stage=stage_label,
            recovered=recovered,
            reason=reason,
            observed_at=_utcnow(),
        )
        with self._lock:
            self._thread_failures.labels(thread_label, stage_label, recovered_label).inc()
            self._thread_failure_events[(thread_label, stage_label)] = sample
            self._tracker.record("stream_thread_recovery", recovered)
            self._refresh_slo_gauges()

    def record_market_data_quality(
        self,
        *,
        source: str,
        symbol: str,
        interval: str,
        score: float,
        staleness_seconds: Optional[float] = None,
        missing_bars: int = 0,
    ) -> None:
        src = source or "unknown"
        sym = symbol or "UNKNOWN"
        itv = interval or "unknown"
        score_value = _clamp(float(score), 0.0, 1.0)
        staleness_value = max(float(staleness_seconds or 0.0), 0.0)
        missing_value = max(int(missing_bars or 0), 0)
        sample = MarketDataQualitySample(
            source=src,
            symbol=sym,
            interval=itv,
            score=score_value,
            staleness_seconds=staleness_value,
            missing_bars=missing_value,
            observed_at=_utcnow(),
        )
        with self._lock:
            self._market_data_quality.labels(src, sym, itv, "score").set(score_value)
            self._market_data_quality.labels(src, sym, itv, "staleness_seconds").set(staleness_value)
            self._market_data_quality.labels(src, sym, itv, "missing_bars").set(float(missing_value))
            self._market_data_events[(src, sym, itv)] = sample
            self._tracker.record("market_data_quality_score", score_value >= self._market_quality_threshold)
            self._tracker.record(
                "market_data_freshness",
                staleness_value <= self._market_freshness_threshold,
            )
            self._refresh_slo_gauges()

    def export_prometheus(self) -> bytes:
        with self._lock:
            return generate_latest(self._registry)

    def slo_report(self) -> Dict[str, Dict[str, float]]:
        with self._lock:
            return self._tracker.report()

    def broker_latency_snapshot(self) -> Dict[str, List[Dict[str, object]]]:
        with self._lock:
            observations = [
                {
                    "broker": sample.broker,
                    "operation": sample.operation,
                    "status": sample.status,
                    "latency_seconds": sample.latency_seconds,
                    "observed_at": sample.observed_at,
                }
                for sample in self._broker_latency_events.values()
            ]
            return {
                "threshold_seconds": self._broker_latency_threshold,
                "observations": sorted(observations, key=lambda item: item["observed_at"], reverse=True),
            }

    def thread_failure_snapshot(self) -> Dict[str, object]:
        with self._lock:
            events = [
                {
                    "thread": sample.thread,
                    "stage": sample.stage,
                    "recovered": sample.recovered,
                    "reason": sample.reason,
                    "observed_at": sample.observed_at,
                }
                for sample in self._thread_failure_events.values()
            ]
            report = self._tracker.report().get("stream_thread_recovery", {})
            return {
                "events": sorted(events, key=lambda item: item["observed_at"], reverse=True),
                "slo": report,
            }

    def market_data_snapshot(self) -> Dict[str, List[Dict[str, object]]]:
        with self._lock:
            items = [
                {
                    "source": sample.source,
                    "symbol": sample.symbol,
                    "interval": sample.interval,
                    "score": sample.score,
                    "staleness_seconds": sample.staleness_seconds,
                    "missing_bars": sample.missing_bars,
                    "observed_at": sample.observed_at,
                }
                for sample in self._market_data_events.values()
            ]
            return {
                "quality_threshold": self._market_quality_threshold,
                "freshness_threshold_seconds": self._market_freshness_threshold,
                "samples": sorted(items, key=lambda item: item["observed_at"], reverse=True),
            }


OBSERVABILITY = ObservabilityHub()
