"""Health check system for trading platform.

Checks health of:
- Database connection
- ML models
- External APIs (exchange)
- Risk management systems
- Memory usage
- Disk space
"""

from __future__ import annotations

import asyncio
import time
import psutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Awaitable, Any
from enum import Enum

from utils.structured_logging import get_logger

LOG = get_logger("ai_trader.health")


class HealthStatus(str, Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    status: HealthStatus
    message: str
    latency_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "latency_ms": round(self.latency_ms, 2),
            "timestamp": self.timestamp,
            "details": self.details,
        }


@dataclass
class SystemHealth:
    """Overall system health."""
    status: HealthStatus
    checks: List[HealthCheckResult]
    uptime_seconds: float
    version: str = "1.0.0"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "version": self.version,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "uptime_human": f"{self.uptime_seconds/3600:.1f}h",
            "timestamp": self.timestamp,
            "checks": [c.to_dict() for c in self.checks],
            "summary": {
                "total": len(self.checks),
                "healthy": sum(1 for c in self.checks if c.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for c in self.checks if c.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for c in self.checks if c.status == HealthStatus.UNHEALTHY),
            }
        }


class HealthChecker:
    """Manages health checks for the trading system."""

    def __init__(self):
        self._start_time = time.time()
        self._checks: Dict[str, Callable[[], Awaitable[HealthCheckResult]]] = {}

    def register_check(
        self,
        name: str,
        check_func: Callable[[], Awaitable[HealthCheckResult]],
    ) -> None:
        """Register a health check function.

        Args:
            name: Check name (e.g., "database")
            check_func: Async function that returns HealthCheckResult
        """
        self._checks[name] = check_func
        LOG.info(f"Registered health check: {name}")

    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a single health check.

        Args:
            name: Check name

        Returns:
            HealthCheckResult
        """
        if name not in self._checks:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Check '{name}' not registered",
                latency_ms=0,
            )

        start = time.time()
        try:
            result = await asyncio.wait_for(
                self._checks[name](),
                timeout=10.0  # 10 second timeout per check
            )
            return result
        except asyncio.TimeoutError:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Health check timed out (>10s)",
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return HealthCheckResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    async def run_all_checks(self) -> SystemHealth:
        """Run all registered health checks.

        Returns:
            SystemHealth with results from all checks
        """
        results = []

        for name in self._checks:
            result = await self.run_check(name)
            results.append(result)

        # Determine overall status
        if any(c.status == HealthStatus.UNHEALTHY for c in results):
            overall_status = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in results):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        uptime = time.time() - self._start_time

        return SystemHealth(
            status=overall_status,
            checks=results,
            uptime_seconds=uptime,
        )


# Global singleton
_health_checker: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get or create global health checker."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
        _register_default_checks(_health_checker)
    return _health_checker


def _register_default_checks(checker: HealthChecker) -> None:
    """Register default health checks."""

    # System resources check
    async def check_system_resources() -> HealthCheckResult:
        start = time.time()
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Status logic
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                status = HealthStatus.UNHEALTHY
                message = "System resources critical"
            elif cpu_percent > 70 or memory.percent > 75 or disk.percent > 80:
                status = HealthStatus.DEGRADED
                message = "System resources high"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources OK"

            return HealthCheckResult(
                name="system_resources",
                status=status,
                message=message,
                latency_ms=(time.time() - start) * 1000,
                details={
                    "cpu_percent": round(cpu_percent, 1),
                    "memory_percent": round(memory.percent, 1),
                    "disk_percent": round(disk.percent, 1),
                }
            )
        except Exception as e:
            return HealthCheckResult(
                name="system_resources",
                status=HealthStatus.UNHEALTHY,
                message=f"Failed to check resources: {e}",
                latency_ms=(time.time() - start) * 1000,
            )

    checker.register_check("system_resources", check_system_resources)


async def check_database_health(db_session_factory) -> HealthCheckResult:
    """Check database connectivity.

    Args:
        db_session_factory: Function to create DB session

    Returns:
        HealthCheckResult
    """
    start = time.time()
    try:
        async with db_session_factory() as session:
            # Simple query to test connection
            result = await session.execute("SELECT 1")
            await result.fetchone()

        latency = (time.time() - start) * 1000

        if latency > 500:
            status = HealthStatus.DEGRADED
            message = f"Database slow ({latency:.0f}ms)"
        else:
            status = HealthStatus.HEALTHY
            message = "Database OK"

        return HealthCheckResult(
            name="database",
            status=status,
            message=message,
            latency_ms=latency,
        )
    except Exception as e:
        return HealthCheckResult(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Database connection failed: {e}",
            latency_ms=(time.time() - start) * 1000,
        )


async def check_ml_models_health(lstm_generator, meta_learner) -> HealthCheckResult:
    """Check ML models availability.

    Args:
        lstm_generator: LSTM signal generator
        meta_learner: Meta-learner model

    Returns:
        HealthCheckResult
    """
    start = time.time()
    try:
        lstm_ok = lstm_generator is not None
        meta_ok = meta_learner is not None and meta_learner.is_trained

        if lstm_ok and meta_ok:
            status = HealthStatus.HEALTHY
            message = "All ML models loaded"
        elif lstm_ok or meta_ok:
            status = HealthStatus.DEGRADED
            message = "Some ML models unavailable"
        else:
            status = HealthStatus.DEGRADED  # Not UNHEALTHY as system can work without ML
            message = "ML models not loaded (fallback to technical)"

        return HealthCheckResult(
            name="ml_models",
            status=status,
            message=message,
            latency_ms=(time.time() - start) * 1000,
            details={
                "lstm_available": lstm_ok,
                "meta_learner_available": meta_ok,
            }
        )
    except Exception as e:
        return HealthCheckResult(
            name="ml_models",
            status=HealthStatus.DEGRADED,
            message=f"ML check failed: {e}",
            latency_ms=(time.time() - start) * 1000,
        )


async def check_risk_engine_health(risk_sizer, correlation_tracker, gap_protector) -> HealthCheckResult:
    """Check risk management systems.

    Args:
        risk_sizer: Advanced position sizer
        correlation_tracker: Portfolio correlation tracker
        gap_protector: Gap protection system

    Returns:
        HealthCheckResult
    """
    start = time.time()
    try:
        sizer_ok = risk_sizer is not None
        corr_ok = correlation_tracker is not None
        gap_ok = gap_protector is not None

        all_ok = sizer_ok and corr_ok and gap_ok

        if all_ok:
            status = HealthStatus.HEALTHY
            message = "All risk systems active"
        elif sizer_ok:  # At minimum need position sizer
            status = HealthStatus.DEGRADED
            message = "Some risk systems unavailable"
        else:
            status = HealthStatus.UNHEALTHY
            message = "Critical risk systems missing"

        return HealthCheckResult(
            name="risk_engine",
            status=status,
            message=message,
            latency_ms=(time.time() - start) * 1000,
            details={
                "position_sizer": sizer_ok,
                "correlation_tracker": corr_ok,
                "gap_protector": gap_ok,
            }
        )
    except Exception as e:
        return HealthCheckResult(
            name="risk_engine",
            status=HealthStatus.UNHEALTHY,
            message=f"Risk engine check failed: {e}",
            latency_ms=(time.time() - start) * 1000,
        )
