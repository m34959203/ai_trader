from __future__ import annotations

import asyncio
import logging
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

from .alerts import alert_crit, alert_info, alert_warn


@dataclass(frozen=True)
class Threshold:
    """Пороговые значения для метрик ресурсов."""

    warn: Optional[float]
    crit: Optional[float]

    def evaluate(self, value: Optional[float]) -> str:
        """Возвращает уровень ""warn""/""crit""/""ok""/""skip"" для значения."""

        if value is None:
            return "skip"
        if isinstance(value, float) and math.isnan(value):
            return "skip"
        if self.crit is not None and value >= self.crit:
            return "crit"
        if self.warn is not None and value >= self.warn:
            return "warn"
        return "ok"

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {"warn": self.warn, "crit": self.crit}


@dataclass(frozen=True)
class ResourceMonitorConfig:
    """Конфигурация мониторинга системных ресурсов."""

    enabled: bool = True
    interval_sec: float = 30.0
    info_interval_sec: float = 600.0
    warn_streak: int = 2
    crit_streak: int = 3
    recovery_streak: int = 2
    exit_on_crit: bool = False
    disk_path: Path = Path("/")
    cpu: Threshold = field(default_factory=lambda: Threshold(85.0, 95.0))
    ram: Threshold = field(default_factory=lambda: Threshold(85.0, 95.0))
    rss: Threshold = field(default_factory=lambda: Threshold(2048.0, 3072.0))
    disk: Threshold = field(default_factory=lambda: Threshold(90.0, 97.0))

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        val = os.getenv(name)
        if val is None:
            return default
        return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        val = os.getenv(name)
        if val is None or str(val).strip() == "":
            return float(default)
        try:
            return float(val)
        except Exception:
            return float(default)

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        val = os.getenv(name)
        if val is None or str(val).strip() == "":
            return int(default)
        try:
            return int(float(val))
        except Exception:
            return int(default)

    @staticmethod
    def _env_optional_float(name: str, default: Optional[float]) -> Optional[float]:
        val = os.getenv(name)
        if val is None or str(val).strip() == "":
            return default
        lowered = str(val).strip().lower()
        if lowered in {"none", "null", "off", "disable", "disabled"}:
            return None
        try:
            parsed = float(val)
            return max(0.0, parsed)
        except Exception:
            return default

    @classmethod
    def _threshold_from_env(cls, prefix: str, *, default_warn: Optional[float], default_crit: Optional[float]) -> Threshold:
        warn = cls._env_optional_float(f"{prefix}_WARN", default_warn)
        crit = cls._env_optional_float(f"{prefix}_CRIT", default_crit)
        return Threshold(warn=warn, crit=crit)

    @classmethod
    def from_env(cls) -> "ResourceMonitorConfig":
        enabled = cls._env_bool("RESOURCE_MONITOR_ENABLED", True)
        interval = max(5.0, cls._env_float("RESOURCE_MONITOR_INTERVAL_SEC", 30.0))
        info_interval = cls._env_float("RESOURCE_MONITOR_INFO_INTERVAL_SEC", 600.0)
        warn_streak = max(1, cls._env_int("RESOURCE_MONITOR_WARN_STREAK", 2))
        crit_streak = max(1, cls._env_int("RESOURCE_MONITOR_CRIT_STREAK", 3))
        recovery_streak = max(1, cls._env_int("RESOURCE_MONITOR_RECOVERY_STREAK", 2))
        exit_on_crit = cls._env_bool("RESOURCE_MONITOR_EXIT_ON_CRIT", False)
        disk_path = Path(os.getenv("RESOURCE_MONITOR_DISK_PATH", "/"))

        cpu = cls._threshold_from_env("RESOURCE_MONITOR_CPU", default_warn=85.0, default_crit=95.0)
        ram = cls._threshold_from_env("RESOURCE_MONITOR_RAM", default_warn=85.0, default_crit=95.0)
        disk = cls._threshold_from_env("RESOURCE_MONITOR_DISK", default_warn=90.0, default_crit=97.0)

        rss_warn = cls._env_optional_float("RESOURCE_MONITOR_RSS_WARN_MB", 2048.0)
        rss_crit = cls._env_optional_float("RESOURCE_MONITOR_RSS_CRIT_MB", 3072.0)
        rss = Threshold(warn=rss_warn, crit=rss_crit)

        return cls(
            enabled=enabled,
            interval_sec=interval,
            info_interval_sec=max(0.0, info_interval),
            warn_streak=warn_streak,
            crit_streak=crit_streak,
            recovery_streak=recovery_streak,
            exit_on_crit=exit_on_crit,
            disk_path=disk_path,
            cpu=cpu,
            ram=ram,
            rss=rss,
            disk=disk,
        )

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "interval_sec": self.interval_sec,
            "info_interval_sec": self.info_interval_sec,
            "warn_streak": self.warn_streak,
            "crit_streak": self.crit_streak,
            "recovery_streak": self.recovery_streak,
            "exit_on_crit": self.exit_on_crit,
            "disk_path": str(self.disk_path),
            "thresholds": {
                "cpu_percent": self.cpu.to_dict(),
                "ram_percent": self.ram.to_dict(),
                "proc_rss_mb": self.rss.to_dict(),
                "disk_percent": self.disk.to_dict(),
            },
        }


@dataclass(frozen=True)
class ResourceSnapshot:
    timestamp: int
    cpu_percent: float
    proc_cpu_percent: float
    ram_percent: float
    proc_rss_mb: float
    disk_percent: Optional[float]
    swap_percent: Optional[float]
    load_avg_1m: Optional[float]
    thread_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": round(float(self.cpu_percent), 2),
            "proc_cpu_percent": round(float(self.proc_cpu_percent), 2),
            "ram_percent": round(float(self.ram_percent), 2),
            "proc_rss_mb": round(float(self.proc_rss_mb), 2),
            "disk_percent": None if self.disk_percent is None else round(float(self.disk_percent), 2),
            "swap_percent": None if self.swap_percent is None else round(float(self.swap_percent), 2),
            "load_avg_1m": None if self.load_avg_1m is None else round(float(self.load_avg_1m), 2),
            "thread_count": int(self.thread_count),
        }


class ResourceMonitor:
    """Асинхронный монитор ресурсов с алертами и опциональным выходом."""

    def __init__(self, config: ResourceMonitorConfig):
        self.config = config
        self.log = logging.getLogger("ai_trader.resource_monitor")
        self._stop = False
        self._proc = psutil.Process(os.getpid())
        try:
            # прогреваем счётчики CPU, чтобы первые значения не были нулевыми
            self._proc.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except Exception:
            pass
        self._level_streak = 0
        self._level_value = "ok"
        self._alert_level = "ok"
        self._last_info_ts = 0.0
        self._last_snapshot: Optional[ResourceSnapshot] = None
        self._last_reasons: List[str] = []
        self._last_snapshot_ts: Optional[int] = None

    def stop(self) -> None:
        self._stop = True

    def _disk_percent(self) -> Optional[float]:
        if self.config.disk.warn is None and self.config.disk.crit is None:
            return None
        try:
            usage = shutil.disk_usage(self.config.disk_path)
        except Exception as exc:
            self.log.debug("Disk usage check failed for %s: %r", self.config.disk_path, exc)
            return None
        if usage.total <= 0:
            return None
        return (usage.used / float(usage.total)) * 100.0

    def snapshot(self) -> ResourceSnapshot:
        vm = psutil.virtual_memory()
        try:
            swap_pct = float(psutil.swap_memory().percent)
        except Exception:
            swap_pct = None
        try:
            load_avg = os.getloadavg()[0]
        except Exception:
            load_avg = None

        rss_mb = float(self._proc.memory_info().rss) / (1024.0 * 1024.0)
        cpu_pct = float(psutil.cpu_percent(interval=None))
        proc_cpu_pct = float(self._proc.cpu_percent(interval=None))

        return ResourceSnapshot(
            timestamp=int(time.time()),
            cpu_percent=cpu_pct,
            proc_cpu_percent=proc_cpu_pct,
            ram_percent=float(vm.percent),
            proc_rss_mb=rss_mb,
            disk_percent=self._disk_percent(),
            swap_percent=swap_pct,
            load_avg_1m=float(load_avg) if load_avg is not None else None,
            thread_count=int(self._proc.num_threads()),
        )

    def _evaluate(self, snap: ResourceSnapshot) -> Tuple[str, List[str]]:
        level = "ok"
        reasons: List[str] = []

        checks = [
            ("CPU", snap.cpu_percent, self.config.cpu, "%"),
            ("RAM", snap.ram_percent, self.config.ram, "%"),
            ("RSS", snap.proc_rss_mb, self.config.rss, " MB"),
            ("Disk", snap.disk_percent, self.config.disk, "%"),
        ]

        for label, value, threshold, unit in checks:
            lvl = threshold.evaluate(value) if threshold else "skip"
            if lvl == "crit":
                level = "crit"
                thr_val = threshold.crit if threshold else None
                msg = f"{label} {value:.1f}{unit} >= {thr_val:.1f}{unit}" if thr_val is not None else f"{label} high"
                reasons.append(msg)
            elif lvl == "warn" and level != "crit":
                level = "warn"
                thr_val = threshold.warn if threshold else None
                msg = f"{label} {value:.1f}{unit} >= {thr_val:.1f}{unit}" if thr_val is not None else f"{label} elevated"
                reasons.append(msg)

        return level, reasons

    async def _safe_alert(self, level: str, message: str, context: Dict[str, Any]) -> None:
        try:
            if level == "info":
                await alert_info(message, context=context)
            elif level == "warn":
                await alert_warn(message, context=context)
            else:
                await alert_crit(message, context=context)
        except Exception as exc:
            self.log.error("Failed to send %s alert: %r", level, exc)

    async def _handle_level(self, level: str, reasons: List[str], snap: ResourceSnapshot) -> None:
        if level == self._level_value:
            self._level_streak += 1
        else:
            self._level_value = level
            self._level_streak = 1

        ctx = {
            "level": level,
            "reasons": reasons,
            "streak": self._level_streak,
            "snapshot": snap.to_dict(),
            "config": self.config.to_public_dict(),
        }

        self._last_snapshot = snap
        self._last_reasons = list(reasons)
        self._last_snapshot_ts = snap.timestamp

        if level == "crit":
            if self._level_streak >= self.config.crit_streak and self._alert_level != "crit":
                msg = "Resource monitor: critical thresholds exceeded"
                if reasons:
                    msg += "; " + ", ".join(reasons)
                self.log.error(msg)
                await self._safe_alert("crit", msg, ctx)
                self._alert_level = "crit"
                if self.config.exit_on_crit:
                    self.log.error("Exiting process due to critical resource exhaustion")
                    await asyncio.sleep(0)
                    os._exit(90)
        elif level == "warn":
            if self._level_streak >= self.config.warn_streak and self._alert_level != "warn":
                msg = "Resource monitor: warning thresholds reached"
                if reasons:
                    msg += "; " + ", ".join(reasons)
                self.log.warning(msg)
                await self._safe_alert("warn", msg, ctx)
                self._alert_level = "warn"
        else:  # ok
            if self._alert_level in {"warn", "crit"} and self._level_streak >= self.config.recovery_streak:
                msg = "Resource monitor: metrics back to normal"
                self.log.info(msg)
                await self._safe_alert("info", msg, ctx)
                self._alert_level = "ok"

    async def _maybe_emit_info(self, snap: ResourceSnapshot) -> None:
        if self.config.info_interval_sec <= 0:
            return
        now = time.time()
        if now - self._last_info_ts < self.config.info_interval_sec:
            return
        self._last_info_ts = now
        ctx = {
            "snapshot": snap.to_dict(),
            "config": self.config.to_public_dict(),
        }
        self.log.debug("Resource monitor heartbeat: %s", ctx)
        await self._safe_alert("info", "Resource monitor heartbeat", ctx)

    async def run(self) -> None:
        if not self.config.enabled:
            self.log.info("Resource monitor disabled via configuration")
            return

        interval = max(5.0, float(self.config.interval_sec))
        self.log.info(
            "Resource monitor started: interval=%.1fs, disk_path=%s", interval, self.config.disk_path
        )

        while not self._stop:
            t0 = time.perf_counter()
            try:
                snap = self.snapshot()
                level, reasons = self._evaluate(snap)
                if level != "ok":
                    self.log.debug("Resource level=%s reasons=%s", level, reasons)
                await self._handle_level(level, reasons, snap)
                if self._alert_level == "ok":
                    await self._maybe_emit_info(snap)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.log.exception("Resource monitor iteration failed: %r", exc)
            dt = time.perf_counter() - t0
            sleep_for = max(1.0, interval - dt)
            try:
                await asyncio.sleep(sleep_for)
            except asyncio.CancelledError:
                raise

        self.log.info("Resource monitor stopped")

    def status(self) -> Dict[str, Any]:
        """Возвращает агрегированную информацию о состоянии монитора."""

        snapshot_dict = self._last_snapshot.to_dict() if self._last_snapshot else None
        return {
            "running": not self._stop,
            "level": self._level_value,
            "alert_level": self._alert_level,
            "streak": self._level_streak,
            "reasons": list(self._last_reasons),
            "last_snapshot_ts": self._last_snapshot_ts,
            "snapshot": snapshot_dict,
            "config": self.config.to_public_dict(),
        }
