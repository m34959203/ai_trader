"""Lightweight adapter for ADWIN concept drift detection."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

LOG = logging.getLogger("ai_trader.models.drift.adwin")


@dataclass
class DriftResult:
    drift: bool
    warning: bool
    details: Dict[str, Any]


class AdaptiveDriftDetector:
    def __init__(self, delta: float = 0.002) -> None:
        self._delta = float(delta)
        self._detector = self._init_detector()

    def _init_detector(self):  # pragma: no cover - optional dependency wrapper
        try:
            from river.drift import ADWIN  # type: ignore

            LOG.info("Initialising river.ADWIN drift detector (delta=%s)", self._delta)
            return ADWIN(delta=self._delta)
        except Exception as exc:
            LOG.warning("river.ADWIN unavailable (%s); drift detection will be heuristic only", exc)
            return None

    def update(self, value: float) -> DriftResult:
        if self._detector is None:
            return DriftResult(drift=False, warning=False, details={"provider": "fallback"})

        in_drift, in_warning = self._detector.update(float(value))
        return DriftResult(
            drift=bool(in_drift),
            warning=bool(in_warning),
            details={"width": getattr(self._detector, "width", None)},
        )
