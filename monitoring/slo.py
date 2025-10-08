"""Service level objective tracking utilities."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable


@dataclass(slots=True)
class SLO:
    name: str
    target: float  # between 0 and 1
    window: int


class SLOTracker:
    def __init__(self, slos: Iterable[SLO]):
        self._slos = {slo.name: slo for slo in slos}
        self._history: Dict[str, Deque[int]] = {name: deque(maxlen=slo.window) for name, slo in self._slos.items()}

    def record(self, name: str, success: bool) -> None:
        if name not in self._history:
            raise KeyError(f"Unknown SLO {name}")
        self._history[name].append(1 if success else 0)

    def compliance(self, name: str) -> float:
        if name not in self._history:
            raise KeyError(f"Unknown SLO {name}")
        hist = self._history[name]
        if not hist:
            return 1.0
        return sum(hist) / len(hist)

    def report(self) -> Dict[str, Dict[str, float]]:
        report: Dict[str, Dict[str, float]] = {}
        for name, slo in self._slos.items():
            compliance = self.compliance(name)
            report[name] = {"target": slo.target, "actual": compliance, "met": compliance >= slo.target}
        return report

