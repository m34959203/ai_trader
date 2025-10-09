"""Mathematical helpers for model sizing and risk control."""
from __future__ import annotations


def clamp01(value: float) -> float:
    """Clamp a floating point value to the [0, 1] interval."""

    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return float(value)
