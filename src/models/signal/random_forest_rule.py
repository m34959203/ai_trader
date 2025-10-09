"""Heuristic random-forest inspired ensemble of technical rules."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

from ..base import ISignalModel, MarketFeatures, SignalOutput
from ...utils_math import clamp01

LOG = logging.getLogger("ai_trader.models.signal.rf_rule")


@dataclass
class RandomForestRuleParams:
    ma_margin_pct: float = 0.001
    rsi_buy: float = 35.0
    rsi_sell: float = 65.0
    atr_risk_floor: float = 0.0
    atr_risk_ceiling: float = 0.05


class RandomForestRuleSignalModel(ISignalModel):
    """Small rule-based ensemble standing in for a RF classifier."""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        raw = params or {}
        self._params = RandomForestRuleParams(
            ma_margin_pct=float(raw.get("ma_margin_pct", RandomForestRuleParams.ma_margin_pct)),
            rsi_buy=float(raw.get("rsi_buy", RandomForestRuleParams.rsi_buy)),
            rsi_sell=float(raw.get("rsi_sell", RandomForestRuleParams.rsi_sell)),
            atr_risk_floor=float(raw.get("atr_risk_floor", RandomForestRuleParams.atr_risk_floor)),
            atr_risk_ceiling=float(raw.get("atr_risk_ceiling", RandomForestRuleParams.atr_risk_ceiling)),
        )

    # pylint: disable=too-many-branches
    def predict(self, features: MarketFeatures) -> SignalOutput:
        votes: Dict[str, float] = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        reasons: Dict[str, Any] = {}

        fast = float(features.get("trend_ma_fast", 0.0) or 0.0)
        slow = float(features.get("trend_ma_slow", 0.0) or 0.0)
        price = float(features.get("price", 0.0) or 0.0)
        rsi = float(features.get("rsi", 50.0) or 50.0)
        atr = float(features.get("atr", 0.0) or 0.0)
        volatility = float(features.get("volatility", features.get("vol", 0.0)) or 0.0)

        margin = abs(price) * self._params.ma_margin_pct
        if fast and slow:
            if fast > slow + margin:
                votes["buy"] += 1.0
                reasons["ma_cross"] = "fast_above_slow"
            elif fast < slow - margin:
                votes["sell"] += 1.0
                reasons["ma_cross"] = "fast_below_slow"
            else:
                votes["hold"] += 0.5
                reasons["ma_cross"] = "flat"

        if rsi <= self._params.rsi_buy:
            votes["buy"] += 1.0
            reasons["rsi"] = f"oversold@{rsi:.1f}"
        elif rsi >= self._params.rsi_sell:
            votes["sell"] += 1.0
            reasons["rsi"] = f"overbought@{rsi:.1f}"
        else:
            votes["hold"] += 0.5
            reasons["rsi"] = f"neutral@{rsi:.1f}"

        macd = float(features.get("macd", 0.0) or 0.0)
        macd_signal = float(features.get("macd_signal", 0.0) or 0.0)
        if macd and macd_signal:
            if macd > macd_signal:
                votes["buy"] += 0.75
                reasons["macd"] = "bullish"
            elif macd < macd_signal:
                votes["sell"] += 0.75
                reasons["macd"] = "bearish"
            else:
                votes["hold"] += 0.25
                reasons["macd"] = "flat"

        atr_floor = max(1e-9, self._params.atr_risk_floor)
        atr_ceiling = max(atr_floor, self._params.atr_risk_ceiling)
        atr_norm = clamp01((atr - atr_floor) / (atr_ceiling - atr_floor) if atr_ceiling > atr_floor else 0.0)
        if atr_norm > 0.75 or volatility > atr_ceiling:
            votes["hold"] += 0.5
            reasons["atr_gate"] = f"high@{atr_norm:.2f}"
        else:
            votes["hold"] += 0.1
            reasons.setdefault("atr_gate", f"ok@{atr_norm:.2f}")

        decision, confidence = self._finalise(votes)
        reasons["atr_norm"] = atr_norm
        reasons["volatility"] = volatility

        return SignalOutput(signal=decision, confidence=confidence, reasons=reasons)

    def _finalise(self, votes: Dict[str, float]) -> tuple[str, float]:
        ordered = sorted(votes.items(), key=lambda item: item[1], reverse=True)
        top_signal, top_score = ordered[0]
        total_votes = sum(votes.values()) or 1.0
        confidence = clamp01(top_score / total_votes)
        if top_signal == "hold" and confidence < 0.55:
            return "hold", confidence
        if top_signal == "buy" and votes["sell"] >= top_score:
            return "hold", 0.2
        if top_signal == "sell" and votes["buy"] >= top_score:
            return "hold", 0.2
        return top_signal, confidence
