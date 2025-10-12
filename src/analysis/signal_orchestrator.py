from __future__ import annotations

"""Signal orchestration utilities for Stage 2 market intelligence."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .analyze_market import AnalysisConfig, DEFAULT_CONFIG, analyze_market
from ..strategy import StrategyEnsembleConfig, run_configured_ensemble

__all__ = ["MultiStrategyOrchestrator", "OrchestratedSignal"]


@dataclass(frozen=True)
class OrchestratedSignal:
    """Final consensus signal produced by the orchestrator."""

    signal: str
    confidence: int
    reasons: List[str]
    sources: Dict[str, object]


class MultiStrategyOrchestrator:
    """Blend rule-based market analysis with configurable strategy ensembles.

    The orchestrator is responsible for Stage 2 signal governance:
      • use :func:`analyze_market` for enriched technical/news reasoning
      • execute a :class:`StrategyEnsembleConfig` and inspect per-strategy votes
      • combine the two streams into a single actionable decision
    """

    def __init__(
        self,
        strategy_config: StrategyEnsembleConfig,
        *,
        analysis_config: AnalysisConfig = DEFAULT_CONFIG,
        history_limit: int = 30,
    ) -> None:
        self._strategy_config = strategy_config
        self._analysis_config = analysis_config
        self._history_limit = max(5, int(history_limit))

    @staticmethod
    def _prepare_strategy_frame(df: pd.DataFrame) -> pd.DataFrame:
        required = {"open", "high", "low", "close"}
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(
                "Strategy input is missing required columns: " + ", ".join(sorted(missing))
            )

        out = df.copy()
        if isinstance(out.index, pd.DatetimeIndex):
            ts = (out.index.view("int64") // 1_000_000_000).astype(np.int64)
            out = out.reset_index(drop=True)
            out["timestamp"] = ts
        return out

    @staticmethod
    def _ensemble_payload(frame: pd.DataFrame, limit: int) -> Dict[str, object]:
        if frame.empty:
            return {"latest": None, "recent": []}

        recent = frame.tail(limit)
        recent_payload = [
            {
                "ts": int(row.ts),
                "signal": int(row.signal),
                "score": float(row.score),
            }
            for row in recent.itertuples()
        ]

        latest = recent.iloc[-1]
        latest_payload = {
            "ts": int(latest.ts),
            "signal": int(latest.signal),
            "score": float(latest.score),
            "details": _normalise_details(latest.details) if "details" in latest else [],
        }
        return {"latest": latest_payload, "recent": recent_payload}

    @staticmethod
    def _blend_signals(
        analysis_signal: str,
        analysis_confidence: int,
        analysis_reasons: Sequence[str],
        ensemble_signal: Optional[int],
        ensemble_score: Optional[float],
    ) -> OrchestratedSignal:
        analysis_signal = (analysis_signal or "flat").lower()
        analysis_conf = int(max(0, min(100, analysis_confidence or 0)))
        analysis_dir = {"buy": 1, "sell": -1}.get(analysis_signal, 0)

        ensemble_dir = int(np.sign(ensemble_signal)) if ensemble_signal else 0
        ensemble_strength = float(ensemble_score or 0.0)
        ensemble_conf = int(round(min(1.0, abs(ensemble_strength)) * 100))

        reasons: List[str] = []

        if ensemble_dir == 0 and analysis_dir == 0:
            final_signal = "flat"
            final_conf = max(analysis_conf, ensemble_conf)
            reasons.append("Ансамбль и анализ нейтральны — удерживаемся вне рынка.")
        elif ensemble_dir == 0:
            final_signal = "buy" if analysis_dir > 0 else "sell" if analysis_dir < 0 else "flat"
            final_conf = analysis_conf
            reasons.append("Ансамбль стратегий нейтрален, решение определяется анализом рынка.")
        elif analysis_dir == 0:
            final_signal = "buy" if ensemble_dir > 0 else "sell"
            final_conf = max(ensemble_conf, analysis_conf // 2)
            reasons.append(
                f"Ансамбль стратегий даёт {'LONG' if ensemble_dir > 0 else 'SHORT'} сигнал"
                f" (score={ensemble_strength:.2f})."
            )
        elif ensemble_dir == analysis_dir:
            final_signal = "buy" if analysis_dir > 0 else "sell"
            final_conf = min(100, int(round(0.6 * analysis_conf + 0.4 * ensemble_conf)))
            reasons.append("Анализ рынка и ансамбль стратегий согласованы по направлению.")
        else:
            final_signal = "flat"
            final_conf = min(analysis_conf, ensemble_conf)
            reasons.append("Конфликт анализа и ансамбля — сигнал подавлен до FLAT.")

        # Добавляем краткие источники для прозрачности
        if analysis_dir != 0 and analysis_reasons:
            reasons.append("Анализ: " + analysis_reasons[0])
        if ensemble_dir != 0:
            direction = "LONG" if ensemble_dir > 0 else "SHORT"
            reasons.append(f"Ансамбль: направление {direction}, score={ensemble_strength:.2f}")

        sources = {
            "analysis": {"signal": analysis_signal, "confidence": analysis_conf},
            "ensemble": {"signal": ensemble_dir, "score": ensemble_strength, "confidence": ensemble_conf},
        }
        return OrchestratedSignal(final_signal, final_conf, reasons, sources)

    def evaluate(
        self,
        df_fast: pd.DataFrame,
        df_slow: Optional[pd.DataFrame] = None,
        *,
        symbol: Optional[str] = None,
    ) -> Dict[str, object]:
        analysis = analyze_market(
            df_fast,
            df_4h=df_slow,
            symbol=symbol,
            config=self._analysis_config,
        )

        try:
            strategy_df = self._prepare_strategy_frame(df_fast)
            ensemble_frame = run_configured_ensemble(strategy_df, self._strategy_config)
        except Exception as exc:
            ensemble_payload = {"latest": None, "recent": [], "error": str(exc)}
            orchestrated = self._blend_signals(
                analysis.get("signal", "flat"),
                int(analysis.get("confidence", analysis.get("signal_score", 0) or 0)),
                analysis.get("reasons", []),
                None,
                None,
            )
        else:
            ensemble_payload = self._ensemble_payload(ensemble_frame, self._history_limit)
            latest = ensemble_payload["latest"] or {}
            orchestrated = self._blend_signals(
                analysis.get("signal", "flat"),
                int(analysis.get("confidence", analysis.get("signal_score", 0) or 0)),
                analysis.get("reasons", []),
                latest.get("signal"),
                latest.get("score"),
            )

        return {
            "analysis": analysis,
            "ensemble": ensemble_payload,
            "final": orchestrated,
        }


def _normalise_details(details: object) -> List[Dict[str, object]]:
    if isinstance(details, list):
        normalized: List[Dict[str, object]] = []
        for item in details:
            if isinstance(item, dict):
                normalized.append(
                    {
                        "name": str(item.get("name", "")),
                        "kind": str(item.get("kind", "")),
                        "signal": int(item.get("signal", 0)),
                        "weight": float(item.get("weight", 0.0)),
                    }
                )
        return normalized
    return []
