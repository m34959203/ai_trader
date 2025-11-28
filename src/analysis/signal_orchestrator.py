from __future__ import annotations

"""Signal orchestration utilities for Stage 2 market intelligence."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .analyze_market import AnalysisConfig, DEFAULT_CONFIG, analyze_market
from ..strategy import StrategyEnsembleConfig, run_configured_ensemble

# ML Integration
try:
    from .lstm_integration import LSTMSignalGenerator, integrate_lstm_with_technical
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

try:
    from ..models.meta_learner import MetaLearner, extract_meta_features
    META_LEARNER_AVAILABLE = True
except ImportError:
    META_LEARNER_AVAILABLE = False

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
      • ENHANCED: ML integration (LSTM + Meta-Learner)
    """

    def __init__(
        self,
        strategy_config: StrategyEnsembleConfig,
        *,
        analysis_config: AnalysisConfig = DEFAULT_CONFIG,
        history_limit: int = 30,
        lstm_model_path: Optional[str] = None,
        meta_learner_path: Optional[str] = None,
        enable_lstm: bool = True,
        enable_meta_learner: bool = True,
        lstm_weight: float = 0.3,  # 30% LSTM, 70% technical
    ) -> None:
        self._strategy_config = strategy_config
        self._analysis_config = analysis_config
        self._history_limit = max(5, int(history_limit))

        # LSTM Integration
        self.lstm_generator = None
        self.enable_lstm = enable_lstm and LSTM_AVAILABLE
        if self.enable_lstm and lstm_model_path:
            try:
                self.lstm_generator = LSTMSignalGenerator(
                    model_path=lstm_model_path,
                    min_confidence=0.55,
                    min_move_pct=0.005,
                )
            except Exception as e:
                print(f"Warning: Failed to load LSTM model: {e}")
                self.enable_lstm = False

        # Meta-Learner Integration
        self.meta_learner = None
        self.enable_meta_learner = enable_meta_learner and META_LEARNER_AVAILABLE
        if self.enable_meta_learner and meta_learner_path:
            try:
                self.meta_learner = MetaLearner()
                self.meta_learner.load(meta_learner_path)
            except Exception as e:
                print(f"Warning: Failed to load Meta-Learner: {e}")
                self.enable_meta_learner = False

        self.lstm_weight = lstm_weight

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

        # ML Enhancement: LSTM Integration
        lstm_result = None
        if self.enable_lstm and self.lstm_generator:
            try:
                lstm_signal = self.lstm_generator.generate_signal(df_fast)
                lstm_result = {
                    "direction": lstm_signal.direction,
                    "confidence": lstm_signal.confidence,
                    "predicted_move": lstm_signal.predicted_move,
                    "signal": lstm_signal.signal,
                }

                # Integrate LSTM with technical signal
                tech_signal = orchestrated.signal
                tech_conf = orchestrated.confidence / 100.0  # Convert to 0-1

                combined = integrate_lstm_with_technical(
                    tech_signal=tech_signal,
                    tech_confidence=tech_conf,
                    lstm_signal=lstm_signal,
                    lstm_weight=self.lstm_weight,
                )

                # Create enhanced orchestrated signal
                enhanced_reasons = list(orchestrated.reasons)
                enhanced_reasons.append(combined["reason"])

                orchestrated = OrchestratedSignal(
                    signal=combined["final_signal"],
                    confidence=int(combined["final_confidence"] * 100),
                    reasons=enhanced_reasons,
                    sources={
                        **orchestrated.sources,
                        "lstm": lstm_result,
                        "ml_combined": {
                            "method": "weighted_ensemble",
                            "lstm_weight": self.lstm_weight,
                            "tech_weight": 1 - self.lstm_weight,
                        },
                    },
                )
            except Exception as e:
                print(f"Warning: LSTM integration failed: {e}")

        # ML Enhancement: Meta-Learner Filtering
        meta_result = None
        if self.enable_meta_learner and self.meta_learner and self.meta_learner.is_trained:
            try:
                # Convert signal to direction
                signal_direction = {"buy": 1, "sell": -1, "flat": 0}.get(
                    orchestrated.signal.lower(), 0
                )

                # Extract meta-features from current market state
                meta_features = extract_meta_features(
                    df=df_fast,
                    signal_direction=signal_direction,
                    signal_confidence=orchestrated.confidence / 100.0,
                    signal_source="orchestrator",
                )

                # Get meta-learner prediction
                meta_prediction = self.meta_learner.predict(
                    features=meta_features,
                    original_signal=signal_direction,
                )

                meta_result = meta_prediction.to_dict()

                # Apply meta-learner filtering
                if not meta_prediction.should_take:
                    # Meta-learner recommends skipping this signal
                    filtered_reasons = list(orchestrated.reasons)
                    filtered_reasons.append(
                        f"🚫 Meta-Learner filtered signal: {meta_prediction.reason}"
                    )

                    orchestrated = OrchestratedSignal(
                        signal="flat",
                        confidence=int(meta_prediction.confidence * 100),
                        reasons=filtered_reasons,
                        sources={
                            **orchestrated.sources,
                            "meta_learner": meta_result,
                            "original_signal_before_filter": orchestrated.signal,
                        },
                    )
                else:
                    # Meta-learner approves - boost confidence
                    boosted_conf = int(
                        (orchestrated.confidence / 100.0 * 0.7 + meta_prediction.confidence * 0.3) * 100
                    )
                    approved_reasons = list(orchestrated.reasons)
                    approved_reasons.append(
                        f"✓ Meta-Learner approved ({meta_prediction.probability_profitable:.0%} prob): {meta_prediction.reason}"
                    )

                    orchestrated = OrchestratedSignal(
                        signal=orchestrated.signal,
                        confidence=min(100, boosted_conf),
                        reasons=approved_reasons,
                        sources={
                            **orchestrated.sources,
                            "meta_learner": meta_result,
                        },
                    )
            except Exception as e:
                print(f"Warning: Meta-learner integration failed: {e}")

        return {
            "analysis": analysis,
            "ensemble": ensemble_payload,
            "final": orchestrated,
            "ml": {
                "lstm": lstm_result,
                "meta_learner": meta_result,
            },
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
