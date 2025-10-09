"""FinBERT-backed sentiment estimator with rule-based fallback."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

from ..base import ISentimentModel, SentimentOutput
from ...utils_math import clamp01

LOG = logging.getLogger("ai_trader.models.sentiment.finbert")


@dataclass
class FinBERTParams:
    model_name: str = "ProsusAI/finbert"
    device: int = -1
    use_pipeline: bool = True


class FinBERTLite(ISentimentModel):
    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        raw = params or {}
        self._params = FinBERTParams(
            model_name=str(raw.get("model_name", FinBERTParams.model_name)),
            device=int(raw.get("device", FinBERTParams.device)),
            use_pipeline=bool(raw.get("use_pipeline", FinBERTParams.use_pipeline)),
        )
        self._pipeline = self._load_pipeline() if self._params.use_pipeline else None
        self._fallback_lexicon = {
            "bull": 1.0,
            "beat": 0.8,
            "growth": 0.7,
            "record": 0.6,
            "profit": 0.6,
            "buyback": 0.6,
            "upgrade": 0.5,
            "positive": 0.4,
            "strong": 0.3,
            "bear": -1.0,
            "miss": -0.8,
            "loss": -0.7,
            "downgrade": -0.6,
            "negative": -0.5,
            "weak": -0.4,
            "fraud": -0.9,
        }

    def _load_pipeline(self):  # pragma: no cover - heavy dependency guard
        try:
            from transformers import pipeline  # type: ignore

            LOG.info("Loading FinBERT pipeline: %s", self._params.model_name)
            return pipeline(
                "sentiment-analysis",
                model=self._params.model_name,
                tokenizer=self._params.model_name,
                device=self._params.device,
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("FinBERT pipeline unavailable (%s). Falling back to rules.", exc)
            return None

    def analyze(self, text: str) -> SentimentOutput:
        text = (text or "").strip()
        if not text:
            return SentimentOutput(label="neu", score=0.0, confidence=0.0, reasons={"empty": True})

        if self._pipeline is not None:
            try:
                result = self._pipeline(text)[0]
                label_raw = str(result["label"]).lower()
                score = float(result.get("score", 0.0))
            except Exception as exc:  # pragma: no cover - runtime guard
                LOG.warning("FinBERT inference failed (%s). Using fallback.", exc)
                return self._fallback(text)

            label, signed_score = self._map_label(label_raw, score)
            return SentimentOutput(
                label=label,
                score=signed_score,
                confidence=clamp01(score),
                reasons={"provider": "finbert", "raw_label": label_raw, "raw_score": score},
            )

        return self._fallback(text)

    def _fallback(self, text: str) -> SentimentOutput:
        tokens = [token.strip(".,!?:;()[]{}\"'`").lower() for token in text.split() if token]
        total = 0.0
        hits = 0
        for token in tokens:
            if token in self._fallback_lexicon:
                hits += 1
                total += self._fallback_lexicon[token]
        if hits == 0:
            return SentimentOutput(label="neu", score=0.0, confidence=0.1, reasons={"provider": "rules"})

        avg_score = max(-1.0, min(1.0, total / hits))
        label = "pos" if avg_score > 0.1 else "neg" if avg_score < -0.1 else "neu"
        confidence = clamp01(abs(avg_score))
        return SentimentOutput(
            label=label,
            score=avg_score,
            confidence=confidence,
            reasons={"provider": "rules", "hits": hits, "tokens": tokens[:10]},
        )

    @staticmethod
    def _map_label(label_raw: str, score: float) -> tuple[str, float]:
        if "neg" in label_raw:
            return "neg", -abs(score)
        if "pos" in label_raw:
            return "pos", abs(score)
        return "neu", 0.0
