"""Market regime classifier based on volatility clustering."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from ..base import IRegimeModel, MarketFeatures, RegimeOutput
from ...utils_math import clamp01

LOG = logging.getLogger("ai_trader.models.regime")


@dataclass
class RegimeParams:
    kmeans_random_state: int = 42


class RegimeClassifier(IRegimeModel):
    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        raw = params or {}
        self._params = RegimeParams(
            kmeans_random_state=int(raw.get("kmeans_random_state", RegimeParams.kmeans_random_state))
        )
        self._model = self._try_init_hmm()

    def _try_init_hmm(self):  # pragma: no cover - optional dependency
        try:
            from hmmlearn.hmm import GaussianHMM  # type: ignore

            LOG.info("Initialising GaussianHMM for regime classification")
            model = GaussianHMM(n_components=2, covariance_type="diag", n_iter=50)
            return ("hmm", model)
        except Exception:
            LOG.debug("hmmlearn unavailable, falling back to scikit-learn KMeans")
            try:
                from sklearn.cluster import KMeans

                model = KMeans(n_clusters=2, random_state=self._params.kmeans_random_state, n_init="auto")
                return ("kmeans", model)
            except Exception as exc:  # pragma: no cover
                LOG.warning("Failed to initialise any regime classifier (%s)", exc)
                return None

    def classify(self, features: MarketFeatures) -> RegimeOutput:
        atr = float(features.get("atr", 0.0) or 0.0)
        vol = float(features.get("volatility", features.get("vol", 0.0)) or 0.0)
        data = np.array([[atr, vol]], dtype=float)

        if self._model is None:
            return RegimeOutput(regime="calm", score=0.0, reasons={"provider": "none"})

        provider, model = self._model
        try:
            if provider == "hmm":
                model: Any
                if not hasattr(model, "means_"):
                    # bootstrap with a simple guess around the provided point
                    model.startprob_ = np.array([0.6, 0.4])
                    model.transmat_ = np.array([[0.9, 0.1], [0.2, 0.8]])
                    means = np.array([[atr * 0.8 + 1e-6, vol * 0.8 + 1e-6], [atr * 1.2 + 1e-6, vol * 1.2 + 1e-6]])
                    model.means_ = means
                    model.covars_ = np.stack([np.eye(2) * 0.05, np.eye(2) * 0.1])
                logprob, state_sequence = model.decode(data, algorithm="viterbi")
                state = int(state_sequence[0])
                score = clamp01(float(np.exp(logprob)))
            else:
                model: Any
                labels = model.fit_predict(data)
                state = int(labels[0])
                centres = model.cluster_centers_
                storm_idx = int(np.argmax(centres[:, 0]))
                score = clamp01(float(np.linalg.norm(data[0] - centres.mean(axis=0))))
                state = 1 if state == storm_idx else 0
        except Exception as exc:  # pragma: no cover
            LOG.warning("Regime classification failed: %s", exc)
            return RegimeOutput(regime="calm", score=0.0, reasons={"provider": provider, "error": str(exc)})

        regime = "storm" if state == 1 else "calm"
        return RegimeOutput(regime=regime, score=score, reasons={"provider": provider, "state": state})
