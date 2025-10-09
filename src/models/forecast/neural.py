"""Adapters for NeuralForecast/Darts style models (skeleton implementation)."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

LOG = logging.getLogger("ai_trader.models.forecast.neural")


class NeuralForecastAdapter:
    """Wrapper that defers heavy imports until fit/predict are called."""

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self._params = params or {}
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from neuralforecast import NeuralForecast  # type: ignore

            LOG.info("Initialising NeuralForecast placeholder with params: %s", self._params)
            self._model = NeuralForecast(models=[], freq=self._params.get("freq", "1H"))
        except Exception as exc:  # pragma: no cover - optional dependency
            LOG.warning("NeuralForecast not available (%s); predictions will be empty", exc)
            self._model = None

    def predict(self, series) -> Dict[str, Any]:  # noqa: ANN001 - generic signature for placeholder
        self._ensure_model()
        if self._model is None:
            return {"forecast": [], "provider": "placeholder"}
        LOG.info("NeuralForecast placeholder invoked; integrate training pipeline to use real forecasts")
        return {"forecast": [], "provider": "neuralforecast"}


class DartsAdapter:
    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self._params = params or {}
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from darts.models import TFTModel  # type: ignore

            LOG.info("Initialising Darts TFT placeholder with params: %s", self._params)
            self._model = TFTModel(input_chunk_length=24, output_chunk_length=12)
        except Exception as exc:  # pragma: no cover
            LOG.warning("Darts not available (%s); predictions will be empty", exc)
            self._model = None

    def predict(self, series) -> Dict[str, Any]:  # noqa: ANN001
        self._ensure_model()
        if self._model is None:
            return {"forecast": [], "provider": "placeholder"}
        LOG.info("Darts placeholder invoked; integrate time-series pipeline for production use")
        return {"forecast": [], "provider": "darts"}
