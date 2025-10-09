"""Centralised access point for AI models used by the trading service."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency is part of base stack
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from src.models import create_model
from src.models.base import MarketFeatures, ModelConfig, RegimeOutput, SentimentOutput, SignalOutput

from utils.structured_logging import get_logger


LOG = get_logger("ai_trader.model_router")


DEFAULT_MODEL_CONFIG = {
    "models": {
        "signal": "signal:rf_rule",
        "sentiment": "sentiment:finbert",
        "regime": "regime:kmeans",
    },
    "risk": {
        "per_trade_cap": 0.02,
        "daily_loss_limit": 0.06,
    },
}


class ModelRouter:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config: Dict[str, Any] = {}
        self._signal_model = None
        self._sentiment_model = None
        self._regime_model = None
        self._risk: Dict[str, Any] = {}
        if config:
            self.configure(config)

    @property
    def risk_config(self) -> Dict[str, Any]:
        return dict(self._risk)

    def configure(self, config: Dict[str, Any]) -> None:
        config = config or {}
        models_cfg = {
            **DEFAULT_MODEL_CONFIG.get("models", {}),
            **(config.get("models") or {}),
        }
        risk_cfg = {
            **DEFAULT_MODEL_CONFIG.get("risk", {}),
            **(config.get("risk") or {}),
        }
        merged = {"models": models_cfg, "risk": risk_cfg}

        signal_cfg = self._parse_model_config(models_cfg, "signal")
        sentiment_cfg = self._parse_model_config(models_cfg, "sentiment")
        regime_cfg = self._parse_model_config(models_cfg, "regime")

        self._signal_model = create_model(signal_cfg.name, signal_cfg.params)
        self._sentiment_model = create_model(sentiment_cfg.name, sentiment_cfg.params)
        self._regime_model = create_model(regime_cfg.name, regime_cfg.params)

        self._risk = {
            "per_trade_cap": float(risk_cfg.get("per_trade_cap", DEFAULT_MODEL_CONFIG["risk"]["per_trade_cap"])),
            "daily_loss_limit": float(risk_cfg.get("daily_loss_limit", DEFAULT_MODEL_CONFIG["risk"]["daily_loss_limit"])),
        }
        self._config = merged
        LOG.info(
            "Model router configured (signal=%s, sentiment=%s, regime=%s)",
            signal_cfg.name,
            sentiment_cfg.name,
            regime_cfg.name,
        )

    def signal(self, features: MarketFeatures) -> SignalOutput:
        if self._signal_model is None:
            raise RuntimeError("Signal model is not initialised")
        return self._signal_model.predict(features)

    def sentiment(self, text: str) -> SentimentOutput:
        if self._sentiment_model is None:
            raise RuntimeError("Sentiment model is not initialised")
        return self._sentiment_model.analyze(text)

    def regime(self, features: MarketFeatures) -> RegimeOutput:
        if self._regime_model is None:
            raise RuntimeError("Regime model is not initialised")
        return self._regime_model.classify(features)

    def export_config(self) -> Dict[str, Any]:
        return dict(self._config)

    @staticmethod
    def load_from_file(path: Path) -> Dict[str, Any]:
        if not path.exists():
            return DEFAULT_MODEL_CONFIG
        if yaml is None:
            raise RuntimeError("PyYAML is required to load execution config files")
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return data

    @staticmethod
    def _parse_model_config(models_cfg: Dict[str, Any], key: str) -> ModelConfig:
        raw = models_cfg.get(key) or DEFAULT_MODEL_CONFIG["models"][key]
        if isinstance(raw, str):
            name, params = raw, {}
        elif isinstance(raw, dict):
            name = raw.get("name") or raw.get("model") or DEFAULT_MODEL_CONFIG["models"][key]
            params = raw.get("params") or {k: v for k, v in raw.items() if k not in {"name", "model"}}
        else:
            raise TypeError(f"Unexpected model config format for '{key}': {raw!r}")
        return ModelConfig(name=name, params=params)


router_singleton = ModelRouter()
