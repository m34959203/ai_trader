"""Centralised access point for AI models used by the trading service."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional dependency is part of base stack
    import yaml
except Exception:  # pragma: no cover
    yaml = None

from src.models import create_model
from src.models.base import MarketFeatures, ModelConfig, RegimeOutput, SentimentOutput, SignalOutput
from src.models.drift.adwin import AdaptiveDriftDetector, DriftResult

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
    "drift": {
        "report_dir": "state/drift_reports",
        "signal": {"delta": 0.002},
        "sentiment": {"delta": 0.003},
        "regime": False,
    },
}


class ModelRouter:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self._config: Dict[str, Any] = {}
        self._signal_model = None
        self._sentiment_model = None
        self._regime_model = None
        self._risk: Dict[str, Any] = {}
        self._drift_cfg: Dict[str, Any] = {}
        self._drift_detectors: Dict[str, AdaptiveDriftDetector] = {}
        self._drift_report_dir: Optional[Path] = None
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
        drift_cfg = {
            **DEFAULT_MODEL_CONFIG.get("drift", {}),
            **(config.get("drift") or {}),
        }
        merged = {"models": models_cfg, "risk": risk_cfg, "drift": drift_cfg}

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
        self._configure_drift(drift_cfg)
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
        output = self._signal_model.predict(features)
        self._process_drift(
            "signal",
            self._extract_signal_metric(output),
            {"features": dict(features), "output": dict(output)},
        )
        return output

    def sentiment(self, text: str) -> SentimentOutput:
        if self._sentiment_model is None:
            raise RuntimeError("Sentiment model is not initialised")
        output = self._sentiment_model.analyze(text)
        self._process_drift(
            "sentiment",
            float(output.get("score", 0.0)),
            {"text": text[:256], "output": dict(output)},
        )
        return output

    def regime(self, features: MarketFeatures) -> RegimeOutput:
        if self._regime_model is None:
            raise RuntimeError("Regime model is not initialised")
        output = self._regime_model.classify(features)
        self._process_drift(
            "regime",
            float(output.get("score", 0.0)),
            {"features": dict(features), "output": dict(output)},
        )
        return output

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

    # ------------------------------------------------------------------
    # Drift detection helpers
    # ------------------------------------------------------------------
    def _configure_drift(self, config: Dict[str, Any]) -> None:
        report_dir_raw = config.get("report_dir") or DEFAULT_MODEL_CONFIG["drift"].get("report_dir")
        try:
            report_dir = Path(report_dir_raw).expanduser()
            report_dir.mkdir(parents=True, exist_ok=True)
        except Exception:  # pragma: no cover - filesystem guard
            LOG.warning("Failed to prepare drift report directory '%s'", report_dir_raw)
            report_dir = None

        self._drift_report_dir = report_dir
        self._drift_cfg = dict(config)
        self._drift_detectors = {}

        for target in ("signal", "sentiment", "regime"):
            cfg = config.get(target, DEFAULT_MODEL_CONFIG["drift"].get(target))
            if not cfg:
                continue
            delta = float(cfg.get("delta", 0.002)) if isinstance(cfg, dict) else 0.002
            self._drift_detectors[target] = AdaptiveDriftDetector(delta=delta)
            LOG.info("Drift detector enabled for %s (delta=%s)", target, delta)

    def _process_drift(self, target: str, value: float, context: Dict[str, Any]) -> None:
        detector = self._drift_detectors.get(target)
        if detector is None:
            return
        result = detector.update(value)
        if result.drift or result.warning:
            self._store_drift_report(target, result, context)

    def _store_drift_report(self, target: str, result: DriftResult, context: Dict[str, Any]) -> None:
        payload = {
            "target": target,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
            "drift": result.drift,
            "warning": result.warning,
            "details": result.details,
            "context": context,
        }
        LOG.warning("Drift event detected for %s: %s", target, result.details)
        if not self._drift_report_dir:
            return
        filename = f"{target}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}.json"
        try:
            with (self._drift_report_dir / filename).open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except Exception:  # pragma: no cover - IO guard
            LOG.exception("Failed to persist drift report for %s", target)

    @staticmethod
    def _extract_signal_metric(output: SignalOutput) -> float:
        signal = output.get("signal")
        confidence = float(output.get("confidence", 0.0) or 0.0)
        mapping = {"buy": 1.0, "sell": -1.0, "hold": 0.0}
        return mapping.get(str(signal).lower(), 0.0) * confidence


router_singleton = ModelRouter()
