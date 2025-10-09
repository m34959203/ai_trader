"""Model registry bootstrap."""
from __future__ import annotations

from .base import ISignalModel, ISentimentModel, IRegimeModel, ExecutionConfig, ModelConfig
from .registry import register_model, create_model, registry_has
from .signal.random_forest_rule import RandomForestRuleSignalModel
from .nlp.finbert_sentiment import FinBERTLite
from .regime.regime_classifier import RegimeClassifier

__all__ = [
    "ISignalModel",
    "ISentimentModel",
    "IRegimeModel",
    "ExecutionConfig",
    "ModelConfig",
    "register_model",
    "create_model",
    "registry_has",
]


def _register_defaults() -> None:
    register_model("signal:rf_rule", lambda params=None: RandomForestRuleSignalModel(params))
    register_model("sentiment:finbert", lambda params=None: FinBERTLite(params))
    register_model("regime:kmeans", lambda params=None: RegimeClassifier(params))


_register_defaults()
