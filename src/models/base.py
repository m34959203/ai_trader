"""Core typing contracts for AI trading models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol, TypedDict, Literal, Any


class MarketFeatures(TypedDict, total=False):
    """Feature bundle shared across signal/regime/sizing models."""

    symbol: str
    price: float
    rsi: float
    macd: float
    macd_signal: float
    atr: float
    volume: float
    trend_ma_fast: float
    trend_ma_slow: float
    volatility: float
    equity: float
    day_pnl: float
    news_score: float


class SignalOutput(TypedDict):
    signal: Literal["buy", "sell", "hold"]
    confidence: float
    reasons: Dict[str, Any]


class SentimentOutput(TypedDict):
    label: Literal["pos", "neu", "neg"]
    score: float
    confidence: float
    reasons: Dict[str, Any]


class RegimeOutput(TypedDict):
    regime: Literal["calm", "storm"]
    score: float
    reasons: Dict[str, Any]


class ISignalModel(Protocol):
    """Trading signal generator interface."""

    def predict(self, features: MarketFeatures) -> SignalOutput:
        ...


class ISentimentModel(Protocol):
    """News/alternate data sentiment estimator interface."""

    def analyze(self, text: str) -> SentimentOutput:
        ...


class IRegimeModel(Protocol):
    """Market regime classifier interface."""

    def classify(self, features: MarketFeatures) -> RegimeOutput:
        ...


@dataclass
class ModelConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionConfig:
    signal: ModelConfig
    sentiment: ModelConfig
    regime: ModelConfig
    risk: Dict[str, Any] = field(default_factory=dict)
