"""Adaptive ML/RL utilities used to augment classical analytics."""
from __future__ import annotations

import asyncio
import dataclasses
import logging
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid

from src.strategy import market_regime_flags

LOG = logging.getLogger("ai_trader.ai.adaptive")


# ---------------------------------------------------------------------------
# Walk-forward optimisation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class WalkForwardConfig:
    """Configuration for the walk-forward trainer."""

    window: int = 400
    step: int = 50
    min_train: int = 120
    horizon: int = 24
    param_grid: Optional[Mapping[str, Sequence[Any]]] = None
    estimator_factory: Callable[[], BaseEstimator] = lambda: LogisticRegression(
        solver="lbfgs", max_iter=250
    )
    scoring: Callable[[np.ndarray, np.ndarray], float] = accuracy_score

    def iter_params(self) -> Iterable[Mapping[str, Any]]:
        if not self.param_grid:
            yield {}
            return
        yield from ParameterGrid(self.param_grid)


@dataclass(slots=True)
class WalkForwardWindow:
    start: int
    end: int
    params: Mapping[str, Any]
    score: float


@dataclass(slots=True)
class WalkForwardResult:
    """Summary of the walk-forward evaluation."""

    windows: List[WalkForwardWindow]
    best_params: Mapping[str, Any]
    best_score: float
    calibrated_at: datetime = field(default_factory=lambda: datetime.now(tz=None))

    def as_dict(self) -> Dict[str, Any]:
        return {
            "best_params": dict(self.best_params),
            "best_score": float(self.best_score),
            "windows": [dataclasses.asdict(w) for w in self.windows],
            "calibrated_at": self.calibrated_at.isoformat(),
        }


class WalkForwardTrainer:
    """Train and evaluate a model on expanding windows."""

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        self._cfg = config or WalkForwardConfig()

    def run(self, X: pd.DataFrame, y: Sequence[int]) -> WalkForwardResult:
        if X.empty:
            raise ValueError("WalkForwardTrainer requires non-empty feature frame")

        cfg = self._cfg
        y_arr = np.asarray(y)
        if y_arr.shape[0] != len(X):
            raise ValueError("Features and labels length mismatch")

        windows: List[WalkForwardWindow] = []
        best_params: Mapping[str, Any] = {}
        best_score = float("-inf")

        ts = np.arange(len(X))
        for params in cfg.iter_params():
            model = clone(cfg.estimator_factory())
            model.set_params(**params)
            start = 0
            scores: List[float] = []
            while start + cfg.min_train < len(ts):
                end = min(start + cfg.window, len(ts))
                train_idx = ts[start:end]
                if len(train_idx) < cfg.min_train:
                    break
                test_end = min(end + cfg.horizon, len(ts))
                test_idx = ts[end:test_end]
                X_train, y_train = X.iloc[train_idx], y_arr[train_idx]
                X_test, y_test = X.iloc[test_idx], y_arr[test_idx]
                if not len(test_idx):
                    break
                try:
                    model.fit(X_train, y_train)
                except Exception as fit_err:
                    LOG.warning("Walk-forward fit failed params=%s: %r", params, fit_err)
                    break
                y_pred = model.predict(X_test)
                score = float(cfg.scoring(y_test, y_pred))
                windows.append(WalkForwardWindow(start=start, end=end, params=params, score=score))
                scores.append(score)
                start += cfg.step
            if scores:
                avg_score = float(np.mean(scores))
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = params

        if not windows:
            raise RuntimeError("Walk-forward training produced no windows")

        return WalkForwardResult(windows=windows, best_params=best_params, best_score=best_score)


# ---------------------------------------------------------------------------
# Reinforcement learning helper (tabular Q-learning)
# ---------------------------------------------------------------------------


class ReinforcementLearner:
    """Minimal tabular Q-learner used for adaptive sizing decisions."""

    def __init__(
        self,
        actions: Sequence[str] = ("scale_down", "keep", "scale_up"),
        *,
        alpha: float = 0.25,
        gamma: float = 0.6,
        epsilon: float = 0.1,
    ):
        self.actions = list(actions)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self._q: Dict[Tuple[str, str], float] = {}

    def _key(self, state: str, action: str) -> Tuple[str, str]:
        return (state, action)

    def policy(self, state: str) -> str:
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        q_values = {action: self._q.get(self._key(state, action), 0.0) for action in self.actions}
        return max(q_values.items(), key=lambda kv: kv[1])[0]

    def update(self, state: str, action: str, reward: float, next_state: str) -> None:
        key = self._key(state, action)
        future = max((self._q.get(self._key(next_state, a), 0.0) for a in self.actions), default=0.0)
        old = self._q.get(key, 0.0)
        new_val = old + self.alpha * (reward + self.gamma * future - old)
        self._q[key] = float(new_val)

    def snapshot(self) -> Dict[str, Any]:
        return {"q": {f"{s}|{a}": v for (s, a), v in self._q.items()}, "actions": list(self.actions)}


# ---------------------------------------------------------------------------
# Adaptive confidence engine
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AdaptiveState:
    last_retrain: Optional[datetime] = None
    walk_forward: Optional[WalkForwardResult] = None
    q_values: Dict[str, Any] = field(default_factory=dict)
    regime_counter: Counter = field(default_factory=Counter)


@dataclass(slots=True)
class AdaptiveSignal:
    confidence: int
    multiplier: float
    source_prob: Optional[float]
    regime: str
    reasons: List[str]


class AdaptiveConfidenceEngine:
    """Blends walk-forward probabilities, RL sizing and regime heuristics."""

    def __init__(
        self,
        *,
        trainer: Optional[WalkForwardTrainer] = None,
        learner: Optional[ReinforcementLearner] = None,
        retrain_interval: timedelta = timedelta(hours=6),
        regime_multipliers: Optional[Mapping[str, float]] = None,
        history_size: int = 240,
    ):
        self._trainer = trainer or WalkForwardTrainer()
        self._learner = learner or ReinforcementLearner()
        self._interval = retrain_interval
        self._state = AdaptiveState()
        self._regime_multipliers = {
            "trend": 1.1,
            "flat": 0.9,
            "turbulent": 0.8,
            "unknown": 0.85,
        }
        if regime_multipliers:
            self._regime_multipliers.update({str(k): float(v) for k, v in regime_multipliers.items()})
        self._history: Deque[Tuple[pd.Series, int]] = deque(maxlen=history_size)
        self._model: Optional[BaseEstimator] = None
        self._lock = asyncio.Lock()

    @staticmethod
    def _extract_features(row: pd.Series) -> pd.Series:
        feats = {
            "ema_fast": row.get("ema_fast", row.get("close", 0.0)),
            "ema_slow": row.get("ema_slow", row.get("close", 0.0)),
            "rsi": row.get("rsi", 50.0),
            "macd": row.get("macd", 0.0),
            "atr": row.get("atr", 0.0),
        }
        feats["spread"] = float(feats["ema_fast"]) - float(feats["ema_slow"])
        return pd.Series(feats)

    def record_example(self, row: pd.Series, label: int) -> None:
        self._history.append((self._extract_features(row), int(label)))

    def _needs_retrain(self) -> bool:
        if self._state.last_retrain is None:
            return True
        return datetime.utcnow() - self._state.last_retrain >= self._interval

    def needs_retrain(self) -> bool:
        """Public accessor indicating whether the engine wants retraining."""
        return self._needs_retrain()

    async def maybe_retrain(self) -> Optional[WalkForwardResult]:
        if not self._needs_retrain():
            return None
        async with self._lock:
            if not self._needs_retrain():
                return None
            if len(self._history) < self._trainer._cfg.min_train:
                LOG.debug("Not enough history for retrain (%d < %d)", len(self._history), self._trainer._cfg.min_train)
                return None
            X = pd.DataFrame([x.to_dict() for x, _ in self._history])
            y = [lbl for _, lbl in self._history]
            wf = self._trainer.run(X, y)
            self._model = clone(self._trainer._cfg.estimator_factory())
            if wf.best_params:
                self._model.set_params(**wf.best_params)
            try:
                self._model.fit(X, y)
            except Exception as fit_err:
                LOG.warning("Adaptive model fit failed: %r", fit_err)
                return None
            self._state.last_retrain = datetime.utcnow()
            self._state.walk_forward = wf
            LOG.info("Adaptive confidence engine retrained: score=%.4f", wf.best_score)
            return wf

    def _regime_for(self, df: pd.DataFrame) -> str:
        try:
            flags = market_regime_flags(df["close"], gap_series=df.get("ema_fast") - df.get("ema_slow"))
        except Exception:
            return "unknown"
        if flags.empty:
            return "unknown"
        last = flags.iloc[-1]
        if bool(last.get("is_turbulent")):
            regime = "turbulent"
        elif bool(last.get("is_trend")):
            regime = "trend"
        elif bool(last.get("is_flat")):
            regime = "flat"
        else:
            regime = "unknown"
        self._state.regime_counter.update([regime])
        return regime

    def _predict_prob(self, row: pd.Series) -> Optional[float]:
        if not self._model:
            return None
        try:
            feats = self._extract_features(row).to_frame().T
            prob = float(self._model.predict_proba(feats)[0][1])
            # clip extreme values to avoid over-confidence
            return max(0.01, min(0.99, prob))
        except Exception as predict_err:
            LOG.debug("Adaptive probability failed: %r", predict_err)
            return None

    def adjust(
        self,
        payload: MutableMapping[str, Any],
        *,
        df: pd.DataFrame,
        label_hint: Optional[int] = None,
    ) -> AdaptiveSignal:
        if "confidence" not in payload:
            payload["confidence"] = 0
        reasons: List[str] = list(payload.get("reasons", []))
        row = df.iloc[-1]
        base_initial = int(payload["confidence"])
        base_conf = base_initial
        regime = self._regime_for(df)
        multiplier = self._regime_multipliers.get(regime, self._regime_multipliers["unknown"])

        prob = self._predict_prob(row)
        if prob is not None:
            reasons.append(f"ML-probability={prob:.2f}")
            calibrated = int(round(prob * 100))
            if label_hint is not None:
                reward = 1.0 if label_hint == int(prob >= 0.5) else -1.0
                self._learner.update(regime, "keep", reward, regime)
            base_conf = int((base_conf + calibrated) / 2)

        sensitivity = 8.0
        adjustment = (multiplier - 1.0) * sensitivity
        adjusted_conf = int(max(0, min(100, round(base_conf + adjustment))))
        payload["confidence"] = adjusted_conf
        payload.setdefault("meta", {})
        payload["meta"]["regime"] = regime
        payload["meta"]["confidence_multiplier"] = multiplier
        payload["meta"]["ml_probability"] = prob
        payload["meta"]["regime_hist"] = dict(self._state.regime_counter)
        payload["meta"]["adaptive_base"] = base_initial
        payload["meta"]["adaptive_adjustment"] = adjustment
        payload["reasons"] = reasons

        return AdaptiveSignal(
            confidence=adjusted_conf,
            multiplier=multiplier,
            source_prob=prob,
            regime=regime,
            reasons=reasons,
        )

    def state(self) -> AdaptiveState:
        snap = dataclasses.replace(self._state)
        if self._state.walk_forward:
            snap.walk_forward = self._state.walk_forward
        snap.q_values = self._learner.snapshot()
        return snap


DEFAULT_ENGINE = AdaptiveConfidenceEngine()

