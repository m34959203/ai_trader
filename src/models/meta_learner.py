"""Meta-learning for signal filtering and quality prediction.

Meta-learning doesn't predict price direction. Instead, it predicts:
"Should I take this technical signal or skip it?"

This is much more reliable than direction prediction because:
- Uses features that are more stable (volatility, trend strength, etc.)
- Binary classification (take/skip) is easier than regression (price)
- Can filter out 40-50% of losing trades while keeping 80-90% of winners

Example:
    Technical indicator says: "BUY"
    Meta-learner checks: volatility high, volume low, news negative
    Meta-learner decides: "Skip this signal - low probability of success"
    Result: Avoided a -2% losing trade
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.structured_logging import get_logger

LOG = get_logger("ai_trader.meta_learner")


@dataclass
class MetaFeatures:
    """Features used by meta-learner to evaluate signal quality."""

    # Signal characteristics
    signal_direction: int  # +1 or -1
    signal_confidence: float  # 0.0 to 1.0
    signal_source: str  # "ema_cross", "rsi_reversion", etc.

    # Market regime
    volatility: float  # ATR as percentage
    trend_strength: float  # ADX or similar
    volume_ratio: float  # Current volume / avg volume

    # Technical context
    rsi_value: float  # 0-100
    price_vs_sma: float  # (price - SMA) / SMA
    distance_from_high: float  # (high - price) / price
    distance_from_low: float  # (price - low) / price

    # Timing features
    hour_of_day: int  # 0-23
    day_of_week: int  # 0-6 (Monday=0)
    bars_since_last_signal: int  # Time since previous signal

    # Sentiment (if available)
    news_sentiment: float = 0.0  # -1 to +1

    def to_array(self) -> np.ndarray:
        """Convert to numpy array for model input."""
        return np.array([
            self.signal_direction,
            self.signal_confidence,
            self.volatility,
            self.trend_strength,
            self.volume_ratio,
            self.rsi_value,
            self.price_vs_sma,
            self.distance_from_high,
            self.distance_from_low,
            self.hour_of_day,
            self.day_of_week,
            self.bars_since_last_signal,
            self.news_sentiment,
        ])

    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names in order."""
        return [
            'signal_direction',
            'signal_confidence',
            'volatility',
            'trend_strength',
            'volume_ratio',
            'rsi_value',
            'price_vs_sma',
            'distance_from_high',
            'distance_from_low',
            'hour_of_day',
            'day_of_week',
            'bars_since_last_signal',
            'news_sentiment',
        ]


@dataclass
class MetaLearnerPrediction:
    """Meta-learner prediction for a signal."""

    should_take: bool  # True if signal should be taken
    confidence: float  # Confidence in prediction (0-1)
    probability_profitable: float  # P(signal will be profitable)
    adjusted_signal: int  # Original signal or 0 (filtered)
    reason: str  # Explanation of decision

    def to_dict(self):
        return {
            'should_take': self.should_take,
            'confidence': self.confidence,
            'probability_profitable': self.probability_profitable,
            'adjusted_signal': self.adjusted_signal,
            'reason': self.reason,
        }


class MetaLearner:
    """Meta-learning model for filtering trading signals.

    Uses XGBoost or Random Forest to predict if a technical signal
    will be profitable based on market context.

    Training:
        - Features: signal characteristics + market regime + context
        - Target: 1 if signal resulted in profit, 0 if loss
        - Model: XGBoost classifier

    Inference:
        - Input: Current signal + market features
        - Output: Probability signal will be profitable
        - Decision: Take signal if probability > threshold (e.g., 0.60)

    Example:
        >>> # Training
        >>> learner = MetaLearner()
        >>> learner.train(historical_signals, outcomes)
        >>>
        >>> # Inference
        >>> features = MetaFeatures(
        ...     signal_direction=1,
        ...     signal_confidence=0.7,
        ...     volatility=0.02,
        ...     ...
        ... )
        >>> prediction = learner.predict(features)
        >>> if prediction.should_take:
        ...     execute_trade()
    """

    def __init__(
        self,
        threshold: float = 0.60,  # Min probability to take signal
        use_xgboost: bool = True,  # Use XGBoost if available, else RandomForest
    ):
        """Initialize meta-learner.

        Args:
            threshold: Minimum probability to take signal
            use_xgboost: Prefer XGBoost over RandomForest
        """
        self.threshold = threshold
        self.model = None
        self.is_trained = False
        self.feature_importance: Optional[Dict[str, float]] = None

        # Try to import ML libraries
        try:
            if use_xgboost:
                from xgboost import XGBClassifier
                self.model = XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    colsample_bytree=0.8,
                    subsample=0.8,
                    random_state=42,
                )
                LOG.info("Using XGBoost for meta-learning")
            else:
                raise ImportError("XGBoost not requested")
        except ImportError:
            # Fallback to sklearn RandomForest
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
            )
            LOG.info("Using RandomForest for meta-learning (XGBoost unavailable)")

    def train(
        self,
        features: List[MetaFeatures],
        outcomes: List[float],  # P&L of each signal
    ) -> Dict[str, float]:
        """Train meta-learner on historical signals.

        Args:
            features: List of signal features
            outcomes: List of P&L for each signal (positive = profit)

        Returns:
            Training metrics
        """
        if len(features) != len(outcomes):
            raise ValueError("Features and outcomes must have same length")

        # Convert to arrays
        X = np.array([f.to_array() for f in features])
        y = (np.array(outcomes) > 0).astype(int)  # 1 if profitable, 0 if not

        LOG.info(f"Training meta-learner on {len(X)} signals")
        LOG.info(f"Profitable signals: {y.sum()} ({y.mean():.1%})")

        # Train model
        self.model.fit(X, y)
        self.is_trained = True

        # Calculate feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_names = MetaFeatures.feature_names()
            self.feature_importance = dict(zip(feature_names, importances))

            # Log top features
            sorted_features = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            LOG.info("Top 5 important features:")
            for name, importance in sorted_features[:5]:
                LOG.info(f"  {name}: {importance:.3f}")

        # Calculate training metrics
        y_pred = self.model.predict(X)
        accuracy = (y_pred == y).mean()

        # Precision and recall for profitable signals
        true_positives = ((y_pred == 1) & (y == 1)).sum()
        false_positives = ((y_pred == 1) & (y == 0)).sum()
        false_negatives = ((y_pred == 0) & (y == 1)).sum()

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'total_signals': len(y),
            'profitable_signals': int(y.sum()),
        }

        LOG.info(f"Training complete: Accuracy={accuracy:.2%}, Precision={precision:.2%}, Recall={recall:.2%}")

        return metrics

    def predict(
        self,
        features: MetaFeatures,
        original_signal: int,
    ) -> MetaLearnerPrediction:
        """Predict if signal should be taken.

        Args:
            features: Signal and market features
            original_signal: Original signal direction (+1, -1, or 0)

        Returns:
            MetaLearnerPrediction with recommendation
        """
        if not self.is_trained:
            # Not trained - pass through original signal
            return MetaLearnerPrediction(
                should_take=True,
                confidence=0.5,
                probability_profitable=0.5,
                adjusted_signal=original_signal,
                reason="Meta-learner not trained - passing through signal",
            )

        # Convert features to array
        X = features.to_array().reshape(1, -1)

        # Get probability
        try:
            prob = self.model.predict_proba(X)[0, 1]  # Probability of class 1 (profitable)
        except Exception as e:
            LOG.error(f"Prediction failed: {e}")
            return MetaLearnerPrediction(
                should_take=True,
                confidence=0.5,
                probability_profitable=0.5,
                adjusted_signal=original_signal,
                reason=f"Prediction error: {e}",
            )

        # Decision logic
        should_take = prob >= self.threshold

        if should_take:
            adjusted_signal = original_signal
            reason = f"High confidence ({prob:.1%}) - taking signal"
        else:
            adjusted_signal = 0  # Filter out signal
            reason = f"Low confidence ({prob:.1%}) - filtering signal (threshold {self.threshold:.1%})"

        return MetaLearnerPrediction(
            should_take=should_take,
            confidence=prob,
            probability_profitable=prob,
            adjusted_signal=adjusted_signal,
            reason=reason,
        )

    def save(self, path: str) -> None:
        """Save trained model to disk.

        Args:
            path: Path to save model pickle
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        import pickle
        from pathlib import Path

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'threshold': self.threshold,
                'feature_importance': self.feature_importance,
            }, f)

        LOG.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load trained model from disk.

        Args:
            path: Path to model pickle
        """
        import pickle
        from pathlib import Path

        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with open(load_path, 'rb') as f:
            data = pickle.load(f)

        self.model = data['model']
        self.threshold = data.get('threshold', 0.60)
        self.feature_importance = data.get('feature_importance')
        self.is_trained = True

        LOG.info(f"Model loaded from {path}")


def extract_meta_features(
    df: pd.DataFrame,
    signal_direction: int,
    signal_confidence: float,
    signal_source: str,
    lookback: int = 20,
) -> MetaFeatures:
    """Extract meta-features from current market state.

    Args:
        df: OHLCV DataFrame with indicators
        signal_direction: Signal direction (+1, -1, 0)
        signal_confidence: Signal confidence (0-1)
        signal_source: Source of signal (e.g., "ema_cross")
        lookback: Lookback period for calculations

    Returns:
        MetaFeatures object
    """
    latest = df.iloc[-1]
    recent = df.tail(lookback)

    # Volatility
    atr = latest.get('atr', 0.0)
    close = latest.get('close', 1.0)
    volatility = atr / close if close > 0 else 0.0

    # Trend strength
    trend_strength = latest.get('adx', 25.0) / 100.0  # Normalize to 0-1

    # Volume
    volume = latest.get('volume', 0.0)
    avg_volume = recent['volume'].mean() if 'volume' in recent.columns else 1.0
    volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0

    # RSI
    rsi_value = latest.get('rsi', 50.0)

    # Price vs SMA
    sma_20 = recent['close'].mean()
    price_vs_sma = (close - sma_20) / sma_20 if sma_20 > 0 else 0.0

    # Distance from high/low
    high_20 = recent['high'].max()
    low_20 = recent['low'].min()
    distance_from_high = (high_20 - close) / close if close > 0 else 0.0
    distance_from_low = (close - low_20) / close if close > 0 else 0.0

    # Time features
    if isinstance(df.index, pd.DatetimeIndex):
        hour_of_day = df.index[-1].hour
        day_of_week = df.index[-1].weekday()
    else:
        hour_of_day = 0
        day_of_week = 0

    # Bars since last signal (simplified - would need signal history)
    bars_since_last_signal = 10  # Placeholder

    # News sentiment (if available in df)
    news_sentiment = latest.get('sentiment', 0.0)

    return MetaFeatures(
        signal_direction=signal_direction,
        signal_confidence=signal_confidence,
        signal_source=signal_source,
        volatility=volatility,
        trend_strength=trend_strength,
        volume_ratio=volume_ratio,
        rsi_value=rsi_value,
        price_vs_sma=price_vs_sma,
        distance_from_high=distance_from_high,
        distance_from_low=distance_from_low,
        hour_of_day=hour_of_day,
        day_of_week=day_of_week,
        bars_since_last_signal=bars_since_last_signal,
        news_sentiment=news_sentiment,
    )
