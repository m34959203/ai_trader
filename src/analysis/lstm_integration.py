"""LSTM model integration for trading signal generation.

This module bridges the LSTM forecaster with the trading signal system,
providing ML-enhanced signal confidence and direction prediction.

Key features:
- LSTM prediction → trading signal conversion
- Confidence scoring based on prediction uncertainty
- Fallback to technical-only signals if LSTM unavailable
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from ..models.forecast.lstm_model import LSTMForecaster, LSTMConfig
from utils.structured_logging import get_logger

LOG = get_logger("ai_trader.lstm_integration")


@dataclass
class LSTMSignal:
    """LSTM-based trading signal."""

    direction: int  # +1 (buy), -1 (sell), 0 (flat)
    confidence: float  # 0.0 to 1.0
    predicted_price: float  # Predicted price at horizon
    current_price: float  # Current price
    price_change_pct: float  # Expected % change
    horizon: int  # Prediction horizon (bars ahead)

    def to_dict(self):
        return {
            "direction": self.direction,
            "confidence": self.confidence,
            "predicted_price": self.predicted_price,
            "current_price": self.current_price,
            "price_change_pct": self.price_change_pct,
            "horizon": self.horizon,
        }


class LSTMSignalGenerator:
    """Generate trading signals from LSTM price predictions.

    Converts LSTM multi-step forecasts into actionable trading signals
    with confidence scoring based on prediction uncertainty.

    Usage:
        >>> generator = LSTMSignalGenerator(
        ...     model_path="models/lstm_btc.pkl",
        ...     min_confidence=0.55,
        ...     min_move_pct=0.005,  # 0.5% minimum move
        ... )
        >>> signal = generator.generate_signal(df)
        >>> if signal.direction == 1 and signal.confidence > 0.6:
        ...     print("Strong BUY signal")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        lstm_config: Optional[LSTMConfig] = None,
        min_confidence: float = 0.55,
        min_move_pct: float = 0.005,  # 0.5% minimum move to generate signal
        horizon: int = 3,  # Use 3rd prediction (3 bars ahead)
    ):
        """Initialize LSTM signal generator.

        Args:
            model_path: Path to trained LSTM model pickle file
            lstm_config: LSTM configuration (if training new model)
            min_confidence: Minimum confidence to generate non-flat signal
            min_move_pct: Minimum predicted price change to generate signal
            horizon: Which prediction step to use (1-3)
        """
        self.min_confidence = min_confidence
        self.min_move_pct = min_move_pct
        self.horizon = max(1, min(horizon, 3))  # Clamp to 1-3

        # Initialize or load model
        self.model: Optional[LSTMForecaster] = None
        self.is_available = False

        if model_path:
            try:
                import pickle
                from pathlib import Path

                model_file = Path(model_path)
                if model_file.exists():
                    with open(model_file, 'rb') as f:
                        self.model = pickle.load(f)
                    self.is_available = True
                    LOG.info(f"Loaded LSTM model from {model_path}")
                else:
                    LOG.warning(f"Model file not found: {model_path}. LSTM signals disabled.")
            except Exception as e:
                LOG.error(f"Failed to load LSTM model: {e}. LSTM signals disabled.")

        if not self.is_available and lstm_config:
            # Create new model with config
            self.model = LSTMForecaster(config=lstm_config)
            LOG.info("Created new LSTM model (not trained)")

    def generate_signal(
        self,
        df: pd.DataFrame,
        fallback_to_flat: bool = True,
    ) -> LSTMSignal:
        """Generate trading signal from recent price data.

        Args:
            df: DataFrame with OHLCV + indicators (close, volume, rsi, macd, atr)
            fallback_to_flat: Return flat signal if LSTM unavailable (vs error)

        Returns:
            LSTMSignal with direction and confidence
        """
        # Check if LSTM is available and trained
        if not self.is_available or not self.model or not self.model.is_fitted:
            if fallback_to_flat:
                current_price = float(df['close'].iloc[-1])
                return LSTMSignal(
                    direction=0,
                    confidence=0.0,
                    predicted_price=current_price,
                    current_price=current_price,
                    price_change_pct=0.0,
                    horizon=self.horizon,
                )
            else:
                raise ValueError("LSTM model not available or not trained")

        try:
            # Get predictions
            predictions = self._get_predictions(df)

            # Extract prediction at desired horizon
            predicted_price = predictions[self.horizon - 1]
            current_price = float(df['close'].iloc[-1])

            # Calculate price change
            price_change_pct = (predicted_price - current_price) / current_price

            # Determine direction
            direction = self._calculate_direction(price_change_pct)

            # Calculate confidence
            confidence = self._calculate_confidence(predictions, current_price)

            # Apply minimum confidence threshold
            if confidence < self.min_confidence:
                direction = 0  # Suppress weak signals

            return LSTMSignal(
                direction=direction,
                confidence=confidence,
                predicted_price=predicted_price,
                current_price=current_price,
                price_change_pct=price_change_pct,
                horizon=self.horizon,
            )

        except Exception as e:
            LOG.error(f"LSTM signal generation failed: {e}")

            if fallback_to_flat:
                current_price = float(df['close'].iloc[-1])
                return LSTMSignal(
                    direction=0,
                    confidence=0.0,
                    predicted_price=current_price,
                    current_price=current_price,
                    price_change_pct=0.0,
                    horizon=self.horizon,
                )
            else:
                raise

    def _get_predictions(self, df: pd.DataFrame) -> np.ndarray:
        """Get LSTM predictions for next N steps.

        Args:
            df: DataFrame with required features

        Returns:
            Array of predicted prices [step1, step2, step3]
        """
        # Prepare features
        feature_cols = self.model.feature_names

        # Check required columns
        missing = [col for col in feature_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required features: {missing}")

        # Get recent data (model's sequence length)
        recent_data = df[feature_cols].tail(self.model.config.sequence_length)

        if len(recent_data) < self.model.config.sequence_length:
            raise ValueError(
                f"Insufficient data: need {self.model.config.sequence_length} bars, "
                f"got {len(recent_data)}"
            )

        # Predict
        predictions = self.model.predict(recent_data.values)

        return predictions

    def _calculate_direction(self, price_change_pct: float) -> int:
        """Calculate signal direction from predicted price change.

        Args:
            price_change_pct: Predicted price change as percentage

        Returns:
            +1 (buy), -1 (sell), or 0 (flat)
        """
        # Require minimum move threshold
        if abs(price_change_pct) < self.min_move_pct:
            return 0

        return 1 if price_change_pct > 0 else -1

    def _calculate_confidence(
        self,
        predictions: np.ndarray,
        current_price: float,
    ) -> float:
        """Calculate confidence score from prediction uncertainty.

        Lower prediction variance → higher confidence
        Larger predicted move → higher confidence (if directionally consistent)

        Args:
            predictions: Array of predicted prices
            current_price: Current price

        Returns:
            Confidence score 0.0 to 1.0
        """
        # Component 1: Prediction consistency (low variance = high confidence)
        pred_std = np.std(predictions)
        pred_mean = np.mean(predictions)

        # Coefficient of variation (normalized volatility)
        cv = pred_std / pred_mean if pred_mean > 0 else 1.0

        # Convert to confidence: cv=0 → conf=1.0, cv=0.1 → conf=0.5
        consistency_conf = max(0.0, 1.0 - cv * 10)

        # Component 2: Magnitude of predicted move
        # Larger moves with consistent direction → higher confidence
        price_changes = (predictions - current_price) / current_price
        mean_change = np.mean(price_changes)

        # Magnitude confidence: 0.5% = 0.5, 1% = 0.7, 2% = 0.9
        magnitude_conf = min(1.0, abs(mean_change) / 0.02)  # Saturate at 2%

        # Component 3: Directional consistency
        # All predictions same direction → high confidence
        directions = np.sign(price_changes)
        directional_agreement = np.mean(directions == np.sign(mean_change))

        # Combined confidence (weighted average)
        confidence = (
            0.4 * consistency_conf +
            0.3 * magnitude_conf +
            0.3 * directional_agreement
        )

        return float(np.clip(confidence, 0.0, 1.0))


def integrate_lstm_with_technical(
    technical_signal: int,
    technical_confidence: float,
    lstm_signal: LSTMSignal,
    lstm_weight: float = 0.3,
) -> Tuple[int, float]:
    """Combine LSTM signal with technical signals.

    Weighted ensemble:
    - 70% technical signals (EMA, RSI, BB, etc.)
    - 30% LSTM forecast

    Args:
        technical_signal: Technical signal (-1, 0, +1)
        technical_confidence: Technical confidence (0-1)
        lstm_signal: LSTM signal object
        lstm_weight: Weight for LSTM (0-1), technical gets (1 - lstm_weight)

    Returns:
        (combined_signal, combined_confidence)
    """
    tech_weight = 1.0 - lstm_weight

    # Normalize signals to -1 to +1 range
    tech_normalized = float(technical_signal) * technical_confidence
    lstm_normalized = float(lstm_signal.direction) * lstm_signal.confidence

    # Weighted average
    combined = tech_normalized * tech_weight + lstm_normalized * lstm_weight

    # Determine final signal direction
    if abs(combined) < 0.2:  # Weak combined signal
        final_signal = 0
    else:
        final_signal = 1 if combined > 0 else -1

    # Combined confidence (average of both, weighted by agreement)
    agreement = 1.0 if technical_signal == lstm_signal.direction else 0.5
    final_confidence = (
        (technical_confidence * tech_weight + lstm_signal.confidence * lstm_weight) *
        agreement
    )

    return final_signal, float(np.clip(final_confidence, 0.0, 1.0))
