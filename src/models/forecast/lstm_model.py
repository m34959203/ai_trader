"""
LSTM Forecaster for cryptocurrency price prediction.

Lightweight implementation that works with or without TensorFlow.
Falls back to simple linear regression if TensorFlow is unavailable.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge


@dataclass
class LSTMConfig:
    """Configuration for LSTM model."""

    sequence_length: int = 60  # Number of past bars to use
    n_forecast: int = 3        # Number of future bars to predict
    features: list[str] = None
    lstm_units: tuple[int, int] = (128, 64)
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2

    def __post_init__(self):
        if self.features is None:
            self.features = ['close', 'volume', 'rsi', 'macd', 'atr']


class LSTMForecaster:
    """
    LSTM model for multi-step ahead price forecasting.

    If TensorFlow is available, uses LSTM layers.
    Otherwise, falls back to Ridge regression with lagged features.
    """

    def __init__(self, config: Optional[LSTMConfig] = None):
        """
        Initialize forecaster.

        Args:
            config: Configuration parameters
        """
        self.config = config or LSTMConfig()
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_names = self.config.features
        self.is_fitted = False
        self.use_tensorflow = False

        # Try to import TensorFlow
        try:
            import tensorflow as tf
            from tensorflow import keras

            self.tf = tf
            self.keras = keras
            self.use_tensorflow = True
            print("Using TensorFlow LSTM model")
        except ImportError:
            print("TensorFlow not available, using Ridge regression fallback")
            self.use_tensorflow = False

    def _create_sequences(
        self,
        data: np.ndarray,
        target: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sequences for supervised learning.

        Args:
            data: Input data [n_samples, n_features]
            target: Target values [n_samples]

        Returns:
            X: [n_sequences, sequence_length, n_features]
            y: [n_sequences, n_forecast] if target provided
        """
        X, y = [], []
        n_samples = len(data)

        for i in range(n_samples - self.config.sequence_length - self.config.n_forecast + 1):
            X.append(data[i : i + self.config.sequence_length])

            if target is not None:
                y.append(target[i + self.config.sequence_length : i + self.config.sequence_length + self.config.n_forecast])

        X = np.array(X)
        y = np.array(y) if target is not None else None

        return X, y

    def _build_lstm_model(self, input_shape: tuple) -> 'keras.Model':
        """Build TensorFlow LSTM model."""
        model = self.keras.Sequential([
            self.keras.layers.LSTM(
                self.config.lstm_units[0],
                return_sequences=True,
                input_shape=input_shape,
            ),
            self.keras.layers.Dropout(self.config.dropout),
            self.keras.layers.LSTM(
                self.config.lstm_units[1],
                return_sequences=False,
            ),
            self.keras.layers.Dropout(self.config.dropout),
            self.keras.layers.Dense(32, activation='relu'),
            self.keras.layers.Dense(self.config.n_forecast),
        ])

        model.compile(
            optimizer=self.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae'],
        )

        return model

    def _build_ridge_model(self) -> Ridge:
        """Build Ridge regression fallback model."""
        return Ridge(alpha=1.0)

    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        verbose: int = 0,
    ) -> dict:
        """
        Fit the model on training data.

        Args:
            df: DataFrame with features
            target_col: Name of target column to predict
            verbose: Verbosity level (0=silent, 1=progress bar)

        Returns:
            Training history/metrics
        """
        # Extract features
        feature_data = df[self.feature_names].values
        target_data = df[target_col].values

        # Scale data
        scaled_features = self.scaler.fit_transform(feature_data)
        scaled_target = (target_data - target_data.mean()) / target_data.std()

        # Create sequences
        X, y = self._create_sequences(scaled_features, scaled_target)

        if X.shape[0] == 0:
            raise ValueError(f"Not enough data. Need at least {self.config.sequence_length + self.config.n_forecast} samples")

        # Train model
        if self.use_tensorflow:
            # Build LSTM model
            if self.model is None:
                input_shape = (self.config.sequence_length, len(self.feature_names))
                self.model = self._build_lstm_model(input_shape)

            # Split train/val
            n_train = int(len(X) * (1 - self.config.validation_split))
            X_train, X_val = X[:n_train], X[n_train:]
            y_train, y_val = y[:n_train], y[n_train:]

            # Early stopping callback
            early_stop = self.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
            )

            # Train
            history = self.model.fit(
                X_train,
                y_train,
                batch_size=self.config.batch_size,
                epochs=self.config.epochs,
                validation_data=(X_val, y_val),
                callbacks=[early_stop],
                verbose=verbose,
            )

            # Save target statistics for inverse transform
            self._target_mean = target_data.mean()
            self._target_std = target_data.std()

            self.is_fitted = True

            return {
                'final_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]),
                'epochs_trained': len(history.history['loss']),
            }

        else:
            # Fallback: Ridge regression with flattened sequences
            if self.model is None:
                self.model = self._build_ridge_model()

            # Flatten sequences for Ridge
            X_flat = X.reshape(X.shape[0], -1)

            # Train
            self.model.fit(X_flat, y)

            # Save target statistics
            self._target_mean = target_data.mean()
            self._target_std = target_data.std()

            self.is_fitted = True

            # Simple train score
            train_score = self.model.score(X_flat, y)

            return {
                'train_score': float(train_score),
                'model_type': 'Ridge',
            }

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict future prices.

        Args:
            df: Recent data with features (at least sequence_length rows)

        Returns:
            Predicted prices [n_forecast]
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Extract and scale features
        feature_data = df[self.feature_names].tail(self.config.sequence_length).values
        scaled_features = self.scaler.transform(feature_data)

        # Create sequence (single sample)
        X = scaled_features.reshape(1, self.config.sequence_length, len(self.feature_names))

        # Predict
        if self.use_tensorflow:
            predictions_scaled = self.model.predict(X, verbose=0)[0]
        else:
            X_flat = X.reshape(1, -1)
            predictions_scaled = self.model.predict(X_flat)[0]

        # Inverse transform
        predictions = predictions_scaled * self._target_std + self._target_mean

        return predictions

    def evaluate(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
    ) -> dict:
        """
        Evaluate model on test data.

        Args:
            df: Test DataFrame
            target_col: Target column name

        Returns:
            Dictionary with metrics (MSE, MAE, directional accuracy, R²)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet. Call fit() first.")

        # Extract features and target
        feature_data = df[self.feature_names].values
        target_data = df[target_col].values

        # Scale
        scaled_features = self.scaler.transform(feature_data)
        scaled_target = (target_data - self._target_mean) / self._target_std

        # Create sequences
        X, y = self._create_sequences(scaled_features, scaled_target)

        if X.shape[0] == 0:
            raise ValueError("Not enough test data")

        # Predict
        if self.use_tensorflow:
            y_pred = self.model.predict(X, verbose=0)
        else:
            X_flat = X.reshape(X.shape[0], -1)
            y_pred = self.model.predict(X_flat)

        # Metrics
        mse = np.mean((y - y_pred) ** 2)
        mae = np.mean(np.abs(y - y_pred))
        rmse = np.sqrt(mse)

        # R² score
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Directional accuracy (did we predict direction correctly?)
        # For first forecast step
        y_direction = np.sign(y[:, 0] - scaled_target[self.config.sequence_length : len(y) + self.config.sequence_length])
        y_pred_direction = np.sign(y_pred[:, 0] - scaled_target[self.config.sequence_length : len(y) + self.config.sequence_length])
        directional_accuracy = np.mean(y_direction == y_pred_direction)

        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2),
            'directional_accuracy': float(directional_accuracy),
        }

    def save(self, path: Union[str, Path]):
        """Save model to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        config_path = path / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

        # Save scaler and metadata
        metadata = {
            'target_mean': float(self._target_mean),
            'target_std': float(self._target_std),
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted,
            'use_tensorflow': self.use_tensorflow,
        }

        metadata_path = path / 'metadata.pkl'
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'metadata': metadata,
            }, f)

        # Save model
        if self.use_tensorflow and self.model is not None:
            model_path = path / 'lstm_model.h5'
            self.model.save(model_path)
        elif self.model is not None:
            model_path = path / 'ridge_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)

        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'LSTMForecaster':
        """Load model from disk."""
        path = Path(path)

        # Load config
        config_path = path / 'config.json'
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
            config = LSTMConfig(**config_dict)

        # Create instance
        forecaster = cls(config=config)

        # Load metadata and scaler
        metadata_path = path / 'metadata.pkl'
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            forecaster.scaler = data['scaler']
            metadata = data['metadata']

        forecaster._target_mean = metadata['target_mean']
        forecaster._target_std = metadata['target_std']
        forecaster.feature_names = metadata['feature_names']
        forecaster.is_fitted = metadata['is_fitted']
        forecaster.use_tensorflow = metadata['use_tensorflow']

        # Load model
        if forecaster.use_tensorflow:
            model_path = path / 'lstm_model.h5'
            if model_path.exists():
                try:
                    import tensorflow as tf
                    forecaster.model = tf.keras.models.load_model(model_path)
                except ImportError:
                    print("Warning: TensorFlow not available, model not loaded")
        else:
            model_path = path / 'ridge_model.pkl'
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    forecaster.model = pickle.load(f)

        print(f"Model loaded from {path}")
        return forecaster


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, '/home/user/ai_trader')

    from src.indicators import rsi, macd, atr

    print("Testing LSTM Forecaster...")
    print("=" * 60)

    # Generate synthetic data
    np.random.seed(42)
    n = 500
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')

    # Random walk with trend
    returns = np.random.randn(n) * 0.02 + 0.001
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    volume = np.random.randint(1000, 10000, n)

    df = pd.DataFrame({
        'close': close,
        'high': high,
        'low': low,
        'volume': volume,
    }, index=dates)

    # Calculate indicators
    df['rsi'] = rsi(df['close'])
    macd_df = macd(df['close'])
    df['macd'] = macd_df['macd']
    df['atr'] = atr(df['high'], df['low'], df['close'])

    # Remove NaN
    df = df.dropna()

    print(f"Data shape: {df.shape}")
    print(f"Features: {['close', 'volume', 'rsi', 'macd', 'atr']}")

    # Split train/test
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Create and train model
    config = LSTMConfig(
        sequence_length=30,
        n_forecast=3,
        features=['close', 'volume', 'rsi', 'macd', 'atr'],
        epochs=50,
    )

    forecaster = LSTMForecaster(config=config)

    print("\nTraining model...")
    history = forecaster.fit(train_df, target_col='close', verbose=0)
    print("Training completed!")
    print(f"Training metrics: {history}")

    # Evaluate
    print("\nEvaluating on test set...")
    metrics = forecaster.evaluate(test_df, target_col='close')
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Make prediction
    print("\nMaking predictions...")
    recent_data = test_df.tail(50)
    predictions = forecaster.predict(recent_data)

    print(f"Last close price: {test_df['close'].iloc[-1]:.2f}")
    print(f"Predicted next {config.n_forecast} prices:")
    for i, pred in enumerate(predictions, 1):
        print(f"  Step {i}: {pred:.2f}")

    # Save model
    save_path = Path('/home/user/ai_trader/models/lstm_test')
    forecaster.save(save_path)

    # Load and test
    print("\nTesting model save/load...")
    loaded_forecaster = LSTMForecaster.load(save_path)
    loaded_predictions = loaded_forecaster.predict(recent_data)

    print("Loaded model predictions match original:", np.allclose(predictions, loaded_predictions))

    print("\n" + "=" * 60)
    print("LSTM Forecaster test completed successfully!")
