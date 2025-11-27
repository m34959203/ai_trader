"""
Triple-Barrier Method for labeling financial time series.

Based on "Advances in Financial Machine Learning" by Marcos López de Prado.

The method places three barriers around each observation:
1. Upper barrier (profit target): entry_price + (pt_multiplier × ATR)
2. Lower barrier (stop-loss): entry_price - (sl_multiplier × ATR)
3. Time barrier: maximum holding period in bars

The label is determined by which barrier is touched first:
- +1 if upper barrier touched (profitable trade)
- -1 if lower barrier touched (stop-loss hit)
- 0 if time barrier reached first (timeout, neutral)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TripleBarrierConfig:
    """Configuration for Triple-Barrier labeling."""

    profit_target_multiplier: float = 2.0  # PT = entry + (multiplier × ATR)
    stop_loss_multiplier: float = 1.0      # SL = entry - (multiplier × ATR)
    max_holding_period: int = 20           # Maximum bars to hold
    min_return_pct: float = 0.0001         # Minimum 0.01% return to consider


def triple_barrier_labels(
    prices: pd.Series,
    *,
    atr: Optional[pd.Series] = None,
    config: Optional[TripleBarrierConfig] = None,
) -> pd.DataFrame:
    """
    Generate labels using the Triple-Barrier method.

    Args:
        prices: Close prices as pandas Series with DatetimeIndex
        atr: Average True Range for dynamic barriers (if None, uses % barriers)
        config: Configuration parameters

    Returns:
        DataFrame with columns:
            - label: {-1, 0, 1} for stop-loss, timeout, profit
            - barrier_hit: {'stop', 'time', 'profit'}
            - holding_period: number of bars held
            - return_pct: actual return percentage
            - profit_barrier: upper barrier price
            - stop_barrier: lower barrier price
            - time_barrier_idx: index where time barrier occurs

    Example:
        >>> prices = df['close']
        >>> atr = calculate_atr(df['high'], df['low'], df['close'])
        >>> labels = triple_barrier_labels(prices, atr=atr)
        >>> print(labels.head())
    """
    if config is None:
        config = TripleBarrierConfig()

    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.reset_index(drop=True)

    n = len(prices)

    # Initialize result arrays
    labels = np.zeros(n, dtype=np.int8)
    barrier_hit = np.array(['none'] * n, dtype=object)
    holding_periods = np.zeros(n, dtype=np.int32)
    returns = np.zeros(n, dtype=np.float32)
    profit_barriers = np.zeros(n, dtype=np.float32)
    stop_barriers = np.zeros(n, dtype=np.float32)
    time_barrier_indices = np.zeros(n, dtype=np.int32)

    # Compute barriers for each entry point
    for i in range(n - 1):
        entry_price = prices.iloc[i]

        # Calculate dynamic barriers using ATR
        if atr is not None and i < len(atr):
            atr_value = atr.iloc[i]
            if np.isnan(atr_value) or atr_value <= 0:
                atr_value = entry_price * 0.02  # Fallback: 2% of price

            profit_barrier = entry_price + (config.profit_target_multiplier * atr_value)
            stop_barrier = entry_price - (config.stop_loss_multiplier * atr_value)
        else:
            # Fallback to percentage-based barriers
            profit_barrier = entry_price * (1 + config.profit_target_multiplier * 0.01)
            stop_barrier = entry_price * (1 - config.stop_loss_multiplier * 0.01)

        profit_barriers[i] = profit_barrier
        stop_barriers[i] = stop_barrier

        # Time barrier
        time_barrier_idx = min(i + config.max_holding_period, n - 1)
        time_barrier_indices[i] = time_barrier_idx

        # Scan forward to find which barrier is hit first
        hit_barrier = 'time'
        hit_idx = time_barrier_idx

        for j in range(i + 1, time_barrier_idx + 1):
            current_price = prices.iloc[j]

            # Check profit barrier
            if current_price >= profit_barrier:
                hit_barrier = 'profit'
                hit_idx = j
                break

            # Check stop-loss barrier
            if current_price <= stop_barrier:
                hit_barrier = 'stop'
                hit_idx = j
                break

        # Calculate results
        holding_periods[i] = hit_idx - i
        exit_price = prices.iloc[hit_idx]
        return_pct = (exit_price - entry_price) / entry_price
        returns[i] = return_pct
        barrier_hit[i] = hit_barrier

        # Assign label
        if abs(return_pct) < config.min_return_pct:
            labels[i] = 0  # Too small movement, neutral
        elif hit_barrier == 'profit':
            labels[i] = 1
        elif hit_barrier == 'stop':
            labels[i] = -1
        else:  # time barrier
            labels[i] = 0

    # Last bar cannot be labeled (no future data)
    labels[-1] = 0
    barrier_hit[-1] = 'none'
    holding_periods[-1] = 0
    returns[-1] = 0.0
    profit_barriers[-1] = prices.iloc[-1]
    stop_barriers[-1] = prices.iloc[-1]
    time_barrier_indices[-1] = n - 1

    # Create result DataFrame
    result = pd.DataFrame(
        {
            'label': labels,
            'barrier_hit': barrier_hit,
            'holding_period': holding_periods,
            'return_pct': returns,
            'profit_barrier': profit_barriers,
            'stop_barrier': stop_barriers,
            'time_barrier_idx': time_barrier_indices,
        },
        index=prices.index,
    )

    return result


def get_label_distribution(labels_df: pd.DataFrame) -> dict:
    """
    Get distribution statistics for labels.

    Args:
        labels_df: Output from triple_barrier_labels()

    Returns:
        Dictionary with label counts and percentages
    """
    label_counts = labels_df['label'].value_counts().sort_index()
    total = len(labels_df)

    distribution = {
        'total': total,
        'profit': int(label_counts.get(1, 0)),
        'neutral': int(label_counts.get(0, 0)),
        'loss': int(label_counts.get(-1, 0)),
        'profit_pct': float(label_counts.get(1, 0) / total * 100),
        'neutral_pct': float(label_counts.get(0, 0) / total * 100),
        'loss_pct': float(label_counts.get(-1, 0) / total * 100),
    }

    # Average returns per label
    for label in [-1, 0, 1]:
        label_name = {-1: 'loss', 0: 'neutral', 1: 'profit'}[label]
        mask = labels_df['label'] == label
        if mask.any():
            avg_return = labels_df.loc[mask, 'return_pct'].mean()
            avg_holding = labels_df.loc[mask, 'holding_period'].mean()
            distribution[f'{label_name}_avg_return'] = float(avg_return)
            distribution[f'{label_name}_avg_holding'] = float(avg_holding)
        else:
            distribution[f'{label_name}_avg_return'] = 0.0
            distribution[f'{label_name}_avg_holding'] = 0.0

    return distribution


def balance_labels(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    method: str = 'undersample',
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Balance imbalanced labels for training.

    Args:
        X: Feature DataFrame
        y: Labels Series
        method: 'undersample', 'oversample', or 'smote'
        random_state: Random seed

    Returns:
        Tuple of balanced (X, y)
    """
    from sklearn.utils import resample

    if method == 'undersample':
        # Undersample majority class to match minority
        label_counts = y.value_counts()
        min_count = label_counts.min()

        X_balanced = []
        y_balanced = []

        for label in y.unique():
            mask = y == label
            X_label = X[mask]
            y_label = y[mask]

            if len(y_label) > min_count:
                X_resampled, y_resampled = resample(
                    X_label,
                    y_label,
                    n_samples=min_count,
                    random_state=random_state,
                    replace=False,
                )
            else:
                X_resampled, y_resampled = X_label, y_label

            X_balanced.append(X_resampled)
            y_balanced.append(y_resampled)

        X_result = pd.concat(X_balanced, axis=0)
        y_result = pd.concat(y_balanced, axis=0)

        # Shuffle
        indices = X_result.index.to_numpy()
        np.random.RandomState(random_state).shuffle(indices)

        return X_result.loc[indices], y_result.loc[indices]

    elif method == 'oversample':
        # Oversample minority classes to match majority
        label_counts = y.value_counts()
        max_count = label_counts.max()

        X_balanced = []
        y_balanced = []

        for label in y.unique():
            mask = y == label
            X_label = X[mask]
            y_label = y[mask]

            if len(y_label) < max_count:
                X_resampled, y_resampled = resample(
                    X_label,
                    y_label,
                    n_samples=max_count,
                    random_state=random_state,
                    replace=True,
                )
            else:
                X_resampled, y_resampled = X_label, y_label

            X_balanced.append(X_resampled)
            y_balanced.append(y_resampled)

        X_result = pd.concat(X_balanced, axis=0)
        y_result = pd.concat(y_balanced, axis=0)

        # Shuffle
        indices = X_result.index.to_numpy()
        np.random.RandomState(random_state).shuffle(indices)

        return X_result.loc[indices], y_result.loc[indices]

    elif method == 'smote':
        try:
            from imblearn.over_sampling import SMOTE

            smote = SMOTE(random_state=random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
        except ImportError:
            raise ImportError(
                "SMOTE requires imbalanced-learn: pip install imbalanced-learn"
            )

    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    # Example usage and testing
    import sys
    sys.path.insert(0, '/home/user/ai_trader')

    from src.indicators import atr as calculate_atr

    # Generate synthetic data for testing
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=1000, freq='1h')

    # Random walk with trend
    returns = np.random.randn(1000) * 0.02 + 0.0005
    prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates)

    # Calculate ATR (simplified)
    high = prices * 1.005
    low = prices * 0.995
    atr_series = calculate_atr(high, low, prices, period=14)

    # Generate labels
    config = TripleBarrierConfig(
        profit_target_multiplier=2.0,
        stop_loss_multiplier=1.0,
        max_holding_period=20,
    )

    labels_df = triple_barrier_labels(prices, atr=atr_series, config=config)

    # Print statistics
    print("Triple-Barrier Labeling Results:")
    print("-" * 50)

    distribution = get_label_distribution(labels_df)
    print(f"\nTotal samples: {distribution['total']}")
    print(f"Profit labels: {distribution['profit']} ({distribution['profit_pct']:.1f}%)")
    print(f"Neutral labels: {distribution['neutral']} ({distribution['neutral_pct']:.1f}%)")
    print(f"Loss labels: {distribution['loss']} ({distribution['loss_pct']:.1f}%)")

    print(f"\nAverage returns:")
    print(f"Profit: {distribution['profit_avg_return']*100:.2f}%")
    print(f"Neutral: {distribution['neutral_avg_return']*100:.2f}%")
    print(f"Loss: {distribution['loss_avg_return']*100:.2f}%")

    print(f"\nAverage holding periods:")
    print(f"Profit: {distribution['profit_avg_holding']:.1f} bars")
    print(f"Neutral: {distribution['neutral_avg_holding']:.1f} bars")
    print(f"Loss: {distribution['loss_avg_holding']:.1f} bars")

    print("\nFirst 10 labels:")
    print(labels_df.head(10))
