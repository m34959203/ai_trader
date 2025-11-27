"""Tests for Triple-Barrier labeling."""

import numpy as np
import pandas as pd
import pytest

from src.models.labeling.triple_barrier import (
    TripleBarrierConfig,
    balance_labels,
    get_label_distribution,
    triple_barrier_labels,
)


@pytest.fixture
def synthetic_prices():
    """Generate synthetic price data for testing."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=200, freq='1h')
    returns = np.random.randn(200) * 0.02 + 0.001  # Slight upward trend
    prices = pd.Series(100 * np.exp(np.cumsum(returns)), index=dates, name='close')
    return prices


@pytest.fixture
def synthetic_atr(synthetic_prices):
    """Generate synthetic ATR data."""
    # Simplified ATR: 2% of price
    atr = synthetic_prices * 0.02
    return atr


def test_triple_barrier_basic(synthetic_prices, synthetic_atr):
    """Test basic triple-barrier labeling."""
    config = TripleBarrierConfig(
        profit_target_multiplier=2.0,
        stop_loss_multiplier=1.0,
        max_holding_period=10,
    )

    labels_df = triple_barrier_labels(
        synthetic_prices,
        atr=synthetic_atr,
        config=config,
    )

    # Check output shape
    assert len(labels_df) == len(synthetic_prices)

    # Check columns exist
    assert 'label' in labels_df.columns
    assert 'barrier_hit' in labels_df.columns
    assert 'holding_period' in labels_df.columns
    assert 'return_pct' in labels_df.columns

    # Check label values are valid
    assert set(labels_df['label'].unique()).issubset({-1, 0, 1})

    # Check barrier_hit values are valid
    assert set(labels_df['barrier_hit'].unique()).issubset({'profit', 'stop', 'time', 'none'})

    # Check holding periods are reasonable
    assert labels_df['holding_period'].max() <= config.max_holding_period + 1


def test_triple_barrier_no_atr(synthetic_prices):
    """Test triple-barrier without ATR (percentage-based)."""
    labels_df = triple_barrier_labels(synthetic_prices)

    assert len(labels_df) == len(synthetic_prices)
    assert 'label' in labels_df.columns


def test_triple_barrier_profit_barrier():
    """Test that profit barrier works correctly."""
    # Create deterministic rising prices
    prices = pd.Series([100, 102, 105, 108, 110, 112])
    atr = pd.Series([1.0] * len(prices))

    config = TripleBarrierConfig(
        profit_target_multiplier=2.0,  # +2 ATR = +2 from entry
        stop_loss_multiplier=1.0,
        max_holding_period=5,
    )

    labels_df = triple_barrier_labels(prices, atr=atr, config=config)

    # First entry at 100 should hit profit barrier at 102 (100 + 2*1)
    assert labels_df.iloc[0]['barrier_hit'] == 'profit'
    assert labels_df.iloc[0]['label'] == 1


def test_triple_barrier_stop_barrier():
    """Test that stop-loss barrier works correctly."""
    # Create deterministic falling prices
    prices = pd.Series([100, 99, 97, 95, 93, 90])
    atr = pd.Series([1.0] * len(prices))

    config = TripleBarrierConfig(
        profit_target_multiplier=2.0,
        stop_loss_multiplier=1.0,  # -1 ATR = -1 from entry
        max_holding_period=5,
    )

    labels_df = triple_barrier_labels(prices, atr=atr, config=config)

    # First entry at 100 should hit stop barrier at 99 (100 - 1*1)
    assert labels_df.iloc[0]['barrier_hit'] == 'stop'
    assert labels_df.iloc[0]['label'] == -1


def test_triple_barrier_time_barrier():
    """Test that time barrier works correctly."""
    # Create sideways prices
    prices = pd.Series([100, 100.5, 100.2, 100.3, 100.1, 100.4])
    atr = pd.Series([2.0] * len(prices))  # Wide barriers

    config = TripleBarrierConfig(
        profit_target_multiplier=5.0,  # Wide profit barrier
        stop_loss_multiplier=5.0,      # Wide stop barrier
        max_holding_period=3,
    )

    labels_df = triple_barrier_labels(prices, atr=atr, config=config)

    # Should hit time barrier (no significant movement)
    assert labels_df.iloc[0]['barrier_hit'] == 'time'
    assert labels_df.iloc[0]['label'] == 0


def test_label_distribution(synthetic_prices, synthetic_atr):
    """Test label distribution calculation."""
    labels_df = triple_barrier_labels(synthetic_prices, atr=synthetic_atr)

    distribution = get_label_distribution(labels_df)

    # Check all keys exist
    assert 'total' in distribution
    assert 'profit' in distribution
    assert 'neutral' in distribution
    assert 'loss' in distribution
    assert 'profit_pct' in distribution
    assert 'neutral_pct' in distribution
    assert 'loss_pct' in distribution

    # Check percentages sum to 100
    total_pct = (
        distribution['profit_pct']
        + distribution['neutral_pct']
        + distribution['loss_pct']
    )
    assert abs(total_pct - 100.0) < 0.1  # Allow small floating-point error

    # Check counts sum to total
    total_count = (
        distribution['profit']
        + distribution['neutral']
        + distribution['loss']
    )
    assert total_count == distribution['total']


def test_balance_labels_undersample():
    """Test label balancing with undersampling."""
    # Create imbalanced dataset
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
    })

    y = pd.Series([1] * 70 + [0] * 20 + [-1] * 10)

    X_balanced, y_balanced = balance_labels(X, y, method='undersample')

    # Check all labels have equal count
    label_counts = y_balanced.value_counts()
    assert len(label_counts.unique()) == 1  # All counts should be equal

    # Check minimum class is preserved
    assert label_counts[0] == 10


def test_balance_labels_oversample():
    """Test label balancing with oversampling."""
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
    })

    y = pd.Series([1] * 70 + [0] * 20 + [-1] * 10)

    X_balanced, y_balanced = balance_labels(X, y, method='oversample')

    # Check all labels have equal count
    label_counts = y_balanced.value_counts()
    assert len(label_counts.unique()) == 1  # All counts should be equal

    # Check maximum class is preserved
    assert label_counts[0] == 70


def test_min_return_threshold():
    """Test that minimum return threshold filters out small movements."""
    # Create prices with tiny movements
    prices = pd.Series([100.0, 100.001, 100.002, 100.001, 100.0])
    atr = pd.Series([0.01] * len(prices))

    config = TripleBarrierConfig(
        profit_target_multiplier=2.0,
        stop_loss_multiplier=1.0,
        max_holding_period=3,
        min_return_pct=0.001,  # 0.1% minimum
    )

    labels_df = triple_barrier_labels(prices, atr=atr, config=config)

    # Small movements should be labeled as neutral (0)
    assert labels_df.iloc[0]['label'] == 0


def test_last_bar_handling(synthetic_prices):
    """Test that last bar is handled correctly."""
    labels_df = triple_barrier_labels(synthetic_prices)

    # Last bar should have neutral label (no future data)
    assert labels_df.iloc[-1]['label'] == 0
    assert labels_df.iloc[-1]['barrier_hit'] == 'none'
    assert labels_df.iloc[-1]['holding_period'] == 0


def test_barriers_are_dynamic():
    """Test that barriers adjust with changing ATR."""
    prices = pd.Series([100, 105, 110, 115, 120])
    atr = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])  # Increasing ATR

    labels_df = triple_barrier_labels(prices, atr=atr)

    # Profit barriers should increase with ATR
    assert labels_df['profit_barrier'].iloc[0] < labels_df['profit_barrier'].iloc[3]


def test_edge_case_single_price():
    """Test edge case with single price point."""
    prices = pd.Series([100.0])
    labels_df = triple_barrier_labels(prices)

    assert len(labels_df) == 1
    assert labels_df.iloc[0]['label'] == 0


def test_edge_case_two_prices():
    """Test edge case with two price points."""
    prices = pd.Series([100.0, 102.0])
    atr = pd.Series([1.0, 1.0])

    labels_df = triple_barrier_labels(prices, atr=atr)

    assert len(labels_df) == 2
    # First can be labeled, second cannot
    assert labels_df.iloc[1]['label'] == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
