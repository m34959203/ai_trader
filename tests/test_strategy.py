import json
import pandas as pd

import pytest

from src.strategy import StrategyEnsembleConfig, ema_cross_signals, load_strategy_config

def test_ema_cross_signals():
    df = pd.DataFrame({
        "timestamp": range(10),
        "close": [1,2,3,4,5,4,3,2,1,2]
    })
    sigs = ema_cross_signals(df, fast=2, slow=3)
    assert "signal" in sigs.columns
    assert set(sigs["signal"].unique()).issubset({-1,0,1})


def test_strategy_config_validation(tmp_path):
    payload = {
        "strategies": [
            {"name": "ema", "kind": "ema_cross", "weight": 0.7, "params": {"fast": 5, "slow": 20}},
            {"name": "rsi", "kind": "rsi_reversion", "weight": 0.3, "params": {"period": 14}},
        ],
        "ensemble": {"threshold": 0.6},
        "frequency_filter": {"min_bars_between": 2, "max_signals_per_day": 10},
    }
    path = tmp_path / "strategy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    cfg = load_strategy_config(path)
    assert isinstance(cfg, StrategyEnsembleConfig)
    assert len(cfg.strategies) == 2
    assert abs(cfg.ensemble_threshold - 0.6) < 1e-9
    assert cfg.frequency_filter.min_bars_between == 2
    assert cfg.frequency_filter.max_signals_per_day == 10


def test_invalid_strategy_config(tmp_path):
    payload = {"strategies": [{"name": "dup", "kind": "ema_cross"}, {"name": "dup", "kind": "ema_cross"}]}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        load_strategy_config(path)
