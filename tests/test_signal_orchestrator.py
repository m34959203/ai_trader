import numpy as np
import pandas as pd

from src.analysis.analyze_market import AnalysisConfig
from src.analysis.signal_orchestrator import MultiStrategyOrchestrator, OrchestratedSignal
from src.strategy import StrategyDefinition, StrategyEnsembleConfig


def _make_ohlcv_frame(rows: int = 240) -> pd.DataFrame:
    now = pd.Timestamp.now(tz="UTC")
    idx = pd.date_range(end=now, periods=rows, freq="h", tz="UTC")
    base = np.linspace(100.0, 130.0, rows)
    wave = np.sin(np.linspace(0, 6 * np.pi, rows)) * 0.3
    close = base + wave
    open_ = close - 0.1
    high = close + 0.2
    low = close - 0.3
    volume = 1_000 + np.linspace(0, 500, rows)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


def _strategy_config() -> StrategyEnsembleConfig:
    return StrategyEnsembleConfig(
        strategies=[
            StrategyDefinition(
                name="ema_trend",
                kind="ema",
                weight=1.0,
                params={"fast": 8, "slow": 21, "persist": 2},
            )
        ],
        ensemble_threshold=0.4,
    )


def _analysis_config() -> AnalysisConfig:
    return AnalysisConfig(
        ema_fast=8,
        ema_mid=21,
        ema_slow=55,
        min_bars_fast=120,
        min_bars_slow=40,
        buy_threshold=55,
        sell_threshold=45,
        news_enabled=False,
    )


def test_orchestrator_evaluate_returns_payload():
    df = _make_ohlcv_frame()
    orchestrator = MultiStrategyOrchestrator(
        _strategy_config(),
        analysis_config=_analysis_config(),
        history_limit=12,
    )

    result = orchestrator.evaluate(df, symbol="BTCUSDT")

    assert set(result.keys()) == {"analysis", "ensemble", "final"}
    assert isinstance(result["final"], OrchestratedSignal)

    ensemble = result["ensemble"]
    assert "latest" in ensemble and "recent" in ensemble
    if ensemble["latest"] is not None:
        assert len(ensemble["latest"]["details"]) == 1
        assert ensemble["latest"]["signal"] in (-1, 0, 1)

    final = result["final"]
    assert final.signal in {"buy", "sell", "flat"}
    assert isinstance(final.confidence, int)
    assert final.reasons  # non-empty reasoning
    assert set(final.sources.keys()) == {"analysis", "ensemble"}


def test_blend_signals_conflict_goes_flat():
    reasons = ["EMA-сходятся"]
    signal = MultiStrategyOrchestrator._blend_signals(
        analysis_signal="buy",
        analysis_confidence=70,
        analysis_reasons=reasons,
        ensemble_signal=-1,
        ensemble_score=-0.8,
    )

    assert isinstance(signal, OrchestratedSignal)
    assert signal.signal == "flat"
    assert any("Конфликт" in reason for reason in signal.reasons)
    assert signal.sources["analysis"]["signal"] == "buy"
    assert signal.sources["ensemble"]["signal"] == -1
