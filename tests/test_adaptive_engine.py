import pandas as pd

from src.ai.adaptive import AdaptiveConfidenceEngine


def make_df():
    idx = pd.date_range("2024-01-01", periods=100, freq="H", tz="UTC")
    base = pd.DataFrame({
        "open": 100 + pd.Series(range(100)) * 0.1,
        "high": 100 + pd.Series(range(100)) * 0.1 + 0.5,
        "low": 100 + pd.Series(range(100)) * 0.1 - 0.5,
        "close": 100 + pd.Series(range(100)) * 0.1,
        "ema_fast": 100 + pd.Series(range(100)) * 0.1,
        "ema_slow": 100 + pd.Series(range(100)) * 0.09,
        "rsi": 55,
        "macd": 0.5,
        "atr": 1.0,
    }, index=idx)
    return base


def test_adaptive_engine_multiplier():
    engine = AdaptiveConfidenceEngine()
    df = make_df()
    payload = {"confidence": 60, "reasons": []}
    signal = engine.adjust(payload, df=df)
    assert 0 <= signal.confidence <= 100
    assert signal.regime in {"trend", "flat", "turbulent", "unknown"}
    assert signal.multiplier > 0


