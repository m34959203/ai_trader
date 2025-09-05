import pandas as pd
from src.strategy import ema_cross_signals

def test_ema_cross_signals():
    df = pd.DataFrame({
        "timestamp": range(10),
        "close": [1,2,3,4,5,4,3,2,1,2]
    })
    sigs = ema_cross_signals(df, fast=2, slow=3)
    assert "signal" in sigs.columns
    assert set(sigs["signal"].unique()).issubset({-1,0,1})
