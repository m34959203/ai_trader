import numpy as np
import pandas as pd

from src.indicators import (
    candlestick_patterns,
    cross_over,
    cross_under,
    ema,
    force_index,
    on_balance_volume,
    resample_ohlcv,
    rsi,
    sma,
)

def test_ema_sma_rsi():
    s = pd.Series([1,2,3,4,5,6,7,8,9,10])
    assert round(ema(s, 3).iloc[-1], 2) > 8
    assert round(sma(s, 3).iloc[-1], 2) == 9.0
    r = rsi(s, 5)
    assert 0 <= r.dropna().iloc[-1] <= 100

def test_cross_over_under():
    f = pd.Series([1,2,3,4,5])
    s = pd.Series([5,4,3,2,1])
    assert cross_over(f, s).any()
    assert not cross_under(f, s).any()


def test_force_index_and_obv():
    idx = pd.date_range("2024-01-01", periods=4, freq="h")
    close = pd.Series([100.0, 105.0, 102.0, 110.0], index=idx)
    volume = pd.Series([10.0, 11.0, 9.0, 15.0], index=idx)

    fi = force_index(close, volume, period=2)
    raw = close.diff().fillna(0.0) * volume
    expected = raw.ewm(span=2, adjust=False).mean()
    assert np.isclose(fi.iloc[-1], expected.iloc[-1])

    obv = on_balance_volume(close, volume)
    manual_obv = pd.Series([0.0, 11.0, 2.0, 17.0], index=idx)
    pd.testing.assert_series_equal(obv, manual_obv)


def test_candlestick_and_resample():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    df = pd.DataFrame(
        {
            "open": [10.0, 10.5, 9.8, 10.1],
            "high": [10.6, 11.0, 10.9, 10.3],
            "low": [9.0, 10.2, 9.7, 9.9],
            "close": [10.4, 10.0, 10.8, 10.1],
        },
        index=dates,
    )

    patterns = candlestick_patterns(df)
    assert patterns.loc[dates[0], "hammer"]
    assert patterns.loc[dates[2], "bullish_engulfing"]
    assert patterns.loc[dates[3], "doji"]

    minute_idx = pd.date_range("2024-01-01", periods=4, freq="min")
    minute_df = pd.DataFrame(
        {
            "open": [1, 2, 3, 4],
            "high": [2, 3, 4, 5],
            "low": [0, 1, 2, 3],
            "close": [1.5, 2.5, 3.5, 4.5],
            "volume": [10, 20, 30, 40],
        },
        index=minute_idx,
    )

    resampled = resample_ohlcv(minute_df, "2min")
    assert list(resampled.index) == list(pd.date_range("2024-01-01", periods=2, freq="2min"))
    first = resampled.iloc[0]
    assert first.open == 1
    assert first.high == 3
    assert first.low == 0
    assert first.close == 2.5
    assert first.volume == 30
