import pandas as pd
from src.indicators import ema, sma, rsi, cross_over, cross_under

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
