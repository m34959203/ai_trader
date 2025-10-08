import math
from datetime import datetime, timezone

import pandas as pd

from tasks import auto_trader
from utils.risk_config import RiskConfig


def _make_df(close: list[float]) -> pd.DataFrame:
    now = datetime.now(timezone.utc)
    rows = [
        {"t": int((now.timestamp() - (len(close) - idx) * 60)), "o": c, "h": c + 5, "l": c - 5, "c": c, "v": 100}
        for idx, c in enumerate(close)
    ]
    return auto_trader._rows_to_df(rows)


def test_atr_stop_pct_uses_atr(monkeypatch):
    df = _make_df([100 + i for i in range(50)])
    monkeypatch.setattr(auto_trader, "ATR_PERIOD", 14, raising=False)
    monkeypatch.setattr(auto_trader, "ATR_MULT", 1.5, raising=False)
    monkeypatch.setattr(auto_trader, "SL_PCT", 0.0, raising=False)
    pct = auto_trader._atr_stop_pct(df)
    assert pct > 0
    # atr * mult / close should be roughly 1.5% for monotonic series
    assert pct <= 0.06


def test_calc_quote_amount_respects_risk_fraction(monkeypatch):
    cfg = RiskConfig(risk_pct_per_trade=0.02)
    monkeypatch.setattr(auto_trader, "QUOTE_USDT", 10_000.0, raising=False)
    quote = auto_trader._calc_quote_amount(equity=5_000.0, sl_pct=0.01, cfg=cfg)
    # risk_fraction = 0.02 / 0.01 = 2, capped at 1.0 inside helper
    assert math.isclose(quote, 5_000.0, rel_tol=1e-6)


def test_risk_snapshot_from_positions(monkeypatch):
    cfg = RiskConfig(risk_pct_per_trade=0.01, min_sl_distance_pct=0.001, portfolio_max_risk_pct=0.1)
    monkeypatch.setattr(auto_trader, "USE_PORTFOLIO_RISK", True, raising=False)
    monkeypatch.setattr(auto_trader, "MAX_RISK_PORTFOLIO", 0.0, raising=False)
    monkeypatch.setattr(auto_trader, "_load_risk_config_cached", lambda: cfg, raising=False)
    positions = [
        {"symbol": "BTCUSDT", "qty": 0.1, "entry_price": 30_000.0, "stop_loss_price": 29_700.0},
    ]
    snap = auto_trader._risk_snapshot_from_positions(positions, equity=10_000.0)
    assert snap.config == cfg
    assert snap.portfolio_used > 0
    assert snap.portfolio_used < 1.0
