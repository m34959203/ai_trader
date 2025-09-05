# tests/test_paper_trader.py
from __future__ import annotations

import math
import pytest

from src.paper import PaperTrader


# ──────────────────────────────────────────────────────────────────────────────
# LONG: базовые сценарии
# ──────────────────────────────────────────────────────────────────────────────
def test_paper_trader_basic_tp_only():
    """
    Базовый сценарий (long):
      - вход по сигналу
      - на следующем баре достигается TP
    """
    trader = PaperTrader(sl_pct=0.10, tp_pct=0.20, fee_pct=0.0)  # SL=90, TP=120
    ts = 1

    # вход
    trader.on_signal(ts, 100.0, +1)
    assert trader.position is not None
    assert pytest.approx(trader.position["tp"], rel=0, abs=1e-9) == 120.0

    # бар, где достигается TP (high >= 120)
    trader.check_sl_tp(ts + 1, high=121.0, low=95.0)
    assert trader.position is None
    assert trader.trades and trader.trades[0]["reason"] == "TP"

    summary = trader.summary()
    assert summary["count"] == 1
    assert summary["pnl_sum"] > 0


def test_paper_trader_sl_has_priority_over_tp():
    """
    Приоритет SL над TP (long):
      - если low <= SL и одновременно high >= TP в пределах одной свечи,
        должна сработать сначала защита SL.
    """
    trader = PaperTrader(sl_pct=0.10, tp_pct=0.10, fee_pct=0.0)  # SL=90, TP=110
    ts = 100

    trader.on_signal(ts, 100.0, +1)
    # Свеча делает прокол вниз до 89 (SL), и вверх до 111 (TP) — должен сработать SL
    trader.check_sl_tp(ts + 1, high=111.0, low=89.0)

    assert trader.position is None
    assert trader.trades[-1]["reason"] in ("SL", "TRAIL_SL")
    assert trader.trades[-1]["pnl"] < 0


def test_trailing_stop_long():
    """
    Трейлинг-стоп (long):
      - вход по 100
      - цена растёт, SL подтягивается на trail_pct
      - затем падение ниже трейлингового SL → reason == "TRAIL_SL"
    """
    trader = PaperTrader(sl_pct=0.0, tp_pct=0.0, trail_pct=0.10, fee_pct=0.0)  # trail=10%
    ts = 10

    trader.on_signal(ts, 100.0, +1)
    assert trader.position is not None
    assert trader.position["sl"] is None  # пока SL не задан, появится после роста

    # Рост цены и закрытие бара выше – активирует подтягивание SL
    # close передаём явно, чтобы обновить max_price логикой трейлинга
    trader.check_sl_tp(ts + 1, high=112.0, low=100.0, close=112.0)
    assert trader.position["sl"] is not None
    # SL ≈ max_price*(1 - 0.1) = 112 * 0.9 = 100.8
    assert pytest.approx(trader.position["sl"], rel=0, abs=1e-9) == 100.8

    # Далее резкое падение ниже трейлингового SL
    trader.check_sl_tp(ts + 2, high=101.0, low=95.0, close=97.0)
    assert trader.position is None
    assert trader.trades[-1]["reason"] == "TRAIL_SL"


def test_multiple_trades_winrate_and_maxdd():
    """
    Делаем три сделки с детерминированным результатом:
      #1 прибыльная, #2 убыточная, #3 прибыльная.
    Проверяем winrate и max drawdown на суммарной кривой PnL.
    """
    trader = PaperTrader(sl_pct=0.10, tp_pct=0.10, fee_pct=0.0)  # symmetric SL/TP

    # trade #1: вход 100 → TP 110
    t = 1
    trader.on_signal(t, 100.0, +1)
    trader.check_sl_tp(t + 1, high=111.0, low=95.0)  # TP
    assert trader.position is None

    # trade #2: вход 100 → SL 90
    t += 10
    trader.on_signal(t, 100.0, +1)
    trader.check_sl_tp(t + 1, high=101.0, low=89.0)  # SL
    assert trader.position is None

    # trade #3: вход 100 → TP 110
    t += 10
    trader.on_signal(t, 100.0, +1)
    trader.check_sl_tp(t + 1, high=112.0, low=96.0)  # TP
    assert trader.position is None

    summ = trader.summary()
    # 3 сделки
    assert summ["count"] == 3
    # winrate = 2/3
    assert math.isclose(summ["winrate"], 2 / 3, rel_tol=1e-6)
    # max drawdown должен быть неотрицательным
    assert summ["max_dd"] >= 0.0
    # в нашем сценарии после первой прибыли идёт убыток -> есть просадка
    assert summ["max_dd"] > 0.0


def test_manual_close_and_current_position_snapshot():
    """
    Проверяем ручное закрытие (reason='signal') и snapshot текущей позиции.
    """
    trader = PaperTrader(sl_pct=0.05, tp_pct=0.10, fee_pct=0.0)

    ts = 1
    trader.on_signal(ts, 200.0, +1)
    snap = trader.current_position()
    assert snap is not None
    assert snap["entry_price"] == 200.0

    # Ручное закрытие (например, переворотной сигнал)
    trader.close(ts + 1, 202.0, reason="signal")
    assert trader.position is None
    assert trader.trades[-1]["reason"] == "signal"


# ──────────────────────────────────────────────────────────────────────────────
# SHORT: зеркальные сценарии (проверяем, что логика корректна для шорта)
# ──────────────────────────────────────────────────────────────────────────────
def test_paper_trader_short_tp_only():
    """
    Базовый сценарий (short):
      - вход по сигналу со стороной "short"
      - на следующем баре достигается TP (цена ниже)
    """
    trader = PaperTrader(sl_pct=0.10, tp_pct=0.20, fee_pct=0.0, side="short")  # SL=110, TP=80
    ts = 1

    trader.on_signal(ts, 100.0, +1)  # +1 сигнал трактуется как шорт для side="short"
    assert trader.position is not None
    assert pytest.approx(trader.position["tp"], rel=0, abs=1e-9) == 80.0

    # бар, где достигается TP (low <= 80)
    trader.check_sl_tp(ts + 1, high=105.0, low=79.0)
    assert trader.position is None
    assert trader.trades[-1]["reason"] == "TP"
    assert trader.trades[-1]["pnl"] > 0


def test_paper_trader_sl_has_priority_over_tp_short():
    """
    Приоритет SL над TP (short):
      - если high >= SL и одновременно low <= TP в пределах одной свечи,
        должна сработать сначала SL.
    """
    trader = PaperTrader(sl_pct=0.10, tp_pct=0.10, fee_pct=0.0, side="short")  # SL=110, TP=90
    ts = 100

    trader.on_signal(ts, 100.0, +1)
    # Свеча сначала «проколет» SL (high 111), хоть и достигнет TP (low 89) — должен быть SL
    trader.check_sl_tp(ts + 1, high=111.0, low=89.0)

    assert trader.position is None
    assert trader.trades[-1]["reason"] in ("SL", "TRAIL_SL")
    assert trader.trades[-1]["pnl"] < 0


def test_trailing_stop_short():
    """
    Трейлинг-стоп (short):
      - вход по 100, трейлинг 10%
      - цена падает, SL подтягивается (выше цены, но ниже/вблизи входа)
      - затем рост выше трейлингового SL → reason == "TRAIL_SL"
    """
    trader = PaperTrader(sl_pct=0.0, tp_pct=0.0, trail_pct=0.10, fee_pct=0.0, side="short")
    ts = 50

    trader.on_signal(ts, 100.0, +1)
    assert trader.position is not None
    assert trader.position["sl"] is None  # появится после снижения цены

    # Снижение и закрытие бара ниже — активирует подтягивание SL (short логика)
    trader.check_sl_tp(ts + 1, high=101.0, low=88.0, close=88.0)
    # новый SL = close * (1 + trail_pct) = 88 * 1.1 = 96.8
    assert trader.position["sl"] is not None
    assert pytest.approx(trader.position["sl"], rel=0, abs=1e-9) == 96.8

    # Дальше рост: high >= 96.8 → сработает трейлинговый SL
    trader.check_sl_tp(ts + 2, high=97.0, low=87.0, close=95.0)
    assert trader.position is None
    assert trader.trades[-1]["reason"] == "TRAIL_SL"


# ──────────────────────────────────────────────────────────────────────────────
# Комиссии и нейтральные сигналы
# ──────────────────────────────────────────────────────────────────────────────
def test_fees_are_applied():
    """
    Проверяем, что комиссия учитывается при входе и выходе:
      pnl = (exit - entry)*qty - entry*fee_pct*qty - exit*fee_pct*qty
    """
    trader = PaperTrader(sl_pct=0.0, tp_pct=0.0, fee_pct=0.01)  # 1% комиссия
    ts = 1

    trader.on_signal(ts, 100.0, +1)
    trader.close(ts + 1, 110.0, reason="signal")

    last = trader.trades[-1]
    # теоретический pnl: 10 - (100*0.01) - (110*0.01) = 10 - 1 - 1.1 = 7.9
    assert pytest.approx(last["pnl"], rel=0, abs=1e-9) == 7.9


def test_signal_zero_is_ignored_and_reset_works():
    """
    Проверяем, что сигнал 0 игнорируется и reset() очищает состояние.
    """
    trader = PaperTrader(sl_pct=0.1, tp_pct=0.1, fee_pct=0.0)

    # сигнал 0 не должен открыть позицию
    trader.on_signal(1, 100.0, 0)
    assert trader.position is None
    assert trader.trades == []

    # откроем и закроем сделку, затем сбросим
    trader.on_signal(2, 100.0, +1)
    trader.check_sl_tp(3, high=112.0, low=95.0)  # TP
    assert trader.position is None
    assert len(trader.trades) == 1

    # reset()
    trader.reset()
    assert trader.position is None
    assert trader.trades == []
