# tests/test_analyze_market.py
from __future__ import annotations

"""
Тесты для высокоуровневого анализа рынка (Спринт 2).

Фокус:
- Контракт выходных данных (схема/домены значений)
- Нематеряемость входного df
- Корректная работа с минимальными данными и NaN на «разогреве»
- Возможность безопасного переопределения порогов через frozen-конфиг
- Регрессия по уровням: штраф при близости к сопротивлению/поддержке, отсутствие штрафа
"""

from dataclasses import replace
import dataclasses
import math
from typing import Dict, Any

import numpy as np
import pandas as pd
import pytest

# Модуль анализа из спринта 2
from src.analysis.analyze_market import (
    analyze_market,
    DEFAULT_CONFIG,
    AnalysisConfig,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_df(bars: int, *, freq: str = "H", seed: int = 7) -> pd.DataFrame:
    """
    Генерируем синтетические OHLCV с DatetimeIndex (UTC).
    Минимально «похожая» серия: случайное блуждание + шум.
    """
    assert bars > 0, "bars must be positive"
    rng = pd.date_range(
        end=pd.Timestamp.now(tz="UTC"),   # FIX: безопасная tz-aware метка времени
        periods=bars,
        freq=freq,
    )
    rs = np.random.RandomState(seed)
    price = 100 + rs.randn(bars).cumsum() * 0.5
    high = price + np.abs(rs.randn(bars) * 0.6)
    low = price - np.abs(rs.randn(bars) * 0.6)
    df = pd.DataFrame(
        {
            "open": price + rs.randn(bars) * 0.2,
            "high": high,
            "low": low,
            "close": price + rs.randn(bars) * 0.2,
            "volume": rs.randint(1_000, 5_000, size=bars),
        },
        index=rng,
    )
    # гарантируем валидный формат
    df.index = df.index.tz_convert("UTC")
    return df


def _assert_basic_schema(out: Dict[str, Any]) -> None:
    """Унифицированные проверки контракта результата analyze_market."""
    # обязательные ключи
    for key in ("trend", "volatility", "signal", "confidence", "reasons", "levels"):
        assert key in out, f"missing key: {key}"

    # домены значений
    assert out["trend"] in {"up", "down", "sideways"}
    assert out["volatility"] in {"low", "medium", "high"}
    assert out["signal"] in {"buy", "sell", "flat"}

    # confidence — целое 0..100
    assert isinstance(out["confidence"], int)
    assert 0 <= out["confidence"] <= 100

    # reasons — список строк
    assert isinstance(out["reasons"], list)
    assert all(isinstance(x, str) for x in out["reasons"])

    # levels — список словарей с полями kind/price/strength
    assert isinstance(out["levels"], list)
    for lvl in out["levels"]:
        assert set(lvl.keys()) == {"kind", "price", "strength"}
        assert lvl["kind"] in {"support", "resistance"}
        assert isinstance(lvl["price"], (int, float)) and math.isfinite(lvl["price"])
        assert isinstance(lvl["strength"], int) and lvl["strength"] >= 1

    # не более 8 уровней по контракту
    assert len(out["levels"]) <= 8


def _make_trend_df(
    bars: int,
    *,
    up: bool = True,
    freq: str = "H",
    slope: float = 0.25,
    noise: float = 0.05,
) -> pd.DataFrame:
    """
    Детминистическая серия с явным трендом (для устойчивых MAs/MACD).
    Также создаём локальный пивот high/low ближе к концу ряда для формирования уровня.
    """
    assert bars >= 220, "для стабильности индикаторов и фракталов нужно >=220 баров"
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=bars, freq=freq)  # FIX

    base = 100.0
    direction = 1.0 if up else -1.0
    close = base + direction * slope * np.arange(bars)
    # лёгкий шум
    rng = np.random.RandomState(123 if up else 321)
    close = close + rng.randn(bars) * noise

    # ohlc: небольшой спред, чтобы ATR был стабильным и не нулевым
    high = close + 0.25 + np.abs(rng.randn(bars) * 0.02)
    low = close - 0.25 - np.abs(rng.randn(bars) * 0.02)
    open_ = close + rng.randn(bars) * 0.03
    vol = rng.randint(1_000, 5_000, size=bars)

    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)

    # Вставим явный пивот:
    pivot_i = bars - 10  # будет за 10 баров до конца
    if up:
        # локальный максимум (resistance)
        df.loc[df.index[pivot_i], "high"] = df["high"].iloc[pivot_i - 2:pivot_i + 3].max() + 1.0
    else:
        # локальный минимум (support)
        df.loc[df.index[pivot_i], "low"] = df["low"].iloc[pivot_i - 2:pivot_i + 3].min() - 1.0

    return df, pivot_i


# ──────────────────────────────────────────────────────────────────────────────
# Базовые тесты из чек-листа спринта 2
# ──────────────────────────────────────────────────────────────────────────────

def test_default_config_invariants():
    """DEFAULT_CONFIG должен быть экземпляром AnalysisConfig и не мутируемым."""
    assert isinstance(DEFAULT_CONFIG, AnalysisConfig)
    # dataclass(frozen=True) → попытка записи должна упасть
    with pytest.raises(dataclasses.FrozenInstanceError):
        DEFAULT_CONFIG.ema_fast = 21  # Обычное присваивание


def test_raises_on_not_enough_bars():
    """Если баров меньше min_bars_fast — ожидаем AssertionError с понятным текстом."""
    cfg = DEFAULT_CONFIG
    few = _make_df(cfg.min_bars_fast - 1)
    with pytest.raises(AssertionError) as e:
        _ = analyze_market(few)  # noqa: F841
    assert "Need at least" in str(e.value)


def test_basic_output_schema_and_bounds():
    """Проверяем ключи, типы и диапазоны confidence/levels без допущений о самом сигнале."""
    cfg = DEFAULT_CONFIG
    df = _make_df(cfg.min_bars_fast + 40)  # запас для тёплого старта индикаторов

    out = analyze_market(df)
    _assert_basic_schema(out)


def test_mtf_key_when_slow_df_passed():
    """При передаче df_4h ожидаем появление блока mtf с полем trend_4h."""
    cfg = DEFAULT_CONFIG
    fast = _make_df(cfg.min_bars_fast + 40, freq="H", seed=11)
    slow = _make_df(cfg.min_bars_slow + 20, freq="4H", seed=13)

    out = analyze_market(fast, slow)

    assert "mtf" in out, "mtf block should be present when df_4h is provided"
    assert isinstance(out["mtf"], dict)
    assert out["mtf"].get("trend_4h") in {"up", "down", "sideways"}


def test_input_df_not_mutated():
    """Функция не должна мутировать входной DataFrame (по крайней мере по значениям close)."""
    cfg = DEFAULT_CONFIG
    df = _make_df(cfg.min_bars_fast + 40, freq="H", seed=17)
    # снимем копию столбца для сравнения
    close_before = df["close"].copy()
    _ = analyze_market(df)
    # индексы/значения не должны измениться
    pd.testing.assert_index_equal(df.index, close_before.index)
    pd.testing.assert_series_equal(df["close"], close_before, check_names=True)


def test_accepts_explicit_config_override():
    """Параметры порогов можно безопасно переопределять копией конфига (frozen dataclass → replace)."""
    cfg = replace(DEFAULT_CONFIG, buy_threshold=55, sell_threshold=45)
    df = _make_df(cfg.min_bars_fast + 40, freq="H", seed=23)
    out = analyze_market(df, config=cfg)
    _assert_basic_schema(out)
    assert out["signal"] in {"buy", "sell", "flat"}  # корректное значение


def test_handles_leading_nans_gracefully():
    """
    На «разогреве» индикаторов нередко встречаются NaN — убеждаемся, что analyze_market
    устойчив к нескольким NaN в начале ряда.
    """
    cfg = DEFAULT_CONFIG
    bars = cfg.min_bars_fast + 50
    df = _make_df(bars, freq="H", seed=29).copy()

    # Внесём NaN в первые 5 баров (типичный тёплый старт индикаторов)
    for col in ("open", "high", "low", "close", "volume"):
        df.loc[df.index[:5], col] = np.nan

    out = analyze_market(df)  # не должно кидать исключений
    _assert_basic_schema(out)


# ──────────────────────────────────────────────────────────────────────────────
# Регрессионные тесты уровней (штраф за близость)
# ──────────────────────────────────────────────────────────────────────────────

def test_level_penalty_buy_near_resistance_vs_far():
    """
    Сценарий BUY: цена близко к сопротивлению → штраф (меньше confidence и причина в reasons).
    Цена далеко → штрафа нет, confidence выше.
    """
    base_cfg = DEFAULT_CONFIG
    bars = max(base_cfg.min_bars_fast + 60, 240)

    df, pivot_i = _make_trend_df(bars, up=True)

    # Цель: сделать так, чтобы базовое решение было "buy" (восходящий тренд).
    # Усилим базовую оценку, чтобы штраф не "сломал" сам факт buy.
    cfg_far = replace(
        base_cfg,
        base_score=80,               # чтобы гарантированно "buy"
        buy_threshold=60,
        sell_threshold=40,
        level_proximity_mult=0.01,   # очень маленький допуск → штраф почти не сработает
    )
    cfg_near = replace(
        cfg_far,
        level_proximity_mult=5.0,    # очень щедрый допуск по ATR → легко получить штраф
    )

    # Оценим уровень (локальный high на pivot_i). Ставим последнюю цену:
    resistance = float(df["high"].iloc[pivot_i])

    # 1) Далеко от сопротивления (на 5 ATR ниже).
    #    Для устойчивости — просто сильно опустим цену.
    df_far = df.copy()
    df_far.iloc[-1, df_far.columns.get_loc("close")] = resistance - 5.0
    df_far.iloc[-1, df_far.columns.get_loc("open")] = df_far["close"].iloc[-1] - 0.05
    df_far.iloc[-1, df_far.columns.get_loc("high")] = df_far["close"].iloc[-1] + 0.3
    df_far.iloc[-1, df_far.columns.get_loc("low")] = df_far["close"].iloc[-1] - 0.3

    out_far = analyze_market(df_far, config=cfg_far)
    _assert_basic_schema(out_far)
    assert out_far["signal"] in {"buy", "flat"}  # при сильном тренде чаще buy

    # 2) Очень близко к сопротивлению (на 0.05 ниже).
    df_near = df.copy()
    df_near.iloc[-1, df_near.columns.get_loc("close")] = resistance - 0.05
    df_near.iloc[-1, df_near.columns.get_loc("open")] = df_near["close"].iloc[-1] - 0.02
    df_near.iloc[-1, df_near.columns.get_loc("high")] = df_near["close"].iloc[-1] + 0.25
    df_near.iloc[-1, df_near.columns.get_loc("low")] = df_near["close"].iloc[-1] - 0.25

    out_near = analyze_market(df_near, config=cfg_near)
    _assert_basic_schema(out_near)

    # Проверяем: confidence должен быть ниже, и в reasons появится про близость к сопротивлению
    assert out_near["confidence"] <= out_far["confidence"]
    assert any("Близко к сопротивлению" in r for r in out_near["reasons"])


def test_level_penalty_sell_near_support_vs_far():
    """
    Сценарий SELL: цена близко к поддержке → штраф (меньше confidence и причина в reasons).
    Цена далеко → штрафа нет.
    """
    base_cfg = DEFAULT_CONFIG
    bars = max(base_cfg.min_bars_fast + 60, 240)

    df, pivot_i = _make_trend_df(bars, up=False)

    cfg_far = replace(
        base_cfg,
        base_score=80,               # базово "sell" обеспечим (вниз тренд + высокий базовый скор)
        buy_threshold=60,
        sell_threshold=40,
        level_proximity_mult=0.01,   # маленький допуск → штраф не сработает
    )
    cfg_near = replace(
        cfg_far,
        level_proximity_mult=5.0,    # большой допуск → легко поймаем штраф
    )

    support = float(df["low"].iloc[pivot_i])

    # 1) Далеко от поддержки (на 5.0 выше)
    df_far = df.copy()
    df_far.iloc[-1, df_far.columns.get_loc("close")] = support + 5.0
    df_far.iloc[-1, df_far.columns.get_loc("open")] = df_far["close"].iloc[-1] + 0.05
    df_far.iloc[-1, df_far.columns.get_loc("high")] = df_far["close"].iloc[-1] + 0.3
    df_far.iloc[-1, df_far.columns.get_loc("low")] = df_far["close"].iloc[-1] - 0.3

    out_far = analyze_market(df_far, config=cfg_far)
    _assert_basic_schema(out_far)
    assert out_far["signal"] in {"sell", "flat"}

    # 2) Очень близко к поддержке (на 0.05 выше)
    df_near = df.copy()
    df_near.iloc[-1, df_near.columns.get_loc("close")] = support + 0.05
    df_near.iloc[-1, df_near.columns.get_loc("open")] = df_near["close"].iloc[-1] + 0.02
    df_near.iloc[-1, df_near.columns.get_loc("high")] = df_near["close"].iloc[-1] + 0.25
    df_near.iloc[-1, df_near.columns.get_loc("low")] = df_near["close"].iloc[-1] - 0.25

    out_near = analyze_market(df_near, config=cfg_near)
    _assert_basic_schema(out_near)

    assert out_near["confidence"] <= out_far["confidence"]
    assert any("Близко к поддержке" in r for r in out_near["reasons"])


def test_no_level_penalty_when_far_even_with_large_tolerance():
    """
    Если цена достаточно далеко от ближайшего уровня, даже при большом level_proximity_mult
    штраф не применяется (в reasons нет «Близко к…»).
    """
    base_cfg = DEFAULT_CONFIG
    bars = max(base_cfg.min_bars_fast + 60, 240)

    df, pivot_i = _make_trend_df(bars, up=True)
    resistance = float(df["high"].iloc[pivot_i])

    cfg = replace(
        base_cfg,
        base_score=80,
        buy_threshold=60,
        sell_threshold=40,
        level_proximity_mult=3.0,  # большой допуск
    )

    # Уйдём далеко от сопротивления
    df_far = df.copy()
    df_far.iloc[-1, df_far.columns.get_loc("close")] = resistance - 10.0
    df_far.iloc[-1, df_far.columns.get_loc("high")] = df_far["close"].iloc[-1] + 0.3
    df_far.iloc[-1, df_far.columns.get_loc("low")] = df_far["close"].iloc[-1] - 0.3

    out = analyze_market(df_far, config=cfg)
    _assert_basic_schema(out)
    assert not any("Близко к сопротивлению" in r or "Близко к поддержке" in r for r in out["reasons"])