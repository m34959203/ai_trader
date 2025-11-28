# src/indicators.py
from __future__ import annotations

"""
Набор векторных индикаторов для анализа рынка.

# Общий контракт
- **Вход**: pandas.Series (или массивоподобное), индекс сохраняется как есть.
- **Выход**:
  - Скользящие/RSI/ATR/… → `pd.Series` float64 с тем же индексом.
  - Составные индикаторы (MACD/BB/ADX/Стохастик) → `pd.DataFrame` с фиксированными колонками.
- **Типы**: все вычисления выполняются в float64; вход приводится к `float64` через `pd.to_numeric(..., errors="coerce")`.
- **Разогрев**:
  - Индикаторы, использующие скользящее окно (`sma`, `wma`, `bollinger_bands`, `stochastic`) возвращают `NaN`
    до накопления достаточного количества баров (`min_periods=period`).
  - EMA/RSI/ATR и др. с EWM/Wilder-сглаживанием возвращают значения с первого бара
    (стартовая инициализация по формуле сглаживания). Это *ожидаемое поведение*.
  - Там, где необходимо бинарное решение (кроссы), `NaN` трактуется как отсутствие сигнала (`False`).
- **Без побочных эффектов**: входные объекты не модифицируются.

Индикаторы ориентированы на стабильную работу в ресемплинге/бэктестах и аккуратно обращаются
с короткими сериями (валидируют периоды и не падают при частичной истории).
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    # MAs
    "ema",
    "sma",
    "wma",
    # Momentum / Volatility
    "rsi",
    "macd",
    "bollinger_bands",
    "true_range",
    "atr",
    "stochastic",
    "adx",
    "force_index",
    "money_flow_index",
    "on_balance_volume",
    "awesome_oscillator",
    # Advanced indicators
    "ichimoku",
    "vwap",
    # Candles / OHLC helpers
    "heikin_ashi",
    "candlestick_patterns",
    "resample_ohlcv",
    # Cross helpers
    "cross_over",
    "cross_under",
    "cross_level",
]

# ─────────────────────────────
# Базовые скользящие
# ─────────────────────────────
def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Экспоненциальная скользящая средняя.

    Parameters
    ----------
    series : pd.Series
        Ряд значений (обычно цена закрытия).
    period : int
        Период EMA (>0). Используется `adjust=False` (Wilder-стиль).

    Returns
    -------
    pd.Series
        EMA в float64 с сохранением индекса. Значения доступны с первого бара
        (разогрев не возвращает NaN из-за природы EWM).

    Raises
    ------
    ValueError
        Если `period <= 0`.
    """
    s = _as_float_series(series)
    if period <= 0:
        raise ValueError("ema: period must be > 0")
    return s.ewm(span=period, adjust=False).mean()


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Простая скользящая средняя.

    - Возвращает NaN до накопления `period` наблюдений.

    Parameters
    ----------
    series : pd.Series
    period : int

    Returns
    -------
    pd.Series
        SMA (float64).

    Raises
    ------
    ValueError
        Если `period <= 0`.
    """
    s = _as_float_series(series)
    if period <= 0:
        raise ValueError("sma: period must be > 0")
    return s.rolling(window=period, min_periods=period).mean()


def wma(series: pd.Series, period: int) -> pd.Series:
    """
    Взвешенная скользящая средняя (Weights = 1..period).

    - Возвращает NaN до накопления `period` наблюдений.

    Parameters
    ----------
    series : pd.Series
    period : int

    Returns
    -------
    pd.Series
        WMA (float64).

    Raises
    ------
    ValueError
        Если `period <= 0`.
    """
    s = _as_float_series(series)
    if period <= 0:
        raise ValueError("wma: period must be > 0")
    w = np.arange(1, period + 1, dtype=float)
    return s.rolling(period, min_periods=period).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


# ─────────────────────────────
# RSI (Wilder)
# ─────────────────────────────
def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Индекс относительной силы (RSI) по Уайлдеру.

    - Сглаживание выполняется через EWM с `alpha=1/period`.
    - Для удобства downstream-логики начальные `NaN` заменяются на 0.0.

    Parameters
    ----------
    series : pd.Series
        Цена (обычно close).
    period : int, default 14

    Returns
    -------
    pd.Series
        RSI в диапазоне [0, 100] (float64). Первые значения могут быть 0.0 из-за инициализации.

    Raises
    ------
    ValueError
        Если `period <= 0`.
    """
    s = _as_float_series(series)
    if period <= 0:
        raise ValueError("rsi: period must be > 0")

    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)

    # Wilder smoothing: alpha = 1/period
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()

    rs = roll_up / (roll_down.replace(0.0, np.nan))
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.fillna(0.0)


# ─────────────────────────────
# MACD
# ─────────────────────────────
def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Moving Average Convergence/Divergence (MACD).

    Parameters
    ----------
    series : pd.Series
    fast : int, default 12
    slow : int, default 26
    signal : int, default 9

    Returns
    -------
    pd.DataFrame
        Колонки:
        - `macd`   — разность EMA(fast) и EMA(slow)
        - `signal` — EMA(macd, signal)
        - `hist`   — macd - signal

        Значения доступны с первых баров (EWM).

    Raises
    ------
    ValueError
        Если любой из периодов <= 0.
    """
    if fast <= 0 or slow <= 0 or signal <= 0:
        raise ValueError("macd: periods must be > 0")
    s = _as_float_series(series)

    ema_fast = ema(s, fast)
    ema_slow = ema(s, slow)
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line

    out = pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": hist},
        index=s.index,
    )
    return out


# ─────────────────────────────
# Полосы Боллинджера
# ─────────────────────────────
def bollinger_bands(series: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.DataFrame:
    """
    Полосы Боллинджера.

    - Возвращает NaN до накопления `period` наблюдений.

    Parameters
    ----------
    series : pd.Series
    period : int, default 20
    n_std : float, default 2.0
        Кол-во стандартных отклонений для верхней/нижней полос.

    Returns
    -------
    pd.DataFrame
        Колонки:
        - `bb_mid`       — SMA(period)
        - `bb_upper`     — mid + n_std * std
        - `bb_lower`     — mid - n_std * std
        - `bb_width`     — (upper - lower) / mid
        - `bb_percent_b` — (price - lower) / (upper - lower)

    Raises
    ------
    ValueError
        Если `period <= 0` или `n_std <= 0`.
    """
    if period <= 0 or n_std <= 0:
        raise ValueError("bollinger_bands: period and n_std must be > 0")

    s = _as_float_series(series)
    mid = sma(s, period)
    std = s.rolling(period, min_periods=period).std(ddof=0)

    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid.replace(0.0, np.nan)
    # %B: положение цены относительно каналов
    percent_b = (s - lower) / (upper - lower)

    out = pd.DataFrame(
        {
            "bb_mid": mid,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_width": width,
            "bb_percent_b": percent_b,
        },
        index=s.index,
    )
    return out


# ─────────────────────────────
# ATR / True Range
# ─────────────────────────────
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    True Range.

    TR = max(
        high - low,
        |high - prev_close|,
        |low - prev_close|
    )

    Parameters
    ----------
    high, low, close : pd.Series

    Returns
    -------
    pd.Series
        True Range (float64). Индекс берётся из входных рядов (ожидается согласованность).
    """
    h, l, c = _as_float_series(high), _as_float_series(low), _as_float_series(close)
    prev_close = c.shift(1)
    tr = pd.concat(
        [
            (h - l),
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range (Wilder).

    - Используется EWM с alpha=1/period (значения доступны с первых баров).

    Parameters
    ----------
    high, low, close : pd.Series
    period : int, default 14

    Returns
    -------
    pd.Series
        ATR (float64).

    Raises
    ------
    ValueError
        Если `period <= 0`.
    """
    if period <= 0:
        raise ValueError("atr: period must be > 0")
    tr = true_range(high, low, close)
    # Wilder smoothing
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ─────────────────────────────
# Стохастик
# ─────────────────────────────
def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> pd.DataFrame:
    """
    Стохастический осциллятор.

    %K = SMA( (close - LL) / (HH - LL) * 100, smooth_k ), где LL/HH — экстремумы за `k_period`  
    %D = SMA(%K, d_period)

    - Возвращает NaN до накопления соответствующих периодов.
    - Деление на ноль при (HH==LL) корректно обрабатывается как NaN.

    Parameters
    ----------
    high, low, close : pd.Series
    k_period : int, default 14
    d_period : int, default 3
    smooth_k : int, default 3

    Returns
    -------
    pd.DataFrame
        Колонки: `stoch_k`, `stoch_d` (в процентах 0..100, где применимо).

    Raises
    ------
    ValueError
        Если любой период <= 0.
    """
    if min(k_period, d_period, smooth_k) <= 0:
        raise ValueError("stochastic: all periods must be > 0")

    h, l, c = _as_float_series(high), _as_float_series(low), _as_float_series(close)

    lowest_low = l.rolling(k_period, min_periods=k_period).min()
    highest_high = h.rolling(k_period, min_periods=k_period).max()
    denom = (highest_high - lowest_low).replace(0.0, np.nan)

    fast_k = ((c - lowest_low) / denom) * 100.0
    stoch_k = fast_k.rolling(smooth_k, min_periods=smooth_k).mean()
    stoch_d = stoch_k.rolling(d_period, min_periods=d_period).mean()

    return pd.DataFrame({"stoch_k": stoch_k, "stoch_d": stoch_d}, index=c.index)


# ─────────────────────────────
# ADX (Average Directional Index, Wilder)
# ─────────────────────────────
def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.DataFrame:
    """
    Индикатор направленного движения ADX.

    Вычисляет:
      - `pdi` (+DI, %)  — сила положительного направления
      - `ndi` (-DI, %)  — сила отрицательного направления
      - `adx` (0..100)  — общая сила тренда (сглаженный DX)

    Периоды сглаживания соответствуют Уайлдеру (`alpha=1/period`).

    Parameters
    ----------
    high, low, close : pd.Series
    period : int, default 14

    Returns
    -------
    pd.DataFrame
        Колонки: `pdi`, `ndi`, `adx` (float64).

    Raises
    ------
    ValueError
        Если `period <= 0`.
    """
    if period <= 0:
        raise ValueError("adx: period must be > 0")

    h, l, c = _as_float_series(high), _as_float_series(low), _as_float_series(close)

    up_move = h.diff()
    down_move = -l.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(h, l, c)

    atr_w = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_dm_w = pd.Series(plus_dm, index=h.index).ewm(alpha=1 / period, adjust=False).mean()
    minus_dm_w = pd.Series(minus_dm, index=h.index).ewm(alpha=1 / period, adjust=False).mean()

    pdi = 100.0 * (plus_dm_w / atr_w.replace(0.0, np.nan))
    ndi = 100.0 * (minus_dm_w / atr_w.replace(0.0, np.nan))

    dx = (np.abs(pdi - ndi) / (pdi + ndi).replace(0.0, np.nan)) * 100.0
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()

    return pd.DataFrame({"pdi": pdi, "ndi": ndi, "adx": adx_val}, index=h.index)


# ─────────────────────────────
# Volume / Momentum расширения
# ─────────────────────────────
def force_index(close: pd.Series, volume: pd.Series, period: int = 13) -> pd.Series:
    """Elder Force Index с экспоненциальным сглаживанием."""

    c = _as_float_series(close)
    v = _as_float_series(volume)
    raw = c.diff().fillna(0.0) * v.fillna(0.0)
    if period <= 1:
        return raw
    if period <= 0:
        raise ValueError("force_index: period must be > 0")
    return raw.ewm(span=period, adjust=False).mean()


def money_flow_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Money Flow Index (MFI) — объёмный RSI.
    Возвращает значения в диапазоне 0..100.
    """

    if period <= 0:
        raise ValueError("money_flow_index: period must be > 0")

    h = _as_float_series(high)
    l = _as_float_series(low)
    c = _as_float_series(close)
    v = _as_float_series(volume)

    typical = (h + l + c) / 3.0
    mf = typical * v
    prev_typical = typical.shift(1)
    pos = mf.where(typical > prev_typical, 0.0)
    neg = mf.where(typical < prev_typical, 0.0)

    pos_sum = pos.rolling(period, min_periods=period).sum()
    neg_sum = neg.rolling(period, min_periods=period).sum()

    ratio = pos_sum / neg_sum.replace(0.0, np.nan)
    mfi = 100.0 - (100.0 / (1.0 + ratio))
    return mfi.fillna(50.0)


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume (кумулятивный объём)."""

    c = _as_float_series(close)
    v = _as_float_series(volume).fillna(0.0)
    direction = np.sign(c.diff().fillna(0.0))
    direction = pd.Series(direction, index=c.index)
    obv = (direction * v).cumsum().ffill()
    return obv.fillna(0.0)


def awesome_oscillator(high: pd.Series, low: pd.Series, fast: int = 5, slow: int = 34) -> pd.Series:
    """Awesome Oscillator (AO) по Биллу Вильямсу."""

    if fast <= 0 or slow <= 0:
        raise ValueError("awesome_oscillator: periods must be > 0")
    if fast >= slow:
        fast, slow = slow, fast

    median_price = (_as_float_series(high) + _as_float_series(low)) / 2.0
    sma_fast = sma(median_price, fast)
    sma_slow = sma(median_price, slow)
    return (sma_fast - sma_slow).fillna(0.0)


# ─────────────────────────────
# Свечные паттерны / OHLC утилиты
# ─────────────────────────────
def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Строит свечи Heikin-Ashi из обычного OHLC DataFrame.

    Возвращает DataFrame с колонками ha_open/ha_high/ha_low/ha_close.
    """

    required = {"open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"heikin_ashi: missing columns {sorted(missing)}")

    o = _as_float_series(df["open"]).copy()
    h = _as_float_series(df["high"]).copy()
    l = _as_float_series(df["low"]).copy()
    c = _as_float_series(df["close"]).copy()

    if len(o) == 0:
        return pd.DataFrame(columns=["ha_open", "ha_high", "ha_low", "ha_close"], index=o.index)

    ha_close = (o + h + l + c) / 4.0
    ha_open = pd.Series(index=o.index, dtype=float)
    ha_open.iloc[0] = (o.iloc[0] + c.iloc[0]) / 2.0
    for i in range(1, len(o)):
        ha_open.iloc[i] = (ha_open.iloc[i - 1] + ha_close.iloc[i - 1]) / 2.0
    ha_high = pd.concat([h, ha_open, ha_close], axis=1).max(axis=1)
    ha_low = pd.concat([l, ha_open, ha_close], axis=1).min(axis=1)

    return pd.DataFrame(
        {
            "ha_open": ha_open,
            "ha_high": ha_high,
            "ha_low": ha_low,
            "ha_close": ha_close,
        },
        index=df.index,
    )


def candlestick_patterns(
    df: pd.DataFrame,
    *,
    hammer_ratio: float = 2.0,
    star_ratio: float = 2.0,
    doji_frac: float = 0.1,
) -> pd.DataFrame:
    """
    Вычисляет базовые свечные паттерны. Возвращает DataFrame с колонками:
      hammer, inverted_hammer, shooting_star, bullish_engulfing,
      bearish_engulfing, morning_star, evening_star, doji.
    Значения bool.
    """

    required = {"open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"candlestick_patterns: missing columns {sorted(missing)}")

    o = _as_float_series(df["open"])
    h = _as_float_series(df["high"])
    l = _as_float_series(df["low"])
    c = _as_float_series(df["close"])

    body = (c - o)
    body_abs = body.abs()
    upper_shadow = h - pd.concat([o, c], axis=1).max(axis=1)
    lower_shadow = pd.concat([o, c], axis=1).min(axis=1) - l
    range_ = h - l

    hammer = (body > 0) & (lower_shadow >= hammer_ratio * body_abs) & (upper_shadow <= body_abs)
    inverted_hammer = (body > 0) & (upper_shadow >= hammer_ratio * body_abs) & (lower_shadow <= body_abs)
    shooting_star = (body < 0) & (upper_shadow >= star_ratio * body_abs) & (lower_shadow <= body_abs)

    prev_body = body.shift(1)
    prev_open = o.shift(1)
    prev_close = c.shift(1)

    bullish_engulf = (body > 0) & (prev_body < 0) & (o <= prev_close) & (c >= prev_open)
    bearish_engulf = (body < 0) & (prev_body > 0) & (o >= prev_close) & (c <= prev_open)

    doji = body_abs <= (doji_frac * range_).replace(0.0, np.nan)

    # Morning/Evening star (3 свечи)
    gap_down = o > prev_close
    gap_up = o < prev_close
    prev_prev_close = c.shift(2)
    prev_prev_open = o.shift(2)
    morning_star = (
        (prev_body < 0)
        & gap_down
        & (body > 0)
        & (c > prev_prev_open)
        & (prev_prev_close > prev_prev_open)
    )
    evening_star = (
        (prev_body > 0)
        & gap_up
        & (body < 0)
        & (c < prev_prev_open)
        & (prev_prev_close < prev_prev_open)
    )

    return pd.DataFrame(
        {
            "hammer": hammer.fillna(False),
            "inverted_hammer": inverted_hammer.fillna(False),
            "shooting_star": shooting_star.fillna(False),
            "bullish_engulfing": bullish_engulf.fillna(False),
            "bearish_engulfing": bearish_engulf.fillna(False),
            "morning_star": morning_star.fillna(False),
            "evening_star": evening_star.fillna(False),
            "doji": doji.fillna(False),
        },
        index=df.index,
    )


def resample_ohlcv(df: pd.DataFrame, rule: str, *, how: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Ресемплинг OHLCV DataFrame (DatetimeIndex → rule)."""

    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("resample_ohlcv: DataFrame index must be DatetimeIndex")

    agg = how or {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    res = df.resample(rule).agg(agg)
    return res.dropna(subset=["open", "high", "low", "close"], how="all")


# ─────────────────────────────
# Cross utils
# ─────────────────────────────
def cross_over(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """
    Истинный ап-кросс: вчера `fast <= slow`, сегодня `fast > slow`.

    - На барах с NaN возвращает False.

    Parameters
    ----------
    fast, slow : pd.Series

    Returns
    -------
    pd.Series
        Булев ряд (dtype=bool), где True — точка кросса вверх.
    """
    f = _as_float_series(fast)
    s = _as_float_series(slow)
    prev = (f.shift(1) <= s.shift(1))
    now = (f > s)
    out = (prev & now)
    return out.fillna(False)


def cross_under(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """
    Истинный даун-кросс: вчера `fast >= slow`, сегодня `fast < slow`.

    - На барах с NaN возвращает False.
    """
    f = _as_float_series(fast)
    s = _as_float_series(slow)
    prev = (f.shift(1) >= s.shift(1))
    now = (f < s)
    out = (prev & now)
    return out.fillna(False)


def cross_level(series: pd.Series, level: float) -> Tuple[pd.Series, pd.Series]:
    """
    Пересечения горизонтального уровня.

    Parameters
    ----------
    series : pd.Series
    level : float

    Returns
    -------
    (pd.Series, pd.Series)
        Кортеж (cross_up, cross_down), булевы серии с тем же индексом.
        - `cross_up`: вчера ≤ level, сегодня > level
        - `cross_down`: вчера ≥ level, сегодня < level

    Notes
    -----
    На барах с NaN возвращает False.
    """
    s = _as_float_series(series)
    lvl = float(level)
    up = (s > lvl) & (s.shift(1) <= lvl)
    dn = (s < lvl) & (s.shift(1) >= lvl)
    return up.fillna(False), dn.fillna(False)


# ─────────────────────────────
# Advanced Indicators
# ─────────────────────────────
def ichimoku(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    """
    Ichimoku Kinko Hyo (Облако Ишимоку) - комплексный индикатор тренда.

    Используется 60% профессиональных трейдеров, особенно эффективен для криптовалют
    и азиатских рынков. Предоставляет полную картину: тренд, моментум, поддержку/сопротивление.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC данные с колонками 'high', 'low', 'close'
    tenkan_period : int, default 9
        Период для Tenkan-sen (линия конверсии) - быстрая линия
    kijun_period : int, default 26
        Период для Kijun-sen (базовая линия) - медленная линия
    senkou_b_period : int, default 52
        Период для Senkou Span B (ведущая линия B)
    displacement : int, default 26
        Смещение облака вперёд (обычно = kijun_period)

    Returns
    -------
    pd.DataFrame
        Колонки:
        - `tenkan_sen`     - Линия конверсии: (max9 + min9) / 2
        - `kijun_sen`      - Базовая линия: (max26 + min26) / 2
        - `senkou_span_a`  - Ведущая линия A: (tenkan + kijun) / 2, сдвинута на +26
        - `senkou_span_b`  - Ведущая линия B: (max52 + min52) / 2, сдвинута на +26
        - `chikou_span`    - Запаздывающая линия: close, сдвинута на -26

    Notes
    -----
    Торговые сигналы:
    - STRONG BUY: цена > облака AND tenkan > kijun AND облако зелёное (span_a > span_b)
    - STRONG SELL: цена < облака AND tenkan < kijun AND облако красное (span_a < span_b)
    - Пробой облака - сильный сигнал изменения тренда
    - Chikou выше цены 26 баров назад - подтверждение восходящего тренда

    Examples
    --------
    >>> df = pd.DataFrame({'high': [...], 'low': [...], 'close': [...]})
    >>> ichimoku_df = ichimoku(df)
    >>> # Проверка бычьего сигнала
    >>> bullish = (
    ...     (df['close'] > ichimoku_df['senkou_span_a']) &
    ...     (df['close'] > ichimoku_df['senkou_span_b']) &
    ...     (ichimoku_df['tenkan_sen'] > ichimoku_df['kijun_sen']) &
    ...     (ichimoku_df['senkou_span_a'] > ichimoku_df['senkou_span_b'])
    ... )
    """
    if not all(col in df.columns for col in ['high', 'low', 'close']):
        raise ValueError("ichimoku: DataFrame must contain 'high', 'low', 'close' columns")

    high = _as_float_series(df['high'])
    low = _as_float_series(df['low'])
    close = _as_float_series(df['close'])

    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    tenkan_sen = (
        high.rolling(window=tenkan_period, min_periods=tenkan_period).max() +
        low.rolling(window=tenkan_period, min_periods=tenkan_period).min()
    ) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    kijun_sen = (
        high.rolling(window=kijun_period, min_periods=kijun_period).max() +
        low.rolling(window=kijun_period, min_periods=kijun_period).min()
    ) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted forward +26
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted forward +26
    senkou_span_b = (
        (
            high.rolling(window=senkou_b_period, min_periods=senkou_b_period).max() +
            low.rolling(window=senkou_b_period, min_periods=senkou_b_period).min()
        ) / 2
    ).shift(displacement)

    # Chikou Span (Lagging Span): Close shifted backward -26
    chikou_span = close.shift(-displacement)

    return pd.DataFrame({
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_span_a': senkou_span_a,
        'senkou_span_b': senkou_span_b,
        'chikou_span': chikou_span,
    }, index=df.index)


def vwap(
    df: pd.DataFrame,
    session_start_hour: int = 0,
    typical_price: bool = True,
) -> pd.Series:
    """
    VWAP (Volume-Weighted Average Price) - институциональный бенчмарк.

    Ключевой индикатор для внутридневной торговли. Институциональные трейдеры
    используют VWAP для оценки качества исполнения ордеров:
    - Покупка ниже VWAP = хорошее исполнение
    - Продажа выше VWAP = хорошее исполнение

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV данные с колонками 'high', 'low', 'close', 'volume'
        Опционально: 'timestamp' для определения сессий
    session_start_hour : int, default 0
        Час начала торговой сессии (UTC). VWAP сбрасывается каждый день.
        0 = полночь UTC (стандарт для крипты)
        9 = 9:00 UTC (для традиционных рынков)
    typical_price : bool, default True
        True: использовать типичную цену (H+L+C)/3
        False: использовать только close

    Returns
    -------
    pd.Series
        VWAP значения. Сбрасывается каждую сессию.

    Notes
    -----
    Торговые сигналы:
    - Цена пересекает VWAP снизу вверх → BUY сигнал
    - Цена пересекает VWAP сверху вниз → SELL сигнал
    - Цена > VWAP → рынок в восходящем тренде
    - Цена < VWAP → рынок в нисходящем тренде

    Институциональное применение:
    - Алгоритмы TWAP/VWAP используются для крупных ордеров
    - Отклонение от VWAP показывает силу покупателей/продавцов
    - VWAP - уровень справедливой цены для данной сессии

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'high': [...], 'low': [...], 'close': [...], 'volume': [...]
    ... })
    >>> df['vwap'] = vwap(df)
    >>> # BUY сигнал: цена пересекает VWAP снизу
    >>> buy_signal = (df['close'] > df['vwap']) & (df['close'].shift(1) <= df['vwap'].shift(1))
    """
    required_cols = ['high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"vwap: DataFrame must contain {required_cols}")

    df_copy = df.copy()

    # Calculate typical price or use close
    if typical_price:
        price = (
            _as_float_series(df_copy['high']) +
            _as_float_series(df_copy['low']) +
            _as_float_series(df_copy['close'])
        ) / 3
    else:
        price = _as_float_series(df_copy['close'])

    volume = _as_float_series(df_copy['volume'])

    # Determine session boundaries
    # If index is DatetimeIndex, use it; otherwise create simple incrementing sessions
    if isinstance(df_copy.index, pd.DatetimeIndex):
        # Extract hour from index
        df_copy['hour'] = df_copy.index.hour
        # New session when hour equals session_start_hour
        df_copy['new_session'] = (df_copy['hour'] == session_start_hour).astype(int)
        # Cumulative sum to create session groups
        df_copy['session'] = df_copy['new_session'].cumsum()
    else:
        # For non-datetime index, treat each day as separate session
        # This is a fallback; assume data is already sorted by time
        # Create session based on index position (every N rows = new session)
        # For simplicity, use a single session for all data
        df_copy['session'] = 0

    # Calculate VWAP for each session
    # VWAP = Σ(Price × Volume) / Σ(Volume)
    pv = price * volume  # Price × Volume

    # Cumulative sums within each session
    cumsum_pv = pv.groupby(df_copy['session']).cumsum()
    cumsum_volume = volume.groupby(df_copy['session']).cumsum()

    # VWAP = cumulative PV / cumulative Volume
    vwap_values = cumsum_pv / cumsum_volume

    # Handle division by zero
    vwap_values = vwap_values.replace([np.inf, -np.inf], np.nan)

    return pd.Series(vwap_values.values, index=df.index, name='vwap')


# ─────────────────────────────
# Вспомогательные
# ─────────────────────────────
def _as_float_series(s: pd.Series) -> pd.Series:
    """
    Приводит вход к `pd.Series` float64 и сохраняет индекс.

    - Нечисловые значения конвертируются в NaN.
    - Входные данные не модифицируются.

    Parameters
    ----------
    s : pd.Series | array-like

    Returns
    -------
    pd.Series
        float64 Series с тем же индексом (если вход был Series).
    """
    if not isinstance(s, pd.Series):
        s = pd.Series(s)
    return pd.to_numeric(s, errors="coerce").astype("float64")
