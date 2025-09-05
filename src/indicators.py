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

from typing import Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    # MAs
    "ema", "sma", "wma",
    # Momentum / Volatility
    "rsi", "macd", "bollinger_bands", "true_range", "atr", "stochastic", "adx",
    # Cross helpers
    "cross_over", "cross_under", "cross_level",
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
