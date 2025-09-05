from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Sequence, Dict, Any, Tuple

from .indicators import ema, cross_over, cross_under


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные утилиты времени
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Унифицируем время и гарантируем наличие И ts, И timestamp:
      • если есть 'ts' (unix int) — нормализуем → int64; 'timestamp' создаём при необходимости
      • если есть только 'timestamp' — создаём 'ts' из неё
      • если нет ни одной — создаём монотонный int64 счётчик
    Оба поля возвращаем как int64 (Unix seconds).
    """
    out = df.copy()

    def _to_unix_seconds(x: pd.Series) -> pd.Series:
        # Пытаемся аккуратно привести к секундам Unix
        if np.issubdtype(x.dtype, np.number):
            # уже число — считаем что это секунды
            s = pd.to_numeric(x, errors="coerce").astype("Int64").dropna().astype(np.int64)
            return s
        # datetime-like → в секунды
        dt = pd.to_datetime(x, utc=True, errors="coerce")
        # .view('int64') в нс → // 1e9
        return (dt.view("int64") // 1_000_000_000).astype("Int64").dropna().astype(np.int64)

    has_ts = "ts" in out.columns
    has_t = "timestamp" in out.columns

    if has_ts:
        out["ts"] = _to_unix_seconds(out["ts"])
        if not has_t:
            out["timestamp"] = out["ts"].astype(np.int64)
        else:
            # Нормализуем и timestamp
            out["timestamp"] = _to_unix_seconds(out["timestamp"])
    elif has_t:
        out["timestamp"] = _to_unix_seconds(out["timestamp"])
        out["ts"] = out["timestamp"].astype(np.int64)
    else:
        # ни одной временной — создадим монотонный индекс
        n = len(out)
        out["ts"] = np.arange(n, dtype=np.int64)
        out["timestamp"] = out["ts"].astype(np.int64)

    # вычищаем возможные NaN и рассинхрон
    out = out.dropna(subset=["ts"]).copy()
    out["ts"] = out["ts"].astype(np.int64)
    if "timestamp" not in out.columns:
        out["timestamp"] = out["ts"].astype(np.int64)
    else:
        out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce").fillna(out["ts"]).astype(np.int64)

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Простая оценка режима рынка (для фильтрации сигналов)
# ──────────────────────────────────────────────────────────────────────────────

def market_regime_flags(
    close: pd.Series,
    *,
    vol_window: int = 50,
    gap_series: Optional[pd.Series] = None,
    gap_quantile_trend: float = 0.6,
    vol_quantiles: Tuple[float, float] = (0.3, 0.7),
) -> pd.DataFrame:
    """
    Возвращает DataFrame с колонками:
      - vol (rolling std доходностей)
      - q_lo, q_hi (пороги волатильности по квантили)
      - is_flat: низкая волатильность (vol <= q_lo)
      - is_turbulent: высокая волатильность (vol >= q_hi)
      - is_trend: если |gap| выше своей квантили и не «турбулент»
    """
    s = pd.Series(close).astype(float)
    ret = s.pct_change().fillna(0.0)
    vol = ret.rolling(vol_window, min_periods=max(5, vol_window // 3)).std().bfill().fillna(0.0)

    # квантильные пороги волатильности
    q_lo = float(vol.quantile(vol_quantiles[0])) if len(vol) else 0.0
    q_hi = float(vol.quantile(vol_quantiles[1])) if len(vol) else 0.0

    is_flat = vol <= q_lo
    is_turbulent = vol >= q_hi

    # трендовость по величине расхождения EMA (если передали)
    if gap_series is None or len(gap_series) == 0:
        gap_abs = pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    else:
        gap_abs = pd.Series(gap_series, index=s.index).abs().astype(float)

    gap_thr = float(gap_abs.quantile(gap_quantile_trend)) if len(gap_abs) else 0.0
    is_trend = (gap_abs >= gap_thr) & (~is_turbulent)

    out = pd.DataFrame(
        {
            "vol": vol.values,
            "q_lo": q_lo,
            "q_hi": q_hi,
            "is_flat": is_flat.values,
            "is_turbulent": is_turbulent.values,
            "is_trend": is_trend.values,
        }
    )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Антишумовые фильтры: подтверждение и «заморозка» после сигнала
# ──────────────────────────────────────────────────────────────────────────────

def _apply_persistence_and_cooldown(
    df: pd.DataFrame,
    *,
    persist: int = 1,
    cooldown: int = 0,
    min_gap_pct: float = 0.0,
) -> pd.DataFrame:
    """
    Применяет три правила:
      1) persist: сигнал +1 разрешён только если fast>slow держится >= persist баров,
         сигнал -1 — если fast<slow держится >= persist баров
      2) cooldown: после любого ненулевого сигнала запрещаем новые сигналы cooldown баров
      3) min_gap_pct: |ema_fast - ema_slow| / close на баре сигнала должен быть >= порога
    На вход ожидаются колонки: 'ema_fast','ema_slow','close','signal'
    """
    s = df.copy()

    # 1) подтверждение persist
    if persist > 1:
        gt = (s["ema_fast"] > s["ema_slow"]).astype(int)
        lt = (s["ema_fast"] < s["ema_slow"]).astype(int)

        # длина текущей серии > / <
        gt_run = (gt.groupby((gt != gt.shift()).cumsum()).cumcount() + 1) * gt
        lt_run = (lt.groupby((lt != lt.shift()).cumsum()).cumcount() + 1) * lt

        allow_long = gt_run >= persist
        allow_short = lt_run >= persist

        s.loc[(s["signal"] == 1) & (~allow_long), "signal"] = 0
        s.loc[(s["signal"] == -1) & (~allow_short), "signal"] = 0

    # 2) минимальный разрыв между EMA в процентах от цены
    if min_gap_pct > 0.0:
        gap_ok = (s["ema_fast"] - s["ema_slow"]).abs() / s["close"].replace(0.0, np.nan) >= float(min_gap_pct)
        s.loc[~gap_ok.fillna(False), "signal"] = 0

    # 3) cooldown — итеративно «гасим» последующие сигналы
    if cooldown > 0:
        active_cooldown = 0
        sig_col = s.columns.get_loc("signal")
        for i in range(len(s)):
            sig = int(s.iat[i, sig_col])
            if active_cooldown > 0:
                if sig != 0:
                    s.iat[i, sig_col] = 0
                active_cooldown -= 1
            else:
                if sig != 0:
                    active_cooldown = cooldown

    return s


# ──────────────────────────────────────────────────────────────────────────────
# Базовая стратегия: EMA-cross с антишумовыми фильтрами
# ──────────────────────────────────────────────────────────────────────────────

def ema_cross_signals(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    *,
    persist: int = 1,
    cooldown: int = 0,
    min_gap_pct: float = 0.0,
    use_regime_filter: bool = False,
    regime_vol_window: int = 50,
) -> pd.DataFrame:
    """
    Возвращает DataFrame с колонками:
      - ts (int, Unix seconds)
      - timestamp (int, копия ts)
      - close, ema_fast, ema_slow
      - signal ∈ {-1, 0, +1}
    """
    if "close" not in df.columns:
        raise ValueError("ema_cross_signals: требуется колонка 'close'")

    s = _ensure_time_cols(df).copy()
    s["close"] = pd.to_numeric(s["close"], errors="coerce")
    s = s.dropna(subset=["close"])

    s["ema_fast"] = ema(s["close"], fast)
    s["ema_slow"] = ema(s["close"], slow)

    # базовые сырые сигналы пересечений
    s["signal"] = 0
    s.loc[cross_over(s["ema_fast"], s["ema_slow"]), "signal"] = 1
    s.loc[cross_under(s["ema_fast"], s["ema_slow"]), "signal"] = -1

    # антишум: подтверждение/порог/кулдаун
    s = _apply_persistence_and_cooldown(
        s,
        persist=max(1, int(persist)),
        cooldown=max(0, int(cooldown)),
        min_gap_pct=float(min_gap_pct),
    )

    # опциональный фильтр режимов (глушим сигналы в турбулентности)
    if use_regime_filter:
        gap = s["ema_fast"] - s["ema_slow"]
        regime = market_regime_flags(
            s["close"],
            vol_window=regime_vol_window,
            gap_series=gap,
        )
        # если сильная турбулентность — обнуляем сигнал
        s.loc[regime["is_turbulent"].values, "signal"] = 0

    # возвращаем только нужные колонки (ts гарантирован)
    out = s[["ts", "timestamp", "close", "ema_fast", "ema_slow", "signal"]].copy()
    out["ts"] = out["ts"].astype(np.int64)
    out["timestamp"] = out["timestamp"].astype(np.int64)
    out["signal"] = pd.to_numeric(out["signal"], errors="coerce").fillna(0).astype(int).clip(-1, 1)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Ансамбль стратегий (голосование/веса)
# ──────────────────────────────────────────────────────────────────────────────

def ensemble_signals(
    frames: Sequence[pd.DataFrame],
    *,
    weights: Optional[Sequence[float]] = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Объединяет несколько DataFrame сигналов в один по 'ts'.
    Требования к каждому df:
      - колонки как минимум: 'ts', 'signal'
    Правила:
      - Для каждого ts собираем взвешенную сумму сигналов в диапазоне [-1, +1]
      - Если |score| >= threshold → итоговый сигнал sign(score), иначе 0
    """
    if not frames:
        raise ValueError("ensemble_signals: пустой список стратегий")

    # нормируем веса
    n = len(frames)
    if weights is None:
        weights = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError("Длина weights должна совпадать с числом фреймов")
        w_sum = float(sum(weights))
        if w_sum <= 0:
            raise ValueError("Сумма weights должна быть > 0")
        weights = [float(w) / w_sum for w in weights]

    # приводим к набору (ts, signal_i)
    base = frames[0][["ts"]].copy()
    base = base.drop_duplicates(subset=["ts"]).sort_values("ts")
    for idx, f in enumerate(frames):
        part = f[["ts", "signal"]].rename(columns={"signal": f"sig_{idx}"})
        base = base.merge(part, on="ts", how="outer", validate="one_to_one")

    base = base.sort_values("ts").reset_index(drop=True)
    # пропуски → 0
    sig_cols = [c for c in base.columns if c.startswith("sig_")]
    for c in sig_cols:
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0.0).clip(-1, 1)

    # взвешенная сумма
    w = np.array(list(weights), dtype=float).reshape(1, -1)
    sig_mat = base[sig_cols].to_numpy(dtype=float)
    score = (sig_mat * w).sum(axis=1)

    # порог
    final = np.where(np.abs(score) >= float(threshold), np.sign(score), 0.0)

    out = pd.DataFrame(
        {
            "ts": base["ts"].astype(np.int64),
            "signal": final.astype(int),
            "score": score.astype(float),
        }
    )
    return out
