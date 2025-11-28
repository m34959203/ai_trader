from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
    model_validator,
)

from .indicators import bollinger_bands, cross_over, cross_under, ema, rsi, ichimoku, vwap


ALLOWED_STRATEGY_KINDS = {
    "ema",
    "ema_cross",
    "rsi_reversion",
    "rsi_mean_reversion",
    "bollinger",
    "bollinger_breakout",
    "ichimoku",
    "ichimoku_cloud",
    "vwap",
    "vwap_reversion",
}


@dataclass(frozen=True)
class StrategyDefinition:
    name: str
    kind: str
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrequencyFilterConfig:
    min_bars_between: int = 0
    max_signals_per_day: Optional[int] = None


class StrategySchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(..., min_length=1)
    kind: str = Field(..., min_length=1)
    weight: float = Field(default=1.0, ge=0.0)
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("name", "kind", mode="before")
    @classmethod
    def _coerce_str(cls, value: Any) -> str:
        if value is None:
            raise ValueError("must not be empty")
        return str(value).strip()

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, value: str) -> str:
        kind = value.lower()
        if kind not in ALLOWED_STRATEGY_KINDS:
            raise ValueError(f"Unsupported strategy kind: {value}")
        return kind

    @field_validator("params", mode="before")
    @classmethod
    def _ensure_params(cls, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("params must be a mapping")
        return dict(value)

    def to_definition(self) -> StrategyDefinition:
        return StrategyDefinition(
            name=self.name,
            kind=self.kind,
            weight=float(self.weight),
            params=self.params,
        )


class FrequencyFilterSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    min_bars_between: int = Field(default=0, ge=0)
    max_signals_per_day: Optional[int] = Field(default=None, ge=1)

    @field_validator("min_bars_between", mode="before")
    @classmethod
    def _coerce_min(cls, value: Any) -> int:
        if value is None:
            return 0
        return int(value)

    @field_validator("max_signals_per_day", mode="before")
    @classmethod
    def _coerce_max(cls, value: Any) -> Optional[int]:
        if value in (None, "", "null"):
            return None
        ivalue = int(value)
        if ivalue <= 0:
            raise ValueError("max_signals_per_day must be positive")
        return ivalue


class EnsembleOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class StrategyConfigSchema(BaseModel):
    model_config = ConfigDict(extra="ignore")

    strategies: List[StrategySchema] = Field(default_factory=list)
    ensemble_threshold: Optional[float] = Field(default=None)
    ensemble: Optional[EnsembleOptions] = None
    frequency_filter: FrequencyFilterSchema = Field(default_factory=FrequencyFilterSchema)

    _resolved_threshold: float = PrivateAttr(default=0.5)

    @field_validator("ensemble_threshold", mode="before")
    @classmethod
    def _coerce_threshold(cls, value: Any) -> Optional[float]:
        if value in (None, "", "null"):
            return None
        return float(value)

    @model_validator(mode="after")
    def _validate_config(self) -> "StrategyConfigSchema":
        if not self.strategies:
            raise ValueError("At least one strategy must be specified")
        if len({s.name for s in self.strategies}) != len(self.strategies):
            raise ValueError("Strategy names must be unique")
        if not any(s.weight > 0 for s in self.strategies):
            raise ValueError("At least one strategy must have weight > 0")

        threshold = self.ensemble_threshold
        if threshold is None and self.ensemble is not None:
            threshold = self.ensemble.threshold
        if threshold is None:
            threshold = 0.5
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError("Ensemble threshold must be between 0 and 1")
        self._resolved_threshold = float(threshold)
        return self

    def to_dataclass(self) -> "StrategyEnsembleConfig":
        freq = FrequencyFilterConfig(
            min_bars_between=int(self.frequency_filter.min_bars_between),
            max_signals_per_day=self.frequency_filter.max_signals_per_day,
        )
        return StrategyEnsembleConfig(
            strategies=[s.to_definition() for s in self.strategies],
            ensemble_threshold=self._resolved_threshold,
            frequency_filter=freq,
        )


@dataclass(frozen=True)
class StrategyEnsembleConfig:
    strategies: List[StrategyDefinition] = field(default_factory=list)
    ensemble_threshold: float = 0.5
    frequency_filter: FrequencyFilterConfig = field(default_factory=FrequencyFilterConfig)

    @staticmethod
    def from_mapping(mapping: Dict[str, Any]) -> "StrategyEnsembleConfig":
        mapping = dict(mapping or {})
        raw_strategies = mapping.get("strategies") or []
        normalized: List[Dict[str, Any]] = []
        for idx, item in enumerate(raw_strategies):
            if not isinstance(item, dict):
                raise ValueError(f"Strategy definition at index {idx} must be a mapping")
            normalized.append(
                {
                    "name": item.get("name") or item.get("id") or f"strategy_{idx}",
                    "kind": item.get("kind") or item.get("type"),
                    "weight": item.get("weight", 1.0),
                    "params": item.get("params") or {},
                }
            )

        payload: Dict[str, Any] = {
            "strategies": normalized,
            "ensemble_threshold": mapping.get("ensemble_threshold"),
        }
        if mapping.get("ensemble") is not None:
            payload["ensemble"] = mapping.get("ensemble")
        freq_raw = mapping.get("frequency_filter")
        if freq_raw is not None:
            payload["frequency_filter"] = freq_raw

        try:
            schema = StrategyConfigSchema.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Invalid strategy config: {exc}") from exc

        return schema.to_dataclass()

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


# ──────────────────────────────────────────────────────────────────────────────
# Расширенные стратегии
# ──────────────────────────────────────────────────────────────────────────────


def load_strategy_config(path: Optional[os.PathLike[str] | str] = None) -> StrategyEnsembleConfig:
    """Читает YAML/JSON конфигурацию ансамбля стратегий."""

    default_path = Path(os.getenv("STRATEGY_CONFIG", "configs/strategy.yaml"))
    cfg_path = Path(path) if path is not None else default_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Strategy config not found: {cfg_path}")

    text = cfg_path.read_text(encoding="utf-8")
    data: Optional[Dict[str, Any]] = None

    suffix = cfg_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore

            loaded = yaml.safe_load(text)
            if isinstance(loaded, dict):
                data = loaded
        except ModuleNotFoundError:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as json_exc:  # pragma: no cover
                raise RuntimeError("PyYAML is required to parse strategy.yaml") from json_exc
        except Exception:
            data = None
    if data is None:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Strategy config must be a mapping at top-level")
        data = parsed

    return StrategyEnsembleConfig.from_mapping(data)


def rsi_reversion_signals(
    df: pd.DataFrame,
    period: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.DataFrame:
    """Простая mean-reversion стратегия по RSI."""

    if "close" not in df.columns:
        raise ValueError("rsi_reversion_signals: требуется колонка 'close'")

    s = _ensure_time_cols(df).copy()
    s["close"] = pd.to_numeric(s["close"], errors="coerce")
    s = s.dropna(subset=["close"])
    s["rsi"] = rsi(s["close"], int(period))
    s["signal"] = 0

    cross_up = (s["rsi"].shift(1) < oversold) & (s["rsi"] >= oversold)
    cross_down = (s["rsi"].shift(1) > overbought) & (s["rsi"] <= overbought)

    s.loc[cross_up, "signal"] = 1
    s.loc[cross_down, "signal"] = -1

    out = s[["ts", "timestamp", "close", "rsi", "signal"]].copy()
    out["ts"] = out["ts"].astype(np.int64)
    out["timestamp"] = out["timestamp"].astype(np.int64)
    out["signal"] = pd.to_numeric(out["signal"], errors="coerce").fillna(0).astype(int).clip(-1, 1)
    return out


def bollinger_breakout_signals(
    df: pd.DataFrame,
    period: int = 20,
    n_std: float = 2.0,
    mode: str = "trend",
) -> pd.DataFrame:
    """Сигналы пробоя/возврата по полосам Боллинджера."""

    if "close" not in df.columns:
        raise ValueError("bollinger_breakout_signals: требуется колонка 'close'")

    s = _ensure_time_cols(df).copy()
    s["close"] = pd.to_numeric(s["close"], errors="coerce")
    s = s.dropna(subset=["close"])

    bb = bollinger_bands(s["close"], int(period), float(n_std))
    s = s.join(bb)
    s["signal"] = 0

    upper_cross = (s["close"].shift(1) <= s["bb_upper"].shift(1)) & (s["close"] > s["bb_upper"])
    lower_cross = (s["close"].shift(1) >= s["bb_lower"].shift(1)) & (s["close"] < s["bb_lower"])

    s.loc[upper_cross, "signal"] = 1
    s.loc[lower_cross, "signal"] = -1

    if mode.lower() in {"revert", "reversion", "mean_reversion"}:
        s["signal"] = -s["signal"]

    out = s[["ts", "timestamp", "close", "bb_upper", "bb_lower", "signal"]].copy()
    out["ts"] = out["ts"].astype(np.int64)
    out["timestamp"] = out["timestamp"].astype(np.int64)
    out["signal"] = pd.to_numeric(out["signal"], errors="coerce").fillna(0).astype(int).clip(-1, 1)
    return out


def ichimoku_signals(
    df: pd.DataFrame,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52,
    displacement: int = 26,
    mode: str = "cloud",
) -> pd.DataFrame:
    """Ichimoku Cloud strategy signals.

    Args:
        df: OHLCV DataFrame
        tenkan_period: Conversion line period (default 9)
        kijun_period: Base line period (default 26)
        senkou_b_period: Leading span B period (default 52)
        displacement: Cloud displacement (default 26)
        mode: Signal mode - "cloud" (cloud breakout), "tenkan_kijun" (TK cross), or "both"

    Returns:
        DataFrame with Ichimoku indicators and signals
    """
    required = ["close", "high", "low"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"ichimoku_signals: missing columns: {missing}")

    s = _ensure_time_cols(df).copy()
    for col in ["close", "high", "low"]:
        s[col] = pd.to_numeric(s[col], errors="coerce")
    s = s.dropna(subset=required)

    # Calculate Ichimoku
    ichi = ichimoku(
        s,
        tenkan_period=tenkan_period,
        kijun_period=kijun_period,
        senkou_b_period=senkou_b_period,
        displacement=displacement,
    )
    s = s.join(ichi)

    s["signal"] = 0

    mode = mode.lower()

    # Cloud breakout signals (price vs cloud)
    if mode in {"cloud", "both"}:
        # Bullish: price crosses above cloud
        above_cloud = (s["close"] > s["senkou_span_a"]) & (s["close"] > s["senkou_span_b"])
        below_cloud = (s["close"] < s["senkou_span_a"]) & (s["close"] < s["senkou_span_b"])

        bullish_break = (
            (s["close"].shift(1) <= s[["senkou_span_a", "senkou_span_b"]].shift(1).max(axis=1)) &
            above_cloud
        )
        bearish_break = (
            (s["close"].shift(1) >= s[["senkou_span_a", "senkou_span_b"]].shift(1).min(axis=1)) &
            below_cloud
        )

        s.loc[bullish_break, "signal"] = 1
        s.loc[bearish_break, "signal"] = -1

    # Tenkan-Kijun cross signals
    if mode in {"tenkan_kijun", "tk_cross", "both"}:
        tk_bullish = cross_over(s["tenkan_sen"], s["kijun_sen"])
        tk_bearish = cross_under(s["tenkan_sen"], s["kijun_sen"])

        if mode == "both":
            # In "both" mode, reinforce signals
            s.loc[tk_bullish & (s["signal"] == 0), "signal"] = 1
            s.loc[tk_bearish & (s["signal"] == 0), "signal"] = -1
        else:
            s.loc[tk_bullish, "signal"] = 1
            s.loc[tk_bearish, "signal"] = -1

    out = s[["ts", "timestamp", "close", "tenkan_sen", "kijun_sen",
             "senkou_span_a", "senkou_span_b", "chikou_span", "signal"]].copy()
    out["ts"] = out["ts"].astype(np.int64)
    out["timestamp"] = out["timestamp"].astype(np.int64)
    out["signal"] = pd.to_numeric(out["signal"], errors="coerce").fillna(0).astype(int).clip(-1, 1)
    return out


def vwap_signals(
    df: pd.DataFrame,
    session_start_hour: int = 0,
    mode: str = "reversion",
    deviation_threshold: float = 0.002,
) -> pd.DataFrame:
    """VWAP (Volume-Weighted Average Price) strategy signals.

    Args:
        df: OHLCV DataFrame with volume
        session_start_hour: Hour to reset VWAP (default 0 = midnight UTC)
        mode: "reversion" (mean reversion) or "trend" (trend following)
        deviation_threshold: Minimum % deviation from VWAP to trigger signal (default 0.2%)

    Returns:
        DataFrame with VWAP and signals
    """
    required = ["close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"vwap_signals: missing columns: {missing}")

    s = _ensure_time_cols(df).copy()
    s["close"] = pd.to_numeric(s["close"], errors="coerce")
    s["volume"] = pd.to_numeric(s["volume"], errors="coerce")
    s = s.dropna(subset=required)

    # Handle typical price if high/low available
    if "high" in s.columns and "low" in s.columns:
        s["high"] = pd.to_numeric(s["high"], errors="coerce")
        s["low"] = pd.to_numeric(s["low"], errors="coerce")
        s["typical_price"] = (s["high"] + s["low"] + s["close"]) / 3.0
    else:
        s["typical_price"] = s["close"]

    # Calculate VWAP
    vwap_series = vwap(
        s,
        session_start_hour=session_start_hour,
        typical_price_col="typical_price",
    )
    s["vwap"] = vwap_series

    # Calculate deviation from VWAP
    s["vwap_dev"] = (s["close"] - s["vwap"]) / s["vwap"].replace(0.0, np.nan)

    s["signal"] = 0

    mode = mode.lower()

    if mode in {"reversion", "mean_reversion"}:
        # Mean reversion: sell when far above VWAP, buy when far below
        overbought = s["vwap_dev"] > deviation_threshold  # Price > VWAP + threshold
        oversold = s["vwap_dev"] < -deviation_threshold  # Price < VWAP - threshold

        # Reversion to mean
        s.loc[oversold, "signal"] = 1  # Buy when below VWAP
        s.loc[overbought, "signal"] = -1  # Sell when above VWAP

    elif mode in {"trend", "breakout"}:
        # Trend following: buy when crossing above VWAP, sell when crossing below
        cross_above = cross_over(s["close"], s["vwap"])
        cross_below = cross_under(s["close"], s["vwap"])

        s.loc[cross_above, "signal"] = 1
        s.loc[cross_below, "signal"] = -1

    out = s[["ts", "timestamp", "close", "vwap", "vwap_dev", "signal"]].copy()
    out["ts"] = out["ts"].astype(np.int64)
    out["timestamp"] = out["timestamp"].astype(np.int64)
    out["signal"] = pd.to_numeric(out["signal"], errors="coerce").fillna(0).astype(int).clip(-1, 1)
    return out


def apply_frequency_filter(
    df: pd.DataFrame,
    *,
    min_bars_between: int = 0,
    max_signals_per_day: Optional[int] = None,
) -> pd.DataFrame:
    """Убирает слишком частые сигналы по окну и дневному лимиту."""

    if "signal" not in df.columns:
        raise ValueError("apply_frequency_filter: требуется колонка 'signal'")
    if "ts" not in df.columns and "timestamp" not in df.columns:
        raise ValueError("apply_frequency_filter: требуется 'ts' или 'timestamp'")

    out = df.sort_values("ts" if "ts" in df.columns else "timestamp").reset_index(drop=True).copy()
    out["signal"] = pd.to_numeric(out["signal"], errors="coerce").fillna(0).astype(int)
    signal_idx = out.columns.get_loc("signal")
    score_idx = out.columns.get_loc("score") if "score" in out.columns else None

    if min_bars_between > 0:
        last_idx: Optional[int] = None
        for i in range(len(out)):
            sig = int(out.iat[i, signal_idx])
            if sig == 0:
                continue
            if last_idx is not None and (i - last_idx) < int(min_bars_between):
                out.iat[i, signal_idx] = 0
                if score_idx is not None:
                    out.iat[i, score_idx] = 0.0
            else:
                last_idx = i

    if max_signals_per_day is not None:
        ts_series = out.get("ts") if "ts" in out.columns else out.get("timestamp")
        ts_dt = pd.to_datetime(ts_series, unit="s", utc=True, errors="coerce")
        if ts_dt.isna().all():
            raise ValueError("apply_frequency_filter: не удалось привести ts к datetime")
        day_counts: Dict[pd.Timestamp, int] = {}
        for i in range(len(out)):
            sig = int(out.iat[i, signal_idx])
            if sig == 0:
                continue
            day = ts_dt.iloc[i].floor("D")
            used = day_counts.get(day, 0)
            if used >= int(max_signals_per_day):
                out.iat[i, signal_idx] = 0
                if score_idx is not None:
                    out.iat[i, score_idx] = 0.0
            else:
                day_counts[day] = used + 1

    return out


def _run_strategy_definition(df: pd.DataFrame, definition: StrategyDefinition) -> pd.DataFrame:
    kind = definition.kind.lower()
    params = dict(definition.params or {})
    if kind in {"ema", "ema_cross"}:
        return ema_cross_signals(df, **params)
    if kind in {"rsi_reversion", "rsi_mean_reversion"}:
        return rsi_reversion_signals(df, **params)
    if kind in {"bollinger", "bollinger_breakout"}:
        return bollinger_breakout_signals(df, **params)
    if kind in {"ichimoku", "ichimoku_cloud"}:
        return ichimoku_signals(df, **params)
    if kind in {"vwap", "vwap_reversion"}:
        return vwap_signals(df, **params)
    raise ValueError(f"Unknown strategy kind: {definition.kind}")


def run_configured_ensemble(df: pd.DataFrame, config: StrategyEnsembleConfig) -> pd.DataFrame:
    if not config.strategies:
        raise ValueError("run_configured_ensemble: нет стратегий в конфиге")

    frames: List[pd.DataFrame] = []
    for strat in config.strategies:
        frame = _run_strategy_definition(df, strat)
        frames.append(frame)

    weights = [max(float(s.weight), 0.0) for s in config.strategies]
    combined = ensemble_signals(frames, weights=weights, threshold=config.ensemble_threshold)

    detail = combined[["ts"]].copy()
    for strat, frame in zip(config.strategies, frames):
        col = strat.name
        detail = detail.merge(
            frame[["ts", "signal"]].rename(columns={"signal": col}),
            on="ts",
            how="left",
        )
        detail[col] = pd.to_numeric(detail[col], errors="coerce").fillna(0).astype(int)

    detail["details"] = detail.apply(
        lambda row: [
            {
                "name": strat.name,
                "kind": strat.kind,
                "signal": int(row[strat.name]),
                "weight": float(strat.weight),
            }
            for strat in config.strategies
        ],
        axis=1,
    )

    out = combined.merge(detail[["ts", "details"]], on="ts", how="left")
    out = apply_frequency_filter(
        out,
        min_bars_between=config.frequency_filter.min_bars_between,
        max_signals_per_day=config.frequency_filter.max_signals_per_day,
    )
    return out


def run_strategy_config(
    df: pd.DataFrame,
    *,
    config_path: Optional[os.PathLike[str] | str] = None,
) -> pd.DataFrame:
    """Удобный хелпер: прочитать конфиг и вернуть итоговые сигналы."""

    cfg = load_strategy_config(config_path)
    return run_configured_ensemble(df, cfg)
