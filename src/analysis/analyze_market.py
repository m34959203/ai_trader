# src/analysis/analyze_market.py
from __future__ import annotations

"""
AI-Trader · Market Analysis Module (enhanced)

Public API (stable):
    - AnalysisConfig (dataclass)
    - DEFAULT_CONFIG (instance)
    - analyze_market(df_1h, df_4h=None, *, symbol=None, config=DEFAULT_CONFIG) -> dict
"""

import asyncio
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Tuple, Optional, Final

import numpy as np
import pandas as pd

# Индикаторы проекта
from src.ai.adaptive import DEFAULT_ENGINE
from src.indicators import (
    ema,
    rsi as rsi_ind,
    macd as macd_ind,
    bollinger_bands as bb_ind,
    atr as atr_ind,
    adx as adx_ind,
    force_index as force_ind,
    money_flow_index as mfi_ind,
    on_balance_volume as obv_ind,
    awesome_oscillator as ao_ind,
    candlestick_patterns,
)
from src.analysis.news_sentiment import collect_recent_news, AggregatedNews

__all__ = ["AnalysisConfig", "DEFAULT_CONFIG", "analyze_market"]


# ──────────────────────────────────────────────────────────────────────────────
# Config — неизменяемый через frozen=True
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    # EMAs
    ema_fast: int = 20
    ema_mid: int = 50
    ema_slow: int = 200

    # Oscillators / Bands
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_nstd: float = 2.0

    # Volatility
    atr_period: int = 14

    # ADX (trend strength)
    adx_period: int = 14
    adx_trend_threshold: float = 20.0  # <=> sideways if ADX < threshold

    # Volume-driven indicators
    force_index_period: int = 13
    mfi_period: int = 14

    # Candles
    hammer_shadow_ratio: float = 2.0   # lower (bull) / upper (bear) ≥ 2×body
    star_shadow_ratio: float = 2.0
    doji_body_frac: float = 0.1        # body ≤ 10% of range ⇒ doji

    # Levels
    fractal_k: int = 2                 # neighbors at each side
    levels_atr_mult: float = 0.5       # cluster tolerance = ATR * mult
    max_levels_out: int = 8
    level_proximity_mult: float = 0.6  # "too close" if dist < ATR*mult
    level_proximity_penalty: int = 10  # penalty to score

    # Scoring
    base_score: int = 50
    score_trend: int = 10
    score_macd_cross: int = 10
    score_macd_slope: int = 5
    score_rsi_zone: int = 5
    score_bb_break: int = 5
    score_candle: int = 5
    score_mtf_agree: int = 12
    score_mtf_conflict: int = 15       # penalty
    score_adx_trend_bonus: int = 6     # bonus if ADX >= threshold & aligned
    score_mfi_zone: int = 4            # бонус/штраф за зоны MFI
    score_force_trend: int = 4         # подтверждение Force Index
    score_obv_confirmation: int = 3    # подтверждение по OBV
    score_ao_momentum: int = 3         # подтверждение по Awesome Oscillator

    # Decision thresholds
    buy_threshold: int = 60
    sell_threshold: int = 40

    # Volatility regimes (ATR/close)
    vol_high: float = 0.02
    vol_medium: float = 0.01

    # Safety / data requirements
    min_bars_fast: int = 160           # ensure stable indicators on fast TF
    min_bars_slow: int = 50            # ensure stable indicators on slow TF
    max_last_bar_age_hours: int = 48   # если свеча старше → flat

    # Output hygiene
    max_reasons: int = 12              # prevent unbounded reasons list

    # MTF controls
    mtf_require_confirmation: bool = False  # if True: downgrade to flat if 4h conflicts strongly

    # News sentiment integration
    news_enabled: bool = True
    news_lookback_hours: int = 12
    news_sentiment_weight: int = 8
    news_min_items: int = 1
    news_allow_remote: bool = False
    news_symbol_aliases: Tuple[Tuple[str, Tuple[str, ...]], ...] = field(
        default_factory=lambda: (
            ("BTCUSDT", ("BTC", "Bitcoin", "BTC/USD")),
            ("ETHUSDT", ("ETH", "Ethereum", "ETH/USD")),
        )
    )


DEFAULT_CONFIG: Final[AnalysisConfig] = AnalysisConfig()


# ──────────────────────────────────────────────────────────────────────────────
# Validation & Utilities
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_COLS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "volume")


def _ensure_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema, ensure UTC DatetimeIndex, sort/dedup, cast to float."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise AssertionError("DataFrame index must be a pandas.DatetimeIndex (UTC).")

    out = df.copy()

    # Безопасно приводим к UTC
    try:
        if out.index.tz is None:
            out.index = out.index.tz_localize("UTC")
        else:
            out.index = out.index.tz_convert("UTC")
    except Exception:
        out.index = pd.DatetimeIndex(out.index, tz="UTC")

    # Сортировка и удаление дублей индекса
    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]

    # Проверка колонок и приведение к числам
    missing = [c for c in REQUIRED_COLS if c not in out.columns]
    if missing:
        raise AssertionError(f"Missing columns: {missing}")

    for c in REQUIRED_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["open", "high", "low", "close"])
    if out.empty:
        raise AssertionError("No valid rows after cleaning (NaNs removed).")

    return out


def _has_enough_bars(df: pd.DataFrame, need: int) -> bool:
    return len(df) >= int(need)


def _fresh_enough(df: pd.DataFrame, max_age_hours: int) -> bool:
    last_ts = df.index[-1]
    now = pd.Timestamp.now(tz="UTC")
    age_h = (now - last_ts).total_seconds() / 3600.0
    return age_h <= float(max_age_hours)


# ──────────────────────────────────────────────────────────────────────────────
# Indicators (via src.indicators)
# ──────────────────────────────────────────────────────────────────────────────

def _compute_indicators(df: pd.DataFrame, cfg: AnalysisConfig) -> pd.DataFrame:
    out = df.copy()

    out["ema_fast"] = ema(out["close"], cfg.ema_fast)
    out["ema_mid"]  = ema(out["close"], cfg.ema_mid)
    out["ema_slow"] = ema(out["close"], cfg.ema_slow)

    out["rsi"] = rsi_ind(out["close"], cfg.rsi_period)

    macd_df = macd_ind(out["close"], cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)
    out["macd"], out["macd_signal"], out["macd_hist"] = macd_df["macd"], macd_df["signal"], macd_df["hist"]

    bb_df = bb_ind(out["close"], cfg.bb_period, cfg.bb_nstd)
    out["bb_upper"] = bb_df["bb_upper"]
    out["bb_mid"] = bb_df["bb_mid"]
    out["bb_lower"] = bb_df["bb_lower"]

    out["atr"] = atr_ind(out["high"], out["low"], out["close"], cfg.atr_period)

    adx_df = adx_ind(out["high"], out["low"], out["close"], cfg.adx_period)
    out["+di"], out["-di"], out["adx"] = adx_df["pdi"], adx_df["ndi"], adx_df["adx"]

    out["force_index"] = force_ind(out["close"], out["volume"], cfg.force_index_period)
    out["mfi"] = mfi_ind(out["high"], out["low"], out["close"], out["volume"], cfg.mfi_period)
    out["obv"] = obv_ind(out["close"], out["volume"])
    out["ao"] = ao_ind(out["high"], out["low"])

    # «Тёплый старт» индикаторов — один раз
    out = out.dropna()
    if out.empty:
        raise AssertionError("Not enough data after indicator warm-up.")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Candles (pattern aggregation)
# ──────────────────────────────────────────────────────────────────────────────

def _candle_signals(df: pd.DataFrame, cfg: AnalysisConfig) -> Tuple[pd.Series, Dict[str, bool]]:
    pats = candlestick_patterns(
        df,
        hammer_ratio=cfg.hammer_shadow_ratio,
        star_ratio=cfg.star_shadow_ratio,
        doji_frac=cfg.doji_body_frac,
    )
    weights = (
        pats["bullish_engulfing"].astype(int)
        - pats["bearish_engulfing"].astype(int)
        + pats["hammer"].astype(int)
        + pats["inverted_hammer"].astype(int)
        - pats["shooting_star"].astype(int)
        + 2 * pats["morning_star"].astype(int)
        - 2 * pats["evening_star"].astype(int)
    )
    weights = weights.fillna(0)
    latest = pats.iloc[-1] if len(pats) else pd.Series(dtype=bool)
    snapshot = {col: bool(latest.get(col, False)) for col in pats.columns}
    return weights, snapshot


def _candle_reasons(snapshot: Dict[str, bool]) -> List[str]:
    label_map = {
        "hammer": "молот",
        "inverted_hammer": "перевёрнутый молот",
        "shooting_star": "падающая звезда",
        "bullish_engulfing": "бычье поглощение",
        "bearish_engulfing": "медвежье поглощение",
        "morning_star": "утренняя звезда",
        "evening_star": "вечерняя звезда",
        "doji": "доджи",
    }
    active = [label_map[name] for name, state in snapshot.items() if state and name in label_map]
    if not active:
        return []
    return ["Свечные сигналы: " + ", ".join(active)]


# ──────────────────────────────────────────────────────────────────────────────
# Levels (fractals + ATR clustering)
# ──────────────────────────────────────────────────────────────────────────────

def _find_pivots(df: pd.DataFrame, k: int = 2) -> Tuple[pd.Series, pd.Series]:
    """Return boolean series for swing highs/lows using simple fractal rule."""
    n = len(df)
    piv_h = pd.Series(False, index=df.index)
    piv_l = pd.Series(False, index=df.index)
    if n < (2 * k + 1):
        return piv_h, piv_l

    h, l = df["high"].to_numpy(), df["low"].to_numpy()
    for i in range(k, n - k):
        window = slice(i - k, i + k + 1)
        if h[i] == np.max(h[window]):
            piv_h.iloc[i] = True
        if l[i] == np.min(l[window]):
            piv_l.iloc[i] = True
    return piv_h, piv_l


def _cluster_levels(prices: List[float], tolerance: float) -> List[Tuple[float, int]]:
    """Simple 1D clustering by proximity tolerance; returns [(level, count), ...]."""
    if not prices:
        return []
    arr = np.array(sorted(prices), dtype=float)
    clusters: List[List[float]] = []
    cur = [arr[0]]
    for p in arr[1:]:
        if abs(p - cur[-1]) <= tolerance:
            cur.append(p)
        else:
            clusters.append(cur)
            cur = [p]
    clusters.append(cur)
    lvls = [float(np.mean(c)) for c in clusters]
    power = [int(len(c)) for c in clusters]
    return list(zip(lvls, power))


def _compute_levels(df: pd.DataFrame, cfg: AnalysisConfig) -> List[Dict]:
    if df.empty:
        return []
    piv_h, piv_l = _find_pivots(df, k=cfg.fractal_k)
    highs = df.loc[piv_h, "high"].astype(float).tolist()
    lows  = df.loc[piv_l, "low"].astype(float).tolist()

    # tolerance from ATR or fallback to avg range
    if "atr" in df.columns and not df["atr"].isna().all():
        tol = float(max(0.0, df["atr"].iloc[-1])) * float(max(0.0, cfg.levels_atr_mult))
    else:
        rng = (df["high"] - df["low"]).rolling(14, min_periods=5).mean()
        tol = float(max(0.0, (rng.iloc[-1] if rng.notna().any() else 0.0))) * float(max(0.0, cfg.levels_atr_mult))

    res = _cluster_levels(highs, tol)
    sup = _cluster_levels(lows, tol)

    levels = (
        [{"kind": "resistance", "price": float(lvl), "strength": int(s)} for (lvl, s) in res] +
        [{"kind": "support",    "price": float(lvl), "strength": int(s)} for (lvl, s) in sup]
    )
    # сортировка: сильнее — выше, затем ближе к текущей цене
    px = float(df["close"].iloc[-1])
    levels.sort(key=lambda x: (x["strength"], -abs(x["price"] - px)), reverse=True)
    # удалим редкие дубликаты цен после округлений толеранса
    dedup: List[Dict] = []
    seen = set()
    for lvl in levels:
        key = (lvl["kind"], round(lvl["price"], 8))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(lvl)
    return dedup


# ──────────────────────────────────────────────────────────────────────────────
# Trend / Volatility
# ──────────────────────────────────────────────────────────────────────────────

def _infer_trend(df: pd.DataFrame) -> str:
    e1, e2, e3 = float(df["ema_fast"].iloc[-1]), float(df["ema_mid"].iloc[-1]), float(df["ema_slow"].iloc[-1])
    if e1 > e2 > e3:
        return "up"
    if e1 < e2 < e3:
        return "down"
    return "sideways"


def _vol_state(df: pd.DataFrame, cfg: AnalysisConfig) -> str:
    close = float(max(df["close"].iloc[-1], 1e-12))
    atrv = float(df["atr"].iloc[-1] / close)
    if atrv > cfg.vol_high:
        return "high"
    if atrv > cfg.vol_medium:
        return "medium"
    return "low"


# ──────────────────────────────────────────────────────────────────────────────
# Rules + Scoring
# ──────────────────────────────────────────────────────────────────────────────

def _macd_slope_sig(df: pd.DataFrame) -> int:
    """MACD histogram slope over last few bars: +1 rising, -1 falling, 0 flat."""
    hist = df["macd_hist"]
    if len(hist) < 3:
        return 0
    last3 = hist.iloc[-3:]
    slope = float(last3.iloc[-1] - last3.iloc[0])
    if slope > 0:
        return +1
    if slope < 0:
        return -1
    return 0


def _closest_level_penalty(
    df: pd.DataFrame,
    levels: List[Dict],
    cfg: AnalysisConfig,
    side: str
) -> Tuple[int, Optional[str]]:
    """
    Штраф за покупку рядом с сопротивлением / продажу рядом с поддержкой.
    Ищем ближайший уровень нужного типа по абсолютной дистанции.
    """
    if not levels or df.empty:
        return 0, None

    price = float(df["close"].iloc[-1])

    # 1) Базовый ATR
    atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0
    if not np.isfinite(atr) or atr <= 0.0:
        # 2) Фолбэк: средний диапазон (high-low) за 14 баров
        rng = (df["high"] - df["low"]).rolling(14, min_periods=5).mean()
        atr = float(rng.iloc[-1]) if rng.notna().any() else 0.0
    if not np.isfinite(atr) or atr <= 0.0:
        # 3) Последний шанс: 0.1% от цены
        atr = max(1e-12, 0.001 * max(price, 1e-12))

    tol = float(cfg.level_proximity_mult) * atr

    if side == "buy":
        rel = [lvl for lvl in levels if lvl["kind"] == "resistance"]
        if not rel:
            return 0, None
        nearest = min(rel, key=lambda x: abs(float(x["price"]) - price))
        if abs(float(nearest["price"]) - price) <= tol:
            return int(cfg.level_proximity_penalty), f"Близко к сопротивлению ({nearest['price']:.4f})"

    if side == "sell":
        rel = [lvl for lvl in levels if lvl["kind"] == "support"]
        if not rel:
            return 0, None
        nearest = min(rel, key=lambda x: abs(float(x["price"]) - price))
        if abs(float(nearest["price"]) - price) <= tol:
            return int(cfg.level_proximity_penalty), f"Близко к поддержке ({nearest['price']:.4f})"

    return 0, None


def _rule_signal(
    df: pd.DataFrame,
    cfg: AnalysisConfig,
    levels: List[Dict],
) -> Tuple[str, int, List[str], Dict[str, object]]:
    row = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else row
    reasons: List[str] = []
    score = int(cfg.base_score)
    extras: Dict[str, object] = {"base_score": score}

    # ADX regime (trend vs sideways)
    adx = float(row["adx"])
    if adx < cfg.adx_trend_threshold:
        reasons.append(f"ADX {adx:.1f} < {cfg.adx_trend_threshold:.0f} (флэт)")
    else:
        reasons.append(f"ADX {adx:.1f} ≥ {cfg.adx_trend_threshold:.0f} (тренд)")

    # Trend by EMA ladder
    ema_bull = row["ema_fast"] > row["ema_mid"] > row["ema_slow"]
    ema_bear = row["ema_fast"] < row["ema_mid"] < row["ema_slow"]
    if ema_bull:
        reasons.append("Тренд: EMA_fast > EMA_mid > EMA_slow (бычий)")
        score += int(cfg.score_trend)
    elif ema_bear:
        reasons.append("Тренд: EMA_fast < EMA_mid < EMA_slow (медвежий)")
        score -= int(cfg.score_trend)
    else:
        reasons.append("Тренд: смешанный")

    # MACD cross
    bull_x = (row["macd"] > row["macd_signal"]) and (prev["macd"] <= prev["macd_signal"])
    bear_x = (row["macd"] < row["macd_signal"]) and (prev["macd"] >= prev["macd_signal"])
    if bull_x:
        reasons.append("MACD: бычий кросс")
        score += int(cfg.score_macd_cross)
    elif bear_x:
        reasons.append("MACD: медвежий кросс")
        score -= int(cfg.score_macd_cross)

    # MACD slope (histogram dynamics)
    slope_sig = _macd_slope_sig(df)
    if slope_sig > 0:
        reasons.append("MACD: гистограмма растёт")
        score += int(cfg.score_macd_slope)
    elif slope_sig < 0:
        reasons.append("MACD: гистограмма падает")
        score -= int(cfg.score_macd_slope)

    # RSI zones
    rsi_val = float(row["rsi"])
    if rsi_val < 30:
        reasons.append("RSI < 30 (перепроданность)")
        score += int(cfg.score_rsi_zone)
    elif rsi_val > 70:
        reasons.append("RSI > 70 (перекупленность)")
        score -= int(cfg.score_rsi_zone)

    # Bollinger breaks
    if row["close"] > row["bb_upper"]:
        reasons.append("Пробой верхней Bollinger")
        score += int(cfg.score_bb_break)
    elif row["close"] < row["bb_lower"]:
        reasons.append("Пробой нижней Bollinger")
        score -= int(cfg.score_bb_break)

    # MFI zones (volume-based oscillator)
    mfi_val = float(row.get("mfi", np.nan))
    if np.isfinite(mfi_val):
        if mfi_val < 20:
            reasons.append(f"MFI {mfi_val:.1f} < 20 (потенциал разворота)")
            score += int(cfg.score_mfi_zone)
        elif mfi_val > 80:
            reasons.append(f"MFI {mfi_val:.1f} > 80 (перекупленность)")
            score -= int(cfg.score_mfi_zone)

    # Force Index trend confirmation
    fi_now = float(row.get("force_index", 0.0))
    fi_prev = float(prev.get("force_index", fi_now)) if hasattr(prev, "get") else fi_now
    if fi_now > 0 and fi_now >= fi_prev:
        reasons.append("Force Index > 0 (покупатели доминируют)")
        score += int(cfg.score_force_trend)
    elif fi_now < 0 and fi_now <= fi_prev:
        reasons.append("Force Index < 0 (продавцы доминируют)")
        score -= int(cfg.score_force_trend)

    # OBV slope confirmation
    obv_now = float(row.get("obv", 0.0))
    obv_prev = float(prev.get("obv", obv_now)) if hasattr(prev, "get") else obv_now
    obv_slope = obv_now - obv_prev
    if obv_slope > 0:
        reasons.append("OBV растёт (подтверждение объёмом)")
        score += int(cfg.score_obv_confirmation)
    elif obv_slope < 0:
        reasons.append("OBV падает (отток объёма)")
        score -= int(cfg.score_obv_confirmation)

    # Awesome Oscillator momentum
    ao_now = float(row.get("ao", 0.0))
    ao_prev = float(prev.get("ao", ao_now)) if hasattr(prev, "get") else ao_now
    if ao_now > 0 and ao_now >= ao_prev:
        reasons.append("Awesome Oscillator > 0 (бычий импульс)")
        score += int(cfg.score_ao_momentum)
    elif ao_now < 0 and ao_now <= ao_prev:
        reasons.append("Awesome Oscillator < 0 (медвежий импульс)")
        score -= int(cfg.score_ao_momentum)

    # Candles
    cscore, candle_snapshot = _candle_signals(df, cfg)
    extras["candles"] = candle_snapshot
    cs = int(cscore.iloc[-1]) if len(cscore) else 0
    candle_reasons = _candle_reasons(candle_snapshot)
    reasons.extend(candle_reasons)
    if cs > 0:
        reasons.append("Свечной блок: бычий перевес")
        score += int(cfg.score_candle)
    elif cs < 0:
        reasons.append("Свечной блок: медвежий перевес")
        score -= int(cfg.score_candle)
    if candle_snapshot.get("doji"):
        reasons.append("Свечной: доджи (неопределённость)")

    # Initial decision from raw score (предварительный)
    score = max(0, min(100, int(score)))
    if score >= cfg.buy_threshold:
        decision = "buy"
    elif score <= cfg.sell_threshold:
        decision = "sell"
    else:
        decision = "flat"

    # ── Level proximity guardrail ──────────────────────────────────────────────
    side_for_penalty: Optional[str] = decision
    if side_for_penalty == "flat":
        side_for_penalty = "buy" if ema_bull else ("sell" if ema_bear else None)
    elif side_for_penalty == "buy" and ema_bear:
        side_for_penalty = "sell"
    elif side_for_penalty == "sell" and ema_bull:
        side_for_penalty = "buy"

    if side_for_penalty in ("buy", "sell"):
        pen, why = _closest_level_penalty(df, levels, cfg, side=side_for_penalty)
        if pen > 0:
            score = max(0, score - pen)
            if why:
                reasons.append(why)
            if decision == "buy" and side_for_penalty == "buy" and score < cfg.buy_threshold:
                decision = "flat"
            if decision == "sell" and side_for_penalty == "sell" and score > cfg.sell_threshold:
                decision = "flat"

    # ADX trend bonus (only if direction aligns with DI)
    if adx >= cfg.adx_trend_threshold:
        plus_di = float(row["+di"])
        minus_di = float(row["-di"])
        if decision == "buy" and plus_di > minus_di:
            score = min(100, score + int(cfg.score_adx_trend_bonus))
            reasons.append("ADX: тренд подтверждён (+DI > -DI)")
        elif decision == "sell" and minus_di > plus_di:
            score = min(100, score + int(cfg.score_adx_trend_bonus))
            reasons.append("ADX: тренд подтверждён (-DI > +DI)")

    # Keep reasons concise
    if len(reasons) > cfg.max_reasons:
        reasons = reasons[: cfg.max_reasons] + ["…"]

    extras["indicators"] = {
        "rsi": float(row.get("rsi", np.nan)),
        "mfi": mfi_val,
        "force_index": fi_now,
        "obv": obv_now,
        "ao": ao_now,
        "adx": adx,
        "macd": float(row.get("macd", np.nan)),
        "macd_signal": float(row.get("macd_signal", np.nan)),
    }
    extras["raw_score"] = int(score)

    return decision, int(score), reasons, extras


def _align_timeframes(df_fast: pd.DataFrame, df_slow: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Trim slow TF to not exceed last timestamp of fast TF."""
    fast_last = df_fast.index[-1]
    slow = df_slow[df_slow.index <= fast_last]
    return df_fast, (slow if len(slow) else None)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def analyze_market(
    df_1h: pd.DataFrame,
    df_4h: Optional[pd.DataFrame] = None,
    *,
    symbol: Optional[str] = None,
    config: AnalysisConfig = DEFAULT_CONFIG,
) -> Dict:
    """
    Analyze market on 1h (fast) with optional 4h (slow) confirmation.
    """
    cfg = config

    # Validate & normalize
    df1 = _ensure_df(df_1h)
    if not _has_enough_bars(df1, cfg.min_bars_fast):
        raise AssertionError(f"Need at least {cfg.min_bars_fast} bars on fast TF.")

    # Freshness gate
    if not _fresh_enough(df1, cfg.max_last_bar_age_hours):
        last_ts = df1.index[-1].isoformat()
        return {
            "trend": "sideways",
            "volatility": "low",
            "signal": "flat",
            "confidence": 0,
            "reasons": [f"Нет свежих данных: последняя свеча {last_ts}"],
            "levels": [],
        }

    # Indicators / features (1h)
    df1 = _compute_indicators(df1, cfg)

    # Levels / trend / volatility / base decision (1h)
    levels = _compute_levels(df1, cfg)
    trend = _infer_trend(df1)
    vol   = _vol_state(df1, cfg)
    sig, score, reasons, extras = _rule_signal(df1, cfg, levels)
    extras["score_after_indicators"] = int(score)
    latest_row = df1.iloc[-1]
    if sig in {"buy", "sell"}:
        DEFAULT_ENGINE.record_example(latest_row, 1 if sig == "buy" else 0)
    if DEFAULT_ENGINE.needs_retrain():
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(DEFAULT_ENGINE.maybe_retrain())
        except RuntimeError:
            asyncio.run(DEFAULT_ENGINE.maybe_retrain())

    # Multi-timeframe confirmation (4h)
    mtf: Optional[Dict[str, str]] = None
    if df_4h is not None:
        mtf = {"trend_4h": "sideways"}  # всегда присутствует при передаче df_4h

        try:
            df4 = _ensure_df(df_4h)
        except AssertionError:
            df4 = None

        if df4 is not None and _has_enough_bars(df4, cfg.min_bars_slow):
            df4 = _compute_indicators(df4, cfg)
            _, df4 = _align_timeframes(df1, df4)
            if df4 is not None and len(df4):
                trend4 = _infer_trend(df4)
                mtf["trend_4h"] = trend4

                # Дополнительные подтверждения 4h
                slope4 = _macd_slope_sig(df4)  # +1 rising, -1 falling, 0 flat
                rsi4   = float(df4["rsi"].iloc[-1])

                agree = False
                conflict = False

                if sig == "buy":
                    agree = (trend4 == "up") or (slope4 > 0) or (rsi4 < 60)
                    conflict = (trend4 == "down") or (slope4 < 0) or (rsi4 > 70)
                elif sig == "sell":
                    agree = (trend4 == "down") or (slope4 < 0) or (rsi4 > 40)
                    conflict = (trend4 == "up") or (slope4 > 0) or (rsi4 < 30)

                if agree and not conflict:
                    score = min(100, score + int(cfg.score_mtf_agree))
                    reasons.append("МТФ: 4h подтверждает сигнал")
                elif conflict and not agree:
                    score = max(0, score - int(cfg.score_mtf_conflict))
                    reasons.append("МТФ: 4h противоречит сигналу")
                    if cfg.mtf_require_confirmation:
                        sig = "flat"
                        reasons.append("МТФ: требовалось подтверждение — сигнал снят")
                else:
                    reasons.append("МТФ: 4h нейтрален")

                # Пересчёт решения после MTF-правок
                if sig != "flat":
                    if score >= cfg.buy_threshold:
                        sig = "buy"
                    elif score <= cfg.sell_threshold:
                        sig = "sell"
                    else:
                        sig = "flat"

    extras["score_after_mtf"] = int(score)

    news_blob: Optional[AggregatedNews] = None
    if symbol and cfg.news_enabled:
        alias_map = dict(cfg.news_symbol_aliases)
        extra_aliases = {k: list(v) for k, v in alias_map.items()}
        news_blob = collect_recent_news(
            symbol,
            lookback_hours=cfg.news_lookback_hours,
            extra_aliases=extra_aliases,
            allow_remote=cfg.news_allow_remote,
        )
        if news_blob.items and len(news_blob.items) >= cfg.news_min_items:
            delta = int(round(news_blob.normalized * cfg.news_sentiment_weight))
            if delta != 0:
                score = max(0, min(100, score + delta))
                label = news_blob.sentiment_label
                if label == "bullish":
                    reasons.append(
                        f"Новости: позитивный фон (score={news_blob.normalized:.2f}, +{delta})"
                    )
                elif label == "bearish":
                    reasons.append(
                        f"Новости: негативный фон (score={news_blob.normalized:.2f}, {delta})"
                    )
                else:
                    reasons.append(
                        f"Новости: нейтральный фон (score={news_blob.normalized:.2f})"
                    )
            else:
                reasons.append("Новости: фон нейтральный")
        extras["news_sentiment"] = {
            "score": news_blob.score if news_blob else 0.0,
            "normalized": news_blob.normalized if news_blob else 0.0,
            "label": news_blob.sentiment_label if news_blob else "neutral",
            "counts": news_blob.counts if news_blob else {"positive": 0, "negative": 0, "neutral": 0},
        }
        extras["score_after_news"] = int(score)
    else:
        extras.setdefault("score_after_news", int(score))

    # ── Финальная нормализация по тренду ───────────────────────────────────────
    extras.setdefault("score_after_news", int(score))
    if trend == "up":
        final_sig = "buy" if score >= cfg.buy_threshold else "flat"
    elif trend == "down":
        final_sig = "sell" if score >= cfg.sell_threshold else "flat"
    else:
        final_sig = "flat"

    final_score = int(max(0, min(100, score)))
    extras["final_score"] = final_score

    # Final payload (JSON-friendly)
    out_levels = [
        {"kind": lvl["kind"], "price": float(lvl["price"]), "strength": int(lvl["strength"])}
        for lvl in levels[: cfg.max_levels_out]
    ]

    result = {
        "trend": trend,
        "volatility": vol,
        "signal": final_sig,
        "confidence": final_score,
        "signal_score": final_score,
        "reasons": list(reasons),
        "levels": out_levels,
    }
    if mtf is not None:
        result["mtf"] = mtf
    adaptive = DEFAULT_ENGINE.adjust(result, df=df1)
    meta = result.setdefault("meta", {})
    meta["adaptive"] = asdict(adaptive)
    meta["indicators"] = extras.get("indicators", {})
    meta["candles"] = extras.get("candles", {})
    meta["score_trace"] = {
        "base": extras.get("base_score"),
        "after_indicators": extras.get("score_after_indicators"),
        "after_mtf": extras.get("score_after_mtf"),
        "after_news": extras.get("score_after_news"),
        "final": extras.get("final_score"),
    }
    meta["news_sentiment"] = extras.get("news_sentiment")
    if news_blob and news_blob.items:
        meta["news"] = {
            "score": news_blob.score,
            "normalized": news_blob.normalized,
            "label": news_blob.sentiment_label,
            "counts": news_blob.counts,
            "headlines": [
                {
                    "title": item.title,
                    "summary": item.summary,
                    "sentiment": item.sentiment,
                    "importance": item.importance,
                    "impact": item.impact,
                    "published_ts": item.published_ts,
                    "link": item.link,
                }
                for item in news_blob.items[:5]
            ],
        }
    return result


if __name__ == "__main__":
    # Synthetic self-test
    rng = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=420, freq="h", tz="UTC")
    np.random.seed(42)
    price = 100 + np.cumsum(np.random.randn(len(rng))) * 0.5
    df_fast = pd.DataFrame(
        {
            "open":  price + np.random.randn(len(rng)) * 0.2,
            "high":  price + np.abs(np.random.randn(len(rng)) * 0.6),
            "low":   price - np.abs(np.random.randn(len(rng)) * 0.6),
            "close": price + np.random.randn(len(rng)) * 0.2,
            "volume": np.random.randint(1_000, 10_000, size=len(rng)),
        },
        index=rng,
    )

    rng4 = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=220, freq="4h", tz="UTC")
    price4 = 100 + np.cumsum(np.random.randn(len(rng4))) * 1.1
    df_slow = pd.DataFrame(
        {
            "open":  price4 + np.random.randn(len(rng4)) * 0.4,
            "high":  price4 + np.abs(np.random.randn(len(rng4)) * 1.2),
            "low":   price4 - np.abs(np.random.randn(len(rng4)) * 1.2),
            "close": price4 + np.random.randn(len(rng4)) * 0.4,
            "volume": np.random.randint(500, 7000, size=len(rng4)),
        },
        index=rng4,
    )

    res = analyze_market(df_fast, df_slow)
    print(res)
