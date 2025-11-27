# ai_trader/tasks/auto_trader.py
from __future__ import annotations

import os
import asyncio
import logging
import time
from dataclasses import dataclass, field, replace
from typing import List, Dict, Any, Optional, Tuple, Iterable

import httpx
import pandas as pd

from src.indicators import atr as atr_indicator
from src.strategy import ema_cross_signals
from utils.risk_config import load_risk_config, RiskConfig

LOG = logging.getLogger("ai_trader.auto_trader")


# ═════════════════════════════════════════════════════════════════════════════
# Конфигурация авто-трейдера
# ═════════════════════════════════════════════════════════════════════════════


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return str(val).strip().lower() not in {"0", "false", "off", ""}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, default)))
    except Exception:
        return int(default)


def _env_list(name: str, default: Iterable[str]) -> List[str]:
    raw = os.getenv(name)
    if raw is None:
        return [item.strip().upper() for item in default if str(item).strip()]
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _default_api_base() -> str:
    port = os.getenv("APP_PORT") or os.getenv("PORT") or "8000"
    base = os.getenv("APP_API_BASE") or f"http://127.0.0.1:{port}"
    return base.rstrip("/")


@dataclass(frozen=True)
class AutoTraderConfig:
    enabled: bool = True
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    timeframe: str = "15m"
    api_base: str = field(default_factory=_default_api_base)
    source: str = "binance"
    mode: str = "binance"
    testnet: bool = True

    sma_fast: int = 20
    sma_slow: int = 50
    confirm_on_close: bool = True
    signal_persist: int = 1
    signal_cooldown: int = 0
    signal_min_gap_pct: float = 0.0
    use_regime_filter: bool = False

    atr_period: int = 14
    atr_mult: float = 2.0
    min_sl_pct: float = 0.001
    max_sl_pct: float = 0.05

    quote_usdt: float = 50.0
    equity_min_usdt: float = 10.0
    sl_pct: float = 0.0
    tp_pct: float = 0.0
    use_risk_autosize: bool = True
    use_portfolio_risk: bool = True
    max_portfolio_risk: float = 0.0

    sell_on_death: bool = True
    sell_pct_of_position: float = 1.0

    loop_sec: float = 30.0
    http_timeout_sec: float = 20.0
    net_max_retries: int = 3
    net_retry_base: float = 0.5
    net_retry_cap: float = 6.0

    dry_run: bool = False
    concurrency: int = 4

    @classmethod
    def from_env(cls) -> "AutoTraderConfig":
        enabled = _env_bool("AUTO_TRADER_ENABLED", True)
        symbols = _env_list("AUTO_TRADER_SYMBOLS", ["BTCUSDT"])
        timeframe = os.getenv("AUTO_TRADER_TIMEFRAME", "15m").strip()
        api_base = os.getenv("AUTO_TRADER_API_BASE") or _default_api_base()
        source = os.getenv("AUTO_TRADER_SOURCE", "binance").strip() or "binance"
        mode = os.getenv("AUTO_TRADER_MODE", "binance").strip() or "binance"
        testnet = _env_bool("AUTO_TRADER_TESTNET", True)

        sma_fast = _env_int("AUTO_TRADER_SMA_FAST", 20)
        sma_slow = _env_int("AUTO_TRADER_SMA_SLOW", 50)
        confirm_on_close = _env_bool("AUTO_TRADER_CONFIRM_ON_CLOSE", True)
        signal_persist = max(1, _env_int("AUTO_TRADER_SIGNAL_PERSIST", 1))
        signal_cooldown = max(0, _env_int("AUTO_TRADER_SIGNAL_COOLDOWN", 0))
        signal_min_gap_pct = float(_env_float("AUTO_TRADER_SIGNAL_MIN_GAP_PCT", 0.0))
        use_regime_filter = _env_bool("AUTO_TRADER_USE_REGIME_FILTER", False)

        atr_period = max(0, _env_int("AUTO_TRADER_ATR_PERIOD", 14))
        atr_mult = max(0.0, _env_float("AUTO_TRADER_ATR_MULT", 2.0))
        min_sl_pct = max(0.0, _env_float("AUTO_TRADER_MIN_SL_PCT", 0.001))
        max_sl_pct = max(min_sl_pct, _env_float("AUTO_TRADER_MAX_SL_PCT", 0.05))

        quote_usdt = max(0.0, _env_float("AUTO_TRADER_QUOTE_USDT", 50.0))
        equity_min_usdt = max(0.0, _env_float("AUTO_TRADER_EQUITY_MIN_USDT", 10.0))
        sl_pct = max(0.0, _env_float("AUTO_TRADER_SL_PCT", 0.0))
        tp_pct = max(0.0, _env_float("AUTO_TRADER_TP_PCT", 0.0))
        use_risk_autosize = _env_bool("AUTO_TRADER_USE_RISK_AUTOSIZE", True)
        use_portfolio_risk = _env_bool("AUTO_TRADER_USE_PORTFOLIO_RISK", True)
        max_portfolio_risk = max(0.0, _env_float("AUTO_TRADER_MAX_PORTFOLIO_RISK", 0.0))

        sell_on_death = _env_bool("AUTO_TRADER_SELL_ON_DEATH", True)
        sell_pct_of_position = max(0.0, min(1.0, _env_float("AUTO_TRADER_SELL_PCT_OF_POSITION", 1.0)))

        loop_sec = max(1.0, float(_env_float("AUTO_TRADER_LOOP_SEC", 30.0)))
        http_timeout_sec = max(1.0, float(_env_float("AUTO_TRADER_HTTP_TIMEOUT", 20.0)))
        net_max_retries = max(1, _env_int("AUTO_TRADER_NET_MAX_RETRIES", 3))
        net_retry_base = max(0.1, float(_env_float("AUTO_TRADER_NET_RETRY_BASE", 0.5)))
        net_retry_cap = max(net_retry_base, float(_env_float("AUTO_TRADER_NET_RETRY_CAP", 6.0)))

        dry_run = _env_bool("AUTO_TRADER_DRY_RUN", False)
        concurrency = max(1, _env_int("AUTO_TRADER_CONCURRENCY", 4))

        return cls(
            enabled=enabled,
            symbols=symbols,
            timeframe=timeframe,
            api_base=(api_base or _default_api_base()).rstrip("/"),
            source=source,
            mode=mode,
            testnet=testnet,
            sma_fast=sma_fast,
            sma_slow=sma_slow,
            confirm_on_close=confirm_on_close,
            signal_persist=signal_persist,
            signal_cooldown=signal_cooldown,
            signal_min_gap_pct=signal_min_gap_pct,
            use_regime_filter=use_regime_filter,
            atr_period=atr_period,
            atr_mult=atr_mult,
            min_sl_pct=min_sl_pct,
            max_sl_pct=max_sl_pct,
            quote_usdt=quote_usdt,
            equity_min_usdt=equity_min_usdt,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            use_risk_autosize=use_risk_autosize,
            use_portfolio_risk=use_portfolio_risk,
            max_portfolio_risk=max_portfolio_risk,
            sell_on_death=sell_on_death,
            sell_pct_of_position=sell_pct_of_position,
            loop_sec=float(loop_sec),
            http_timeout_sec=float(http_timeout_sec),
            net_max_retries=int(net_max_retries),
            net_retry_base=float(net_retry_base),
            net_retry_cap=float(net_retry_cap),
            dry_run=dry_run,
            concurrency=int(concurrency),
        )

    def to_public_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "symbols": list(self.symbols),
            "timeframe": self.timeframe,
            "api_base": self.api_base,
            "source": self.source,
            "mode": self.mode,
            "testnet": self.testnet,
            "sma_fast": self.sma_fast,
            "sma_slow": self.sma_slow,
            "signal_persist": self.signal_persist,
            "signal_cooldown": self.signal_cooldown,
            "signal_min_gap_pct": self.signal_min_gap_pct,
            "use_regime_filter": self.use_regime_filter,
            "atr_period": self.atr_period,
            "atr_mult": self.atr_mult,
            "min_sl_pct": self.min_sl_pct,
            "max_sl_pct": self.max_sl_pct,
            "quote_usdt": self.quote_usdt,
            "equity_min_usdt": self.equity_min_usdt,
            "sl_pct": self.sl_pct,
            "tp_pct": self.tp_pct,
            "use_risk_autosize": self.use_risk_autosize,
            "use_portfolio_risk": self.use_portfolio_risk,
            "max_portfolio_risk": self.max_portfolio_risk,
            "sell_on_death": self.sell_on_death,
            "sell_pct_of_position": self.sell_pct_of_position,
            "loop_sec": self.loop_sec,
            "http_timeout_sec": self.http_timeout_sec,
            "net_max_retries": self.net_max_retries,
            "net_retry_base": self.net_retry_base,
            "net_retry_cap": self.net_retry_cap,
            "dry_run": self.dry_run,
            "concurrency": self.concurrency,
        }


CONFIG = AutoTraderConfig.from_env()


@dataclass
class AutoTraderState:
    last_candle_ts: Dict[str, int] = field(default_factory=dict)
    last_signal: Dict[str, int] = field(default_factory=dict)
    iteration: int = 0
    started_at: float = 0.0
    last_cycle_started: float = 0.0
    last_cycle_finished: float = 0.0
    last_error: Optional[str] = None
    last_trade: Optional[Dict[str, Any]] = None


_RUNTIME_STATUS: Dict[str, Any] = {
    "running": False,
    "config": CONFIG.to_public_dict(),
    "started_at": None,
    "last_cycle_started": None,
    "last_cycle_finished": None,
    "iteration": 0,
    "last_error": None,
    "last_trade": None,
}


def set_config(config: AutoTraderConfig) -> None:
    """Обновляет конфигурацию по умолчанию (используется при автозапуске)."""
    global CONFIG
    CONFIG = config
    _RUNTIME_STATUS["config"] = CONFIG.to_public_dict()


def get_config() -> AutoTraderConfig:
    return CONFIG


def get_runtime_status() -> Dict[str, Any]:
    return dict(_RUNTIME_STATUS)


@dataclass(frozen=True)
class RiskSnapshot:
    config: RiskConfig
    portfolio_used: float
    positions: List[Dict[str, Any]]


def _load_risk_config_cached() -> RiskConfig:
    try:
        cfg = load_risk_config().validate()
    except Exception:
        cfg = RiskConfig()  # type: ignore[call-arg]
    return cfg

# ═════════════════════════════════════════════════════════════════════════════
# Утилиты
# ═════════════════════════════════════════════════════════════════════════════
def _rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["t", "o", "h", "l", "c", "v"])
    df = pd.DataFrame(rows)
    rename = {"t": "ts", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df = df.rename(columns=rename)
    for col in ("ts", "open", "high", "low", "close", "volume"):
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ts", "close"]).copy()
    df["ts"] = df["ts"].astype(int)
    df = df.drop_duplicates(subset=["ts"], keep="last").sort_values("ts")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df["timestamp"] = pd.to_datetime(df["ts"], unit="ms" if df["ts"].max() > 10**12 else "s", utc=True)
    df = df.set_index("timestamp")
    return df


def _last_signal(df: pd.DataFrame, cfg: AutoTraderConfig) -> Tuple[int, int]:
    signals = ema_cross_signals(
        df,
        fast=cfg.sma_fast,
        slow=cfg.sma_slow,
        persist=max(1, cfg.signal_persist),
        cooldown=max(0, cfg.signal_cooldown),
        min_gap_pct=max(0.0, cfg.signal_min_gap_pct),
        use_regime_filter=cfg.use_regime_filter,
    )
    sig = int(signals["signal"].iloc[-1]) if not signals.empty else 0
    ts = int(signals["ts"].iloc[-1]) if not signals.empty else int(df.index[-1].timestamp())
    return sig, ts


def _atr_stop_pct(df: pd.DataFrame, cfg: AutoTraderConfig) -> float:
    if cfg.atr_period <= 0 or cfg.atr_mult <= 0:
        return max(0.0, cfg.sl_pct)
    try:
        atr_series = atr_indicator(df["high"], df["low"], df["close"], cfg.atr_period)
        atr_val = float(atr_series.iloc[-1]) if len(atr_series) else 0.0
        last_close = float(df["close"].iloc[-1])
    except Exception:
        atr_val = 0.0
        last_close = 0.0
    if atr_val <= 0 or last_close <= 0:
        return max(0.0, cfg.sl_pct)
    pct = (atr_val * cfg.atr_mult) / last_close
    pct = max(cfg.min_sl_pct, pct)
    pct = max(pct, cfg.sl_pct)
    pct = min(cfg.max_sl_pct, pct)
    return pct

async def _sleep_backoff(attempt: int, cfg: AutoTraderConfig) -> None:
    # экспоненциальная задержка с капом
    delay = min(cfg.net_retry_cap, cfg.net_retry_base * (2 ** attempt))
    await asyncio.sleep(delay)


def _position_risk_fraction(*, equity: float, qty: float, entry_price: float, stop_price: Optional[float], cfg: RiskConfig) -> float:
    if equity <= 0 or qty <= 0 or entry_price <= 0 or stop_price is None:
        return float(cfg.risk_pct_per_trade)
    dist = abs(entry_price - stop_price)
    min_dist = max(cfg.min_sl_distance_pct * entry_price, 1e-12)
    dist = max(dist, min_dist)
    return max(0.0, min(1.0, (dist * qty) / max(equity, 1e-9)))


def _portfolio_risk_used(positions: List[Dict[str, Any]], *, cfg: RiskConfig, equity: float) -> float:
    total = 0.0
    for pos in positions:
        try:
            qty = float(pos.get("qty") or pos.get("positionAmt") or 0.0)
        except Exception:
            qty = 0.0
        try:
            entry_price = float(pos.get("entry_price") or pos.get("entryPrice") or pos.get("avg_price") or 0.0)
        except Exception:
            entry_price = 0.0
        sl_price: Optional[float] = None
        raw_sl = pos.get("stop_loss_price") or pos.get("sl_price")
        try:
            if raw_sl is not None:
                sl_price = float(raw_sl)
        except Exception:
            sl_price = None
        if sl_price is None:
            try:
                sl_pct = float(pos.get("sl_pct") or 0.0)
                if sl_pct > 0 and entry_price > 0:
                    side = str(pos.get("side") or pos.get("positionSide") or "long").lower()
                    if side.startswith("short"):
                        sl_price = entry_price * (1.0 + sl_pct)
                    else:
                        sl_price = entry_price * (1.0 - sl_pct)
            except Exception:
                sl_price = None
        total += _position_risk_fraction(
            equity=equity,
            qty=float(qty),
            entry_price=float(entry_price),
            stop_price=sl_price,
            cfg=cfg,
        )
    return max(0.0, min(1.0, total))


def _risk_snapshot_from_positions(
    positions: List[Dict[str, Any]], *, equity: float, settings: AutoTraderConfig
) -> RiskSnapshot:
    cfg = _load_risk_config_cached()
    used = _portfolio_risk_used(positions, cfg=cfg, equity=equity) if settings.use_portfolio_risk else 0.0
    if settings.max_portfolio_risk > 0:
        used = min(used, settings.max_portfolio_risk)
    return RiskSnapshot(config=cfg, portfolio_used=used, positions=positions)

async def _post_json(client: httpx.AsyncClient, url: str, json: Any, cfg: AutoTraderConfig) -> httpx.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(cfg.net_max_retries):
        try:
            r = await client.post(url, json=json, timeout=cfg.http_timeout_sec)
            if r.status_code >= 500:
                raise RuntimeError(f"{r.status_code} server error: {r.text[:200]}")
            return r
        except Exception as e:
            last_exc = e
            if attempt < cfg.net_max_retries - 1:
                await _sleep_backoff(attempt, cfg)
            else:
                break
    raise RuntimeError(f"POST {url} failed: {last_exc}")

async def _get(client: httpx.AsyncClient, url: str, params: Dict[str, Any], cfg: AutoTraderConfig) -> httpx.Response:
    last_exc: Optional[Exception] = None
    for attempt in range(cfg.net_max_retries):
        try:
            r = await client.get(url, params=params, timeout=cfg.http_timeout_sec)
            if r.status_code >= 500:
                raise RuntimeError(f"{r.status_code} server error: {r.text[:200]}")
            return r
        except Exception as e:
            last_exc = e
            if attempt < cfg.net_max_retries - 1:
                await _sleep_backoff(attempt, cfg)
            else:
                break
    raise RuntimeError(f"GET {url} failed: {last_exc}")

async def _fetch_candles(
    client: httpx.AsyncClient, symbol: str, timeframe: str, limit: int, cfg: AutoTraderConfig
) -> List[Dict[str, Any]]:
    """
    /ohlcv/prices/query (POST JSON)
    допускает 2 формата ответа:
      1) {"rows":[{t,o,h,l,c,v}, ...]}
      2) [{t,o,h,l,c,v}, ...]
    """
    url = f"{cfg.api_base}/ohlcv/prices/query"
    payload = {
        "source": cfg.source,
        "symbol": symbol,
        "timeframe": timeframe,
        "limit": limit,
        "testnet": bool(cfg.testnet),
    }
    r = await _post_json(client, url, payload, cfg)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return data["rows"]
    if isinstance(data, list):
        return data
    raise RuntimeError(f"Unexpected /ohlcv/prices/query response: {type(data)}")

async def _fetch_equity(client: httpx.AsyncClient, cfg: AutoTraderConfig) -> float:
    """
    /exec/balance -> берем equity_usdt (если нет — 0)
    """
    url = f"{cfg.api_base}/exec/balance"
    params = {"mode": cfg.mode, "testnet": str(cfg.testnet).lower(), "fast": "false"}
    r = await _get(client, url, params, cfg)
    r.raise_for_status()
    data = r.json()
    try:
        return float(data.get("equity_usdt") or 0.0)
    except Exception:
        return 0.0

async def _fetch_free_base_asset(client: httpx.AsyncClient, symbol: str, cfg: AutoTraderConfig) -> float:
    """
    Для spot продажи: узнаём, сколько базовой валюты (например BTC в BTCUSDT) свободно.
    """
    base = symbol.replace("USDT", "")
    url = f"{cfg.api_base}/exec/balance"
    params = {"mode": cfg.mode, "testnet": str(cfg.testnet).lower(), "fast": "true"}
    r = await _get(client, url, params, cfg)
    r.raise_for_status()
    data = r.json()
    balances = data.get("balances") or []
    free = 0.0
    for b in balances:
        if str(b.get("asset")).upper() == base:
            try:
                free = float(b.get("free") or 0.0)
            except Exception:
                free = 0.0
            break
    return max(0.0, free)


async def _fetch_positions(client: httpx.AsyncClient, cfg: AutoTraderConfig) -> List[Dict[str, Any]]:
    url = f"{cfg.api_base}/exec/positions"
    params = {"mode": cfg.mode, "testnet": str(cfg.testnet).lower()}
    r = await _get(client, url, params, cfg)
    if r.status_code >= 400:
        return []
    try:
        data = r.json()
    except Exception:
        return []
    if isinstance(data, dict) and "positions" in data:
        positions = data.get("positions") or []
        if isinstance(positions, list):
            return positions
    if isinstance(data, list):
        return data
    return []

async def _place_market_buy(
    client: httpx.AsyncClient,
    symbol: str,
    quote_usdt: float,
    *,
    sl_pct: float,
    tp_pct: float,
    cfg: AutoTraderConfig,
) -> Dict[str, Any]:
    """
    /exec/open (MARKET BUY).
    Если USE_RISK_AUTOSIZE=True и SL_PCT>0: не передаем qty/quote_qty — сервер сам рассчитает qty по риску.
    В остальных случаях передаём quote_qty (сумма в USDT).
    """
    url = f"{cfg.api_base}/exec/open"
    params = {
        "mode": cfg.mode,
        "testnet": str(cfg.testnet).lower(),
        "symbol": symbol,
        "side": "buy",
        "type": "market",
    }
    if cfg.use_risk_autosize and sl_pct > 0:
        # ничего не добавляем (qty/quote_qty), но передадим sl_pct/tp_pct
        pass
    else:
        params["quote_qty"] = f"{quote_usdt}"

    if sl_pct > 0:
        params["sl_pct"] = f"{sl_pct}"
    if tp_pct > 0:
        params["tp_pct"] = f"{tp_pct}"

    if cfg.dry_run:
        return {"status": "DRY", "params": params}

    r = await client.post(url, params=params, timeout=cfg.http_timeout_sec)
    if r.status_code == 409:
        try:
            detail = r.json()
        except Exception:
            detail = {"error": "risk_blocked", "detail": r.text[:200]}
        return {"status": "BLOCKED", "detail": detail}
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text[:300]
        raise RuntimeError(f"open BUY failed {r.status_code}: {detail}")
    payload = r.json()
    payload.setdefault("status", "UNKNOWN")
    return payload

async def _place_market_sell(
    client: httpx.AsyncClient, symbol: str, qty: float, cfg: AutoTraderConfig
) -> Dict[str, Any]:
    """
    /exec/open (MARKET SELL) для спота — просто рыночная продажа qty базового актива.
    """
    if qty <= 0:
        raise RuntimeError("sell qty must be positive")

    url = f"{cfg.api_base}/exec/open"
    params = {
        "mode": cfg.mode,
        "testnet": str(cfg.testnet).lower(),
        "symbol": symbol,
        "side": "sell",
        "type": "market",
        "qty": f"{qty}",
    }
    if cfg.dry_run:
        return {"status": "DRY", "params": params}
    # Для SELL SL/TP не имеет смысла в споте; опускаем.
    r = await client.post(url, params=params, timeout=cfg.http_timeout_sec)
    if r.status_code >= 400:
        raise RuntimeError(f"open SELL failed {r.status_code}: {r.text[:300]}")
    payload = r.json()
    payload.setdefault("status", "UNKNOWN")
    return payload

def _base_from_symbol(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def _calc_quote_amount(*, equity: float, sl_pct: float, cfg: RiskConfig, settings: AutoTraderConfig) -> float:
    if equity <= 0:
        return 0.0
    base_cap = equity
    if settings.quote_usdt > 0:
        base_cap = min(base_cap, settings.quote_usdt)
    if sl_pct <= 0:
        return max(0.0, min(base_cap, equity))
    risk_fraction = cfg.risk_fraction_for_trade(sl_pct)
    if risk_fraction <= 0:
        return 0.0
    allowed = equity * risk_fraction
    return max(0.0, min(base_cap, allowed))

# ═════════════════════════════════════════════════════════════════════════════
# Основная логика по символу (один шаг цикла)
# ═════════════════════════════════════════════════════════════════════════════
async def _process_symbol(
    client: httpx.AsyncClient, symbol: str, cfg: AutoTraderConfig, state: AutoTraderState
) -> None:
    try:
        limit = max(cfg.sma_slow + 10, 200)
        candles = await _fetch_candles(client, symbol, cfg.timeframe, limit, cfg)
        if len(candles) < (cfg.sma_slow + 2):
            LOG.info("[auto] %s %s — мало свечей: %d", symbol, cfg.timeframe, len(candles))
            return

        df = _rows_to_df(candles)
        if df.empty or len(df) < (cfg.sma_slow + 2):
            LOG.info("[auto] %s %s — недостаточно данных для сигналов (%d)", symbol, cfg.timeframe, len(df))
            return

        sig, last_closed_ts = _last_signal(df, cfg)

        if (
            cfg.confirm_on_close
            and state.last_candle_ts.get(symbol) == last_closed_ts
            and state.last_signal.get(symbol) == sig
        ):
            return

        state.last_candle_ts[symbol] = last_closed_ts
        state.last_signal[symbol] = sig

        close_price = float(df["close"].iloc[-1])
        sl_pct_dynamic = _atr_stop_pct(df, cfg)
        tp_pct = cfg.tp_pct if cfg.tp_pct > 0 else 0.0

        if sig == +1:
            equity = await _fetch_equity(client, cfg)
            if equity <= 0:
                LOG.warning("[auto] %s — не удалось получить equity, пропуск", symbol)
                return
            if equity < max(cfg.equity_min_usdt, 1.0) and not (cfg.use_risk_autosize and sl_pct_dynamic > 0):
                LOG.info(
                    "[auto] %s — equity %.2f ниже порога %.2f",
                    symbol,
                    equity,
                    max(cfg.equity_min_usdt, 1.0),
                )
                return

            positions = await _fetch_positions(client, cfg)
            snapshot = _risk_snapshot_from_positions(positions, equity=equity, settings=cfg)

            if cfg.use_portfolio_risk:
                if len(snapshot.positions) >= snapshot.config.max_open_positions:
                    have_symbol = any(str(p.get("symbol", "")).upper() == symbol.upper() for p in snapshot.positions)
                    if not have_symbol:
                        LOG.info("[auto] %s — достигнут лимит позиций (%d)", symbol, snapshot.config.max_open_positions)
                        return
                next_risk = snapshot.config.risk_fraction_for_trade(sl_pct_dynamic)
                portfolio_cap = (
                    cfg.max_portfolio_risk if cfg.max_portfolio_risk > 0 else snapshot.config.portfolio_max_risk_pct
                )
                if snapshot.portfolio_used + next_risk > portfolio_cap + 1e-6:
                    LOG.info("[auto] %s — портфельный риск %.3f > лимита %.3f, пропуск",
                             symbol, snapshot.portfolio_used + next_risk, portfolio_cap)
                    return

            quote_amt = _calc_quote_amount(
                equity=equity,
                sl_pct=sl_pct_dynamic,
                cfg=snapshot.config,
                settings=cfg,
            )
            if quote_amt <= 0 and not (cfg.use_risk_autosize and sl_pct_dynamic > 0):
                LOG.info("[auto] %s — рассчитанный объём сделки <= 0 (equity=%.2f, sl_pct=%.4f)",
                         symbol, equity, sl_pct_dynamic)
                return

            try:
                res = await _place_market_buy(
                    client,
                    symbol,
                    quote_amt,
                    sl_pct=sl_pct_dynamic,
                    tp_pct=tp_pct,
                    cfg=cfg,
                )
            except RuntimeError as exc:
                LOG.warning("[auto] BUY %s ошибка: %s", symbol, exc)
                state.last_error = repr(exc)
                _RUNTIME_STATUS["last_error"] = state.last_error
                return

            status = str(res.get("status") or "").upper()
            if status == "BLOCKED":
                LOG.warning("[auto] BUY %s отклонён риском: %s", symbol, res.get("detail"))
                state.last_error = f"risk_blocked:{res.get('detail')}"
                _RUNTIME_STATUS["last_error"] = state.last_error
                return

            LOG.info(
                "[auto] BUY %s %s price≈%.4f quote≈%.2f sl_pct=%.4f tp_pct=%.4f -> %s (order_id=%s)",
                symbol,
                cfg.timeframe,
                close_price,
                quote_amt,
                sl_pct_dynamic,
                tp_pct,
                status or res.get("status"),
                res.get("order_id"),
            )
            state.last_trade = {
                "symbol": symbol,
                "side": "buy",
                "quote": float(quote_amt),
                "sl_pct": float(sl_pct_dynamic),
                "tp_pct": float(tp_pct),
                "status": status or res.get("status"),
                "order_id": res.get("order_id"),
                "ts": int(time.time()),
            }
            state.last_error = None
            _RUNTIME_STATUS.update({"last_trade": state.last_trade, "last_error": None})
            return

        if sig == -1 and cfg.sell_on_death:
            free_base = await _fetch_free_base_asset(client, symbol, cfg)
            sell_qty = max(0.0, free_base * cfg.sell_pct_of_position)
            if sell_qty <= 0:
                return
            try:
                res = await _place_market_sell(client, symbol, sell_qty, cfg)
            except RuntimeError as exc:
                LOG.warning("[auto] SELL %s ошибка: %s", symbol, exc)
                state.last_error = repr(exc)
                _RUNTIME_STATUS["last_error"] = state.last_error
                return
            LOG.info(
                "[auto] SELL %s qty≈%.8f -> %s (order_id=%s)",
                symbol,
                sell_qty,
                res.get("status"),
                res.get("order_id"),
            )
            state.last_trade = {
                "symbol": symbol,
                "side": "sell",
                "qty": float(sell_qty),
                "status": res.get("status"),
                "order_id": res.get("order_id"),
                "ts": int(time.time()),
            }
            state.last_error = None
            _RUNTIME_STATUS.update({"last_trade": state.last_trade, "last_error": None})

    except asyncio.CancelledError:
        raise
    except Exception as e:
        LOG.warning("[auto] %s ошибка шага: %r", symbol, e)
        state.last_error = repr(e)
        _RUNTIME_STATUS["last_error"] = state.last_error

# ═════════════════════════════════════════════════════════════════════════════
# Публичная фон.корутина — добавьте в main.py
# ═════════════════════════════════════════════════════════════════════════════
async def background_loop(
    config: Optional[AutoTraderConfig] = None,
    *,
    state: Optional[AutoTraderState] = None,
) -> None:
    """
    Подключение в main.py:

        from tasks.auto_trader import background_loop

        @app.on_event("startup")
        async def start_auto_trader():
            asyncio.create_task(background_loop())

    Управление — через ENV (см. переменные вверху файла) или передачей кастомной конфигурации.
    """
    cfg = config or CONFIG
    run_state = state or AutoTraderState()

    if not cfg.enabled:
        LOG.warning("[auto] авто-трейдер выключен (enabled=0).")
        _RUNTIME_STATUS.update({"running": False, "config": cfg.to_public_dict()})
        return

    if cfg.sma_fast <= 1 or cfg.sma_slow <= 2 or cfg.sma_fast >= cfg.sma_slow:
        LOG.error("[auto] Некорректные SMA: fast=%d slow=%d (fast < slow и >=2)", cfg.sma_fast, cfg.sma_slow)
        _RUNTIME_STATUS.update({"running": False, "config": cfg.to_public_dict(), "last_error": "bad_sma_config"})
        return

    LOG.info(
        "[auto] старт: symbols=%s tf=%s ema(%d/%d) loop=%.1fs quote_cap=%.2f testnet=%s dry=%s risk_auto=%s "
        "atr=(period=%d mult=%.2f) sell_on_death=%s api=%s exec=%s/%s",
        ",".join(cfg.symbols),
        cfg.timeframe,
        cfg.sma_fast,
        cfg.sma_slow,
        cfg.loop_sec,
        cfg.quote_usdt,
        cfg.testnet,
        cfg.dry_run,
        cfg.use_risk_autosize,
        cfg.atr_period,
        cfg.atr_mult,
        cfg.sell_on_death,
        cfg.api_base,
        cfg.mode,
        cfg.source,
    )
    if cfg.use_risk_autosize and cfg.sl_pct <= 0 and (cfg.atr_period <= 0 or cfg.atr_mult <= 0):
        LOG.warning("[auto] USE_RISK_AUTOSIZE включен, но стоп не задан (SL_PCT/ATR) — qty не будет рассчитан.")

    run_state.started_at = time.time()
    run_state.iteration = 0
    run_state.last_error = None
    _RUNTIME_STATUS.update(
        {
            "running": True,
            "config": cfg.to_public_dict(),
            "started_at": run_state.started_at,
            "iteration": 0,
            "last_cycle_started": None,
            "last_cycle_finished": None,
            "last_error": None,
        }
    )

    sem = asyncio.Semaphore(cfg.concurrency)

    async def _guarded(sym: str) -> None:
        async with sem:
            await _process_symbol(client, sym, cfg, run_state)

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            while True:
                run_state.iteration += 1
                cycle_start = time.time()
                run_state.last_cycle_started = cycle_start
                _RUNTIME_STATUS.update(
                    {
                        "iteration": run_state.iteration,
                        "last_cycle_started": cycle_start,
                    }
                )
                try:
                    if not cfg.symbols:
                        await asyncio.sleep(max(1.0, cfg.loop_sec))
                        continue
                    tasks = [asyncio.create_task(_guarded(s)) for s in cfg.symbols]
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=False)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    LOG.error("[auto] ошибка итерации: %r", e)
                    run_state.last_error = repr(e)
                    _RUNTIME_STATUS["last_error"] = run_state.last_error
                finally:
                    run_state.last_cycle_finished = time.time()
                    _RUNTIME_STATUS["last_cycle_finished"] = run_state.last_cycle_finished

                took = run_state.last_cycle_finished - cycle_start
                pause = max(1.0, cfg.loop_sec - min(max(cfg.loop_sec - 1.0, 0.0), took))
                await asyncio.sleep(pause)
        except asyncio.CancelledError:
            LOG.info("[auto] остановка по CancelledError")
            raise
        except Exception as e:
            LOG.exception("[auto] критическая ошибка цикла: %r", e)
            run_state.last_error = repr(e)
            _RUNTIME_STATUS["last_error"] = run_state.last_error
        finally:
            _RUNTIME_STATUS["running"] = False
            _RUNTIME_STATUS["last_cycle_finished"] = time.time()
            LOG.info("[auto] фон-петля завершена")
