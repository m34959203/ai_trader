# ai_trader/routers/trading.py
from __future__ import annotations

import logging
import json
import time
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Tuple

import math
import pandas as pd
from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session
from db import crud
from schemas.trading import (
    SignalsResponse,
    BacktestRequest,
    BacktestResponse,
    BacktestSummary,
    Trade,
    EquityPoint,
)
from src.strategy import (
    ema_cross_signals,
    load_strategy_config,
    run_configured_ensemble,
)
from src.paper import PaperTrader
from src.slippage_model import SlippageModel, create_realistic_model

LOG = logging.getLogger("ai_trader.trading")
router = APIRouter(tags=["trading"])

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_OHLCV = ("source", "asset", "tf", "ts", "open", "high", "low", "close", "volume")


def _coerce_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _rows_to_df(rows: Iterable[Any]) -> pd.DataFrame:
    """
    Конвертирует ORM-объекты (OHLCV) в DataFrame с ожидаемыми колонками.
    - пропускает битые строки и логирует предупреждение
    - приводит числовые типы
    - сортирует по ts и удаляет дубликаты
    """
    if not rows:
        return pd.DataFrame(columns=REQUIRED_OHLCV)

    data: List[Dict[str, Any]] = []
    bad = 0
    for r in rows:
        try:
            ts = int(r.ts)
            o = _coerce_float(r.open)
            h = _coerce_float(r.high)
            l = _coerce_float(r.low)
            c = _coerce_float(r.close)
            v = _coerce_float(getattr(r, "volume", 0.0))
            if None in (o, h, l, c):
                raise ValueError("NaN in OHLC")
            data.append(
                {
                    "source": str(r.source),
                    "asset": str(r.asset),
                    "tf": str(r.tf),
                    "ts": ts,
                    "open": float(o),
                    "high": float(h),
                    "low": float(l),
                    "close": float(c),
                    "volume": float(v if v is not None else 0.0),
                }
            )
        except Exception as e:  # pragma: no cover — защита от неожиданных типов
            bad += 1
            LOG.warning("Bad OHLCV row skipped: %r (%s)", r, e)

    if bad:
        LOG.info("Skipped %d malformed OHLCV rows", bad)

    df = pd.DataFrame(data, columns=REQUIRED_OHLCV)
    if df.empty:
        return df

    # Сортировка, дедупликация по ts
    df = df.dropna(subset=["ts", "close"]).copy()
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["ts"]).copy()
    df["ts"] = df["ts"].astype(int)
    df = df.drop_duplicates(subset=["ts"], keep="last").sort_values("ts")

    # Числовые поля → float
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def _iso_from_ts_col(df: pd.DataFrame, col: str = "ts") -> pd.Series:
    return pd.to_datetime(df[col], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_signals_df(df: pd.DataFrame) -> None:
    need = {"ts", "close", "signal"}
    if not need.issubset(df.columns):
        missing = list(need.difference(df.columns))
        raise ValueError(f"Signals DF missing columns: {missing}")


def _ensure_min_bars(df: pd.DataFrame, fast: int, slow: int) -> None:
    """
    Проверка достаточности истории для EMA.
    Берём запас (slow + 5), чтобы сигналы были устойчивее.
    """
    min_need = max(fast, slow) + 5
    if len(df) < min_need:
        raise HTTPException(
            status_code=400,
            detail=f"Not enough bars for EMA({fast},{slow}). Need >= {min_need}, got {len(df)}.",
        )


def _compute_profit_factor(profits: List[float], losses: List[float]) -> Optional[float]:
    sum_profit = sum(p for p in profits if p > 0)
    sum_loss = -sum(l for l in losses if l < 0)
    if sum_loss <= 0:
        # Без убытков PF не определён (или бесконечность) — вернём None
        return None
    return sum_profit / sum_loss if sum_loss > 0 else None


def _max_drawdown(equity: List[float]) -> float:
    """
    Максимальная просадка в долях (0.2 = −20%)
    """
    if not equity:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for x in equity:
        if x > peak:
            peak = x
        dd = (peak - x) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def _build_equity_curve(
    start_equity: float,
    trades: List[Dict[str, Any]],
    first_ts: Optional[int],
) -> List[EquityPoint]:
    """
    Формирует кривую капитала:
      - начальная точка на first_ts (если есть),
      - затем точки на exit_ts каждой сделки (кумулятивный PnL).
    """
    eq_points: List[EquityPoint] = []
    equity = start_equity
    if first_ts is not None:
        eq_points.append(EquityPoint(ts=int(first_ts), equity=float(equity)))

    for t in trades:
        pnl = float(t.get("pnl", 0.0))
        exit_ts = t.get("exit_ts")
        if exit_ts is None:
            # если сделка ещё не закрыта — пропускаем
            continue
        equity += pnl
        eq_points.append(EquityPoint(ts=int(exit_ts), equity=float(equity)))

    return eq_points


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_trades_csv(path: Path, trades: List[Dict[str, Any]]) -> None:
    if not trades:
        # создаём пустой файл с заголовком для стабильности пайплайнов
        cols = [
            "entry_ts", "exit_ts", "entry_price", "exit_price", "qty", "side",
            "pnl", "ret_pct", "fees", "sl_hit", "tp_hit", "bars_held", "notes"
        ]
        pd.DataFrame(columns=cols).to_csv(path, index=False)
        return
    df = pd.DataFrame(trades)
    # порядок колонок
    cols_order = [
        "entry_ts", "exit_ts", "entry_price", "exit_price", "qty", "side",
        "pnl", "ret_pct", "fees", "sl_hit", "tp_hit", "bars_held", "notes"
    ]
    for c in cols_order:
        if c not in df.columns:
            df[c] = None
    df = df[cols_order]
    df.to_csv(path, index=False)


def _save_trades_json(path: Path, trades: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(trades or [], f, ensure_ascii=False, indent=2)


def _save_equity_csv(path: Path, equity_curve: List[EquityPoint]) -> None:
    rows = [{"ts": int(pt.ts), "equity": float(pt.equity)} for pt in (equity_curve or [])]
    pd.DataFrame(rows).to_csv(path, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# Sharpe ratio (annualized) по сделкам
# ──────────────────────────────────────────────────────────────────────────────

def _estimate_trades_per_year(n_trades: int, start_ts: Optional[int], end_ts: Optional[int], annualization_days: int) -> float:
    """
    Оценивает интенсивность сделок (кол-во сделок в год) по длительности теста.
    Если длительность < 1 дня — используем fallback на годовую нормировку по календарю.
    """
    if n_trades <= 0 or start_ts is None or end_ts is None or end_ts <= start_ts:
        return float(annualization_days)  # нейтральный fallback
    days = max(1.0, (end_ts - start_ts) / 86400.0)
    trades_per_day = n_trades / days
    return trades_per_day * float(annualization_days)


def _rolling_window_n_by_days(n_trades: int, start_ts: Optional[int], end_ts: Optional[int], window_days: int) -> int:
    """
    Перевод rolling window в днях в эквивалентное количество последних сделок, пропорционально плотности.
    """
    if n_trades <= 0 or start_ts is None or end_ts is None or end_ts <= start_ts:
        return n_trades
    total_days = max(1.0, (end_ts - start_ts) / 86400.0)
    frac = max(0.0, min(1.0, float(window_days) / total_days))
    # минимум 5 сделок для статистической устойчивости
    return max(5, int(round(n_trades * frac)))


def _sharpe_from_returns(returns: List[float], periods_per_year: float, risk_free_annual: float = 0.0) -> Optional[float]:
    """
    Sharpe = (mean(r - rf_per_period) / std(r - rf_per_period)) * sqrt(periods_per_year)
    где r — доходность за «период» (здесь — сделка), periods_per_year — оценка «периодов в год».
    """
    if not returns:
        return None
    mu = float(pd.Series(returns).mean())
    sigma = float(pd.Series(returns).std(ddof=1))  # sample std
    if sigma <= 0:
        return None
    rf_per_period = float(risk_free_annual) / float(periods_per_year) if periods_per_year > 0 else 0.0
    adj = [r - rf_per_period for r in returns]
    mu_adj = float(pd.Series(adj).mean())
    sigma_adj = float(pd.Series(adj).std(ddof=1))
    if sigma_adj <= 0:
        return None
    return (mu_adj / sigma_adj) * math.sqrt(periods_per_year)


def _compute_sharpe_metrics(
    trades: List[Dict[str, Any]],
    *,
    df_ts_start: Optional[int],
    df_ts_end: Optional[int],
    window_days: int,
    annualization_days: int,
    risk_free_annual: float = 0.0,
) -> Dict[str, Optional[float]]:
    """
    Возвращает:
      {
        'sharpe_annualized': float|None,
        'sharpe_annualized_window': float|None
      }
    Считается по ret_pct из сделок. Rolling window — по пропорции длительности периода в днях.
    """
    if not trades:
        return {"sharpe_annualized": None, "sharpe_annualized_window": None}

    rets = [float(t.get("ret_pct", 0.0)) for t in trades if t.get("ret_pct") is not None]
    n = len(rets)
    if n == 0:
        return {"sharpe_annualized": None, "sharpe_annualized_window": None}

    periods_per_year = _estimate_trades_per_year(n, df_ts_start, df_ts_end, annualization_days)
    sharpe_all = _sharpe_from_returns(rets, periods_per_year, risk_free_annual=risk_free_annual)

    # rolling window → число последних сделок
    win_n = _rolling_window_n_by_days(n, df_ts_start, df_ts_end, window_days)
    rets_win = rets[-win_n:] if win_n < n else rets
    sharpe_win = _sharpe_from_returns(rets_win, periods_per_year, risk_free_annual=risk_free_annual)

    return {
        "sharpe_annualized": (None if sharpe_all is None or math.isnan(sharpe_all) else float(sharpe_all)),
        "sharpe_annualized_window": (None if sharpe_win is None or math.isnan(sharpe_win) else float(sharpe_win)),
    }


def _make_backtest_summary(
    *,
    req: BacktestRequest,
    trades: List[Dict[str, Any]],
    start_equity: float,
    equity_curve: List[EquityPoint],
    artifacts: Optional[Dict[str, Optional[str]]] = None,
    extra_params: Optional[Dict[str, Any]] = None,
) -> BacktestSummary:
    n = len(trades)
    profits = [float(t.get("pnl", 0.0)) for t in trades if float(t.get("pnl", 0.0)) > 0]
    losses = [float(t.get("pnl", 0.0)) for t in trades if float(t.get("pnl", 0.0)) < 0]
    pnl_sum = sum(profits) + sum(losses)
    wins = len(profits)
    win_rate = (wins / n) if n > 0 else 0.0
    end_equity = equity_curve[-1].equity if equity_curve else start_equity
    pnl_pct = (end_equity - start_equity) / start_equity if start_equity > 0 else 0.0
    max_dd = _max_drawdown([pt.equity for pt in equity_curve]) if equity_curve else 0.0

    avg_win = (sum(profits) / len(profits)) if profits else None
    avg_loss = (sum(losses) / len(losses)) if losses else None
    expectancy = (pnl_sum / n) if n > 0 else None
    profit_factor = _compute_profit_factor(profits, losses)

    params: Dict[str, Any] = {
        "fast": req.fast,
        "slow": req.slow,
        "sl_pct": req.sl_pct,
        "tp_pct": req.tp_pct,
        "fee_pct": req.fee_pct,
        "tf": req.tf,
        "side": getattr(req, "side", "long_only"),
        "limit": req.limit,
    }
    # небьющаяся расширяемость: дополнительные параметры/артефакты — внутрь params
    if artifacts:
        params["artifacts"] = artifacts
    if extra_params:
        params.update(extra_params)

    return BacktestSummary(
        symbol=req.symbol,
        tf=req.tf,
        start_equity=float(start_equity),
        end_equity=float(end_equity),
        pnl_sum=float(pnl_sum),
        pnl_pct=float(pnl_pct),
        n_trades=int(n),
        win_rate=float(win_rate),
        max_dd=float(max_dd),
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=expectancy,
        params=params,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@router.get("/strategy/signals", response_model=SignalsResponse)
async def get_signals(
    source: str = Query(..., description="Источник (как в БД)"),
    symbol: str = Query(..., description="Тикер/пара, как в БД)"),
    tf: str = Query("1h", description="Таймфрейм, как в БД"),
    fast: int = Query(12, ge=1, description="Период быстрой EMA"),
    slow: int = Query(26, ge=1, description="Период медленной EMA"),
    limit: int = Query(360, ge=50, le=5000, description="Кол-во баров (рекомендуется >= slow+5)"),
    ensemble: bool = Query(False, description="Использовать ансамбль стратегий из configs/strategy.yaml"),
    strategy_config: Optional[str] = Query(
        None,
        description="Путь к YAML/JSON конфигу стратегий (по умолчанию configs/strategy.yaml)",
    ),
    session: AsyncSession = Depends(get_session),
):
    """
    Возвращает сигналы EMA-cross в разрезе времени (по свечам из БД).

    Примечания:
    - Если fast >= slow — автоматически меняем местами и логируем.
    - Требуется достаточно данных: len(df) >= max(fast, slow) + 5.
    """
    try:
        # 1) загрузка свечей
        rows = await crud.query_ohlcv(session, source=source, asset=symbol, tf=tf, limit=limit, order="asc")
        if not rows:
            return {"signals": []}

        df = _rows_to_df(rows)
        if df.empty:
            return {"signals": []}

        # 2) авто-правка параметров при перепутанных периодах
        if fast >= slow:
            LOG.info("Swapping fast/slow: fast=%s, slow=%s", fast, slow)
            fast, slow = slow, fast

        _ensure_min_bars(df, fast, slow)

        if ensemble:
            try:
                cfg_path = Path(strategy_config) if strategy_config else None
                cfg = load_strategy_config(cfg_path)
            except FileNotFoundError as e:
                raise HTTPException(status_code=404, detail=str(e))
            except Exception as e:  # pragma: no cover
                LOG.exception("Failed to load strategy config: %r", e)
                raise HTTPException(status_code=400, detail=f"Strategy config error: {e}")

            sigs = run_configured_ensemble(df, cfg)
            base_cols = df[["ts", "close"]].drop_duplicates(subset=["ts"])
            base_cols["timestamp"] = base_cols["ts"]
            sigs = sigs.merge(base_cols, on="ts", how="left")
            sigs["timestamp"] = pd.to_numeric(sigs.get("timestamp"), errors="coerce").fillna(sigs["ts"]).astype(np.int64)
            _ensure_signals_df(sigs)
            sigs["timestamp_iso"] = _iso_from_ts_col(sigs, "ts")
            out = sigs.to_dict(orient="records")
            LOG.info(
                "Signals ensemble %s/%s tf=%s bars=%d -> %d rows",
                source,
                symbol,
                tf,
                len(df),
                len(out),
            )
            return {"signals": out, "mode": "ensemble"}
        else:
            # 3) сигналы EMA
            sigs = ema_cross_signals(df, fast, slow).copy()
            _ensure_signals_df(sigs)

            # 4) добавим ISO-время для удобства UI/отладки
            sigs["timestamp_iso"] = _iso_from_ts_col(sigs, "ts")

            # 5) стабильная сериализация
            out = sigs.to_dict(orient="records")
            LOG.info(
                "Signals %s/%s tf=%s fast=%d slow=%d bars=%d -> %d rows",
                source, symbol, tf, fast, slow, len(df), len(out),
            )
            return {"signals": out, "mode": "ema"}

    except HTTPException:
        raise
    except AssertionError as e:
        LOG.exception("Signals assertion failed: %r", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        LOG.exception("GET /strategy/signals failed: %r", e)
        raise HTTPException(status_code=500, detail="Failed to compute signals")


@router.post("/paper/backtest", response_model=BacktestResponse)
async def run_backtest(
    req: BacktestRequest,
    # Новые флаги экспорта (query-параметры, чтобы не ломать тело запроса):
    save_trades_csv: bool = Query(False, description="Сохранить сделки в CSV"),
    save_trades_json: bool = Query(False, description="Сохранить сделки в JSON"),
    save_equity_csv: bool = Query(False, description="Сохранить кривую капитала в CSV"),
    out_dir: str = Query("data", description="Каталог для сохранения артефактов"),
    # Sharpe config:
    sharpe_window_days: int = Query(90, ge=1, le=3650, description="Rolling window (календарные дни) для Sharpe"),
    annualization_days: int = Query(252, ge=1, le=366, description="Число торговых дней в году для годовой Sharpe-нормировки"),
    risk_free_annual: float = Query(0.0, ge=0.0, le=0.2, description="Годовая ставка безрисковой доходности (0..0.2)"),
    session: AsyncSession = Depends(get_session),
):
    """
    Бэктест стратегии EMA-cross на исторических свечах из БД.

    Порядок симуляции по каждому бару:
      1) сначала проверяем SL/TP по high/low (интрабар события),
      2) затем обрабатываем сигнал (вход/выход) по цене close.

    В конце всегда закрываем незакрытую позицию по последнему close (reason="eod"),
    чтобы метрики были полными и консистентными.

    Дополнительно:
      • по флагам сохраняем сделки (CSV/JSON) и кривую капитала (CSV) в каталог out_dir,
        а пути возвращаем в summary.params.artifacts — так мы «расширяем» ответ,
        не меняя pydantic-схемы.
      • считаем Sharpe (annualized) по ret_pct сделок; rolling window — по последним
        N сделкам, эквивалентным `sharpe_window_days` длительности теста.
    """
    try:
        # 1) читаем историю из БД
        rows = await crud.query_ohlcv(
            session,
            source=req.source,
            asset=req.symbol,
            tf=req.tf,
            limit=req.limit,
            order="asc",
        )

        def _empty_response() -> BacktestResponse:
            empty_curve: List[EquityPoint] = []
            summary = BacktestSummary(
                symbol=req.symbol,
                tf=req.tf,
                start_equity=float(req.start_equity),
                end_equity=float(req.start_equity),
                pnl_sum=0.0,
                pnl_pct=0.0,
                n_trades=0,
                win_rate=0.0,
                max_dd=0.0,
                profit_factor=None,
                avg_win=None,
                avg_loss=None,
                expectancy=None,
                params={
                    "fast": req.fast, "slow": req.slow, "sl_pct": req.sl_pct,
                    "tp_pct": req.tp_pct, "fee_pct": req.fee_pct, "tf": req.tf,
                    "side": getattr(req, "side", "long_only"),
                    "limit": req.limit,
                    "artifacts": None,
                    "metrics": {"sharpe_annualized": None, "sharpe_annualized_window": None},
                },
            )
            return BacktestResponse(summary=summary, trades=[], equity_curve=empty_curve)

        if not rows:
            return _empty_response()

        df = _rows_to_df(rows)
        if df.empty:
            return _empty_response()

        # 2) параметры EMA
        fast = int(req.fast)
        slow = int(req.slow)
        if fast >= slow:
            LOG.info("Swapping fast/slow in backtest: fast=%s, slow=%s", fast, slow)
            fast, slow = slow, fast

        _ensure_min_bars(df, fast, slow)

        # 3) сигналы
        sigs = ema_cross_signals(df, fast, slow)
        _ensure_signals_df(sigs)

        # 4) объединяем для SL/TP
        merged = pd.merge(
            sigs,
            df[["ts", "high", "low"]],
            on="ts",
            how="left",
            validate="one_to_one",
        ).sort_values("ts")

        # 5) Слипpage модель для реалистичных fills
        slippage_model = create_realistic_model()

        # Рассчитываем ATR для slippage
        from src.indicators import atr as calc_atr
        df_with_atr = df.copy()
        df_with_atr['atr'] = calc_atr(df['high'], df['low'], df['close'], period=14)

        # 6) симулятор сделок
        trader = PaperTrader(
            sl_pct=float(req.sl_pct),
            tp_pct=float(req.tp_pct),
            fee_pct=float(req.fee_pct),
            trail_pct=getattr(req, "trail_pct", None) if hasattr(req, "trail_pct") else None,
            qty=getattr(req, "qty", 1.0) if hasattr(req, "qty") else 1.0,
        )

        for idx, row in merged.iterrows():
            ts = int(row["ts"])
            price_close = float(row["close"])
            # если high/low нет, берём close
            high = float(row["high"]) if pd.notna(row["high"]) else price_close
            low = float(row["low"]) if pd.notna(row["low"]) else price_close
            signal = int(row["signal"])

            # Сначала SL/TP (интрабар), затем обработка сигнала на close
            trader.check_sl_tp(ts, high=high, low=low)

            if signal != 0:
                # ИНТЕГРАЦИЯ SLIPPAGE: Реалистичная цена исполнения
                qty = trader.qty

                # Получаем ATR для данного бара
                atr_value = df_with_atr.loc[df_with_atr['ts'] == ts, 'atr'].values
                atr_val = float(atr_value[0]) if len(atr_value) > 0 else 0.0
                volatility = atr_val / price_close if price_close > 0 else 0.02

                # Средний volume за последние 20 баров
                current_idx = df[df['ts'] == ts].index[0]
                lookback_start = max(0, current_idx - 20)
                avg_volume = df.iloc[lookback_start:current_idx]['volume'].mean()
                if pd.isna(avg_volume) or avg_volume <= 0:
                    avg_volume = df['volume'].mean()

                # Детект gap events
                is_gap = False
                if current_idx > 0:
                    prev_close = df.iloc[current_idx - 1]['close']
                    gap_pct = abs(price_close - prev_close) / prev_close
                    is_gap = gap_pct > 0.02  # 2% gap threshold

                # Рассчитываем realistic fill price с slippage
                fill_price = slippage_model.calculate_fill_price(
                    entry_price=price_close,
                    quantity=qty,
                    avg_volume=avg_volume,
                    volatility=volatility,
                    side="buy" if signal > 0 else "sell",
                    is_market_order=True,
                    is_gap_event=is_gap,
                )

                # Логируем slippage для первых нескольких сделок
                if len(trader.trades) < 3:
                    slippage_pct = abs(fill_price - price_close) / price_close
                    LOG.info(
                        f"[SLIPPAGE] Signal={signal}, Close={price_close:.2f}, "
                        f"Fill={fill_price:.2f}, Slippage={slippage_pct:.3%}, "
                        f"Vol={volatility:.3%}, Gap={is_gap}"
                    )

                # Используем realistic fill price
                trader.on_signal(ts, fill_price, signal)

        # 6) гарантия закрытия позиции по последнему close
        if trader.position is not None and len(merged) > 0:
            last = merged.iloc[-1]
            trader.close(int(last["ts"]), float(last["close"]), reason="eod")

        # Получаем агрегатную сводку и сделки из PaperTrader (как dict-списки)
        raw_trades: List[Dict[str, Any]] = list(trader.trades or [])

        # 7) Кривая капитала (по закрытиям сделок)
        first_ts: Optional[int] = int(df.iloc[0]["ts"]) if len(df) else None
        last_ts: Optional[int] = int(df.iloc[-1]["ts"]) if len(df) else None
        equity_curve: List[EquityPoint] = _build_equity_curve(
            start_equity=float(req.start_equity),
            trades=raw_trades,
            first_ts=first_ts,
        )

        # 8) Sharpe metrics
        sharpe_metrics = _compute_sharpe_metrics(
            raw_trades,
            df_ts_start=first_ts,
            df_ts_end=last_ts,
            window_days=int(sharpe_window_days),
            annualization_days=int(annualization_days),
            risk_free_annual=float(risk_free_annual),
        )

        # 9) Экспорт артефактов (по флагам)
        artifacts: Dict[str, Optional[str]] = {}
        if save_trades_csv or save_trades_json or save_equity_csv:
            out_root = _ensure_dir(out_dir)
            ts_tag = int(time.time())
            base = f"{req.source}_{req.symbol}_{req.tf}_{fast}-{slow}_{len(df)}_{ts_tag}"
            try:
                if save_trades_csv:
                    p = out_root / f"{base}.trades.csv"
                    _save_trades_csv(p, raw_trades)
                    artifacts["trades_csv"] = str(p)
                if save_trades_json:
                    p = out_root / f"{base}.trades.json"
                    _save_trades_json(p, raw_trades)
                    artifacts["trades_json"] = str(p)
                if save_equity_csv:
                    p = out_root / f"{base}.equity.csv"
                    _save_equity_csv(p, equity_curve)
                    artifacts["equity_csv"] = str(p)
            except Exception as e:
                LOG.warning("Artifacts save failed: %r", e)

        # 10) Сводка + Sharpe в params.metrics (чтобы не менять pydantic-схему)
        summary = _make_backtest_summary(
            req=req,
            trades=raw_trades,
            start_equity=float(req.start_equity),
            equity_curve=equity_curve,
            artifacts=(artifacts or None) if artifacts else None,
            extra_params={"metrics": sharpe_metrics},
        )

        # 11) Нормируем сделки под схему Trade
        def _map_trade(t: Dict[str, Any]) -> Trade:
            return Trade(
                entry_ts=int(t.get("entry_ts")),
                exit_ts=int(t.get("exit_ts")),
                entry_price=float(t.get("entry_price")),
                exit_price=float(t.get("exit_price")),
                qty=float(t.get("qty", 1.0)),
                side=str(t.get("side", "long")),
                pnl=float(t.get("pnl", 0.0)),
                ret_pct=float(t.get("ret_pct", 0.0)),
                fees=float(t.get("fees", 0.0)),
                sl_hit=bool(t.get("sl_hit", False)),
                tp_hit=bool(t.get("tp_hit", False)),
                bars_held=int(t["bars_held"]) if t.get("bars_held") is not None else None,
                notes=t.get("notes"),
            )

        trades_out: List[Trade] = [_map_trade(t) for t in raw_trades]

        LOG.info(
            "Backtest %s/%s tf=%s bars=%d -> trades=%d winrate=%.2f pnl=%.6f dd=%.6f sharpe=%.4f win_sharpe=%.4f",
            req.source,
            req.symbol,
            req.tf,
            len(df),
            summary.n_trades,
            summary.win_rate,
            summary.pnl_sum,
            summary.max_dd,
            (sharpe_metrics.get("sharpe_annualized") or float("nan")),
            (sharpe_metrics.get("sharpe_annualized_window") or float("nan")),
        )

        return BacktestResponse(
            summary=summary,
            trades=trades_out,
            equity_curve=equity_curve,
        )

    except HTTPException:
        raise
    except AssertionError as e:
        LOG.exception("Backtest assertion failed: %r", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        LOG.exception("POST /paper/backtest failed: %r", e)
        raise HTTPException(status_code=500, detail="Failed to run backtest")
