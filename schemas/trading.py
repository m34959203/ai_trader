from __future__ import annotations

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Сигналы
# ──────────────────────────────────────────────────────────────────────────────

class Signal(BaseModel):
    timestamp: Optional[int] = Field(
        default=None,
        description="UNIX ms, если доступно (иначе None)"
    )
    close: float = Field(..., description="Цена закрытия бара")
    ema_fast: float = Field(..., description="Быстрая EMA")
    ema_slow: float = Field(..., description="Медленная EMA")
    signal: int = Field(..., description="−1 = sell, 0 = flat, +1 = buy")


class SignalsResponse(BaseModel):
    signals: List[Dict[str, Any]]
    mode: Optional[str] = Field(None, description="Какой генератор сигналов использовался (ema|ensemble)")


# ──────────────────────────────────────────────────────────────────────────────
# Бэктест: запрос
# ──────────────────────────────────────────────────────────────────────────────

class BacktestRequest(BaseModel):
    source: str = Field(..., description="Источник котировок (например, 'alpha_vantage' или 'binance')")
    symbol: str = Field(..., description="Тикер/пара (например, 'BTC/USDT' или 'AAPL')")
    tf: str = Field("1h", description="Таймфрейм, напр. 1m/5m/15m/1h/4h/1d")
    fast: int = Field(12, ge=1, description="Параметр быстрой EMA")
    slow: int = Field(26, ge=2, description="Параметр медленной EMA")
    limit: int = Field(200, ge=50, description="Сколько баров загрузить/использовать")
    sl_pct: float = Field(0.02, ge=0.0, description="Стоп-лосс в долях (2% = 0.02)")
    tp_pct: float = Field(0.04, ge=0.0, description="Тейк-профит в долях (4% = 0.04)")
    fee_pct: float = Field(0.001, ge=0.0, description="Комиссия в долях (0.1% = 0.001)")
    start_equity: float = Field(10_000.0, gt=0.0, description="Начальный капитал для кривой капитала")
    side: Literal["long_only", "short_only", "both"] = Field(
        "long_only",
        description="Разрешённые стороны торговли"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Бэктест: структуры ответа (подробные)
# ──────────────────────────────────────────────────────────────────────────────

class Trade(BaseModel):
    entry_ts: int = Field(..., description="UNIX ms входа")
    exit_ts: int = Field(..., description="UNIX ms выхода")
    entry_price: float = Field(..., description="Цена входа")
    exit_price: float = Field(..., description="Цена выхода")
    qty: float = Field(..., description="Количество/объём позиции")
    side: Literal["long", "short"] = Field(..., description="Сторона сделки")
    pnl: float = Field(..., description="Абсолютный PnL сделки")
    ret_pct: float = Field(..., description="Доходность сделки в долях (0.05 = +5%)")
    fees: float = Field(0.0, description="Совокупные комиссии по сделке")
    sl_hit: bool = Field(False, description="Сработал ли стоп-лосс")
    tp_hit: bool = Field(False, description="Сработал ли тейк-профит")
    bars_held: Optional[int] = Field(None, description="Сколько баров удерживалась позиция")
    notes: Optional[str] = Field(None, description="Причина выхода/комментарии")


class EquityPoint(BaseModel):
    ts: int = Field(..., description="UNIX ms на конец бара/сделки")
    equity: float = Field(..., description="Значение капитала после применённых сделок")


class BacktestSummary(BaseModel):
    symbol: str = Field(..., description="Тикер/пара")
    tf: str = Field(..., description="Таймфрейм")
    start_equity: float = Field(..., description="Начальный капитал")
    end_equity: float = Field(..., description="Итоговый капитал")
    pnl_sum: float = Field(..., description="Суммарный PnL (денежный)")
    pnl_pct: float = Field(..., description="Суммарная доходность в долях от start_equity")
    n_trades: int = Field(..., ge=0, description="Количество закрытых сделок")
    win_rate: float = Field(..., ge=0.0, le=1.0, description="Доля прибыльных сделок (0.55 = 55%)")
    max_dd: float = Field(..., description="Максимальная просадка по капиталу в долях (0.2 = −20%)")
    profit_factor: Optional[float] = Field(None, description="Сумма прибыли / сумма убытков (>1 лучше)")
    avg_win: Optional[float] = Field(None, description="Средний выигрыш на сделку (денежный)")
    avg_loss: Optional[float] = Field(None, description="Средний проигрыш на сделку (денежный)")
    expectancy: Optional[float] = Field(
        None,
        description="Ожидаемость на сделку (средний PnL), может быть в деньгах или в %"
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Использованные параметры стратегии (fast, slow, sl_pct, tp_pct, fee_pct, side и т.д.)"
    )


class BacktestResponse(BaseModel):
    summary: BacktestSummary
    trades: List[Trade]
    equity_curve: List[EquityPoint]
