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


# ──────────────────────────────────────────────────────────────────────────────
# Walk-Forward Backtest Schemas
# ──────────────────────────────────────────────────────────────────────────────

class WalkForwardRequest(BaseModel):
    source: str = Field(..., description="Data source (e.g., 'binance', 'alpha_vantage')")
    symbol: str = Field(..., description="Trading pair/ticker (e.g., 'BTC/USDT', 'AAPL')")
    tf: str = Field("1h", description="Timeframe (e.g., 1m/5m/15m/1h/4h/1d)")

    # Walk-forward configuration
    train_window_days: int = Field(365, ge=30, description="Training window in days (e.g., 365 = 1 year)")
    test_window_days: int = Field(90, ge=7, description="Testing window in days (e.g., 90 = 3 months)")
    step_days: int = Field(30, ge=1, description="Step forward in days between iterations (e.g., 30 = 1 month)")
    use_anchored_walk: bool = Field(False, description="True = expanding window, False = rolling window")

    # Strategy parameters (base values, will be optimized if optimization enabled)
    fast: int = Field(12, ge=1, description="Base fast EMA parameter")
    slow: int = Field(26, ge=2, description="Base slow EMA parameter")
    optimize_params: bool = Field(True, description="Optimize EMA parameters on training data")

    # Risk management
    sl_pct: float = Field(0.02, ge=0.0, description="Stop-loss percentage (0.02 = 2%)")
    tp_pct: float = Field(0.04, ge=0.0, description="Take-profit percentage (0.04 = 4%)")
    fee_pct: float = Field(0.001, ge=0.0, description="Trading fee percentage (0.001 = 0.1%)")

    # General settings
    start_equity: float = Field(10_000.0, gt=0.0, description="Initial capital")
    limit: int = Field(1000, ge=200, description="Maximum bars to load from DB")
    side: Literal["long_only", "short_only", "both"] = Field("long_only", description="Allowed trade sides")


class WalkForwardIterationResult(BaseModel):
    """Result from a single walk-forward iteration."""
    iteration: int = Field(..., description="Iteration number (1-indexed)")

    # Time windows
    train_start: str = Field(..., description="Training period start (ISO format)")
    train_end: str = Field(..., description="Training period end (ISO format)")
    test_start: str = Field(..., description="Testing period start (ISO format)")
    test_end: str = Field(..., description="Testing period end (ISO format)")

    # Optimized parameters for this iteration
    optimized_params: Dict[str, Any] = Field(default_factory=dict, description="Parameters used in this iteration")

    # Training metrics
    train_sharpe: float = Field(..., description="Training Sharpe ratio")
    train_returns: float = Field(..., description="Training returns (fraction, 0.15 = 15%)")
    train_max_dd: float = Field(..., description="Training max drawdown (fraction, 0.10 = 10%)")
    train_win_rate: float = Field(..., description="Training win rate (0.60 = 60%)")
    train_trades: int = Field(..., description="Number of training trades")

    # Testing metrics (out-of-sample)
    test_sharpe: float = Field(..., description="Testing Sharpe ratio")
    test_returns: float = Field(..., description="Testing returns (fraction)")
    test_max_dd: float = Field(..., description="Testing max drawdown (fraction)")
    test_win_rate: float = Field(..., description="Testing win rate")
    test_trades: int = Field(..., description="Number of testing trades")

    # Degradation metrics
    sharpe_degradation: float = Field(..., description="Sharpe degradation from train to test")
    returns_degradation: float = Field(..., description="Returns degradation from train to test")


class WalkForwardSummary(BaseModel):
    """Aggregate summary of walk-forward test."""
    total_iterations: int = Field(..., description="Total number of iterations completed")

    # Configuration used
    train_window_days: int = Field(..., description="Training window size in days")
    test_window_days: int = Field(..., description="Testing window size in days")
    step_days: int = Field(..., description="Step size in days")
    use_anchored_walk: bool = Field(..., description="Whether anchored walk was used")

    # Aggregate test metrics
    avg_test_sharpe: float = Field(..., description="Average test Sharpe across all iterations")
    avg_test_returns: float = Field(..., description="Average test returns")
    avg_test_max_dd: float = Field(..., description="Average test max drawdown")
    avg_test_win_rate: float = Field(..., description="Average test win rate")

    # Degradation analysis
    avg_sharpe_degradation: float = Field(..., description="Average Sharpe degradation")
    avg_returns_degradation: float = Field(..., description="Average returns degradation")

    # Consistency metrics
    positive_test_periods: int = Field(..., description="Number of periods with positive returns")
    sharpe_above_1_periods: int = Field(..., description="Number of periods with Sharpe > 1")

    # Overfitting detection
    overfitting_detected: bool = Field(..., description="Whether overfitting was detected")
    overfitting_reason: str = Field(..., description="Reason for overfitting detection")

    # Overall assessment
    params: Dict[str, Any] = Field(default_factory=dict, description="Base parameters and settings")


class WalkForwardResponse(BaseModel):
    """Complete walk-forward backtest response."""
    summary: WalkForwardSummary = Field(..., description="Aggregate summary")
    iterations: List[WalkForwardIterationResult] = Field(..., description="Detailed results for each iteration")
