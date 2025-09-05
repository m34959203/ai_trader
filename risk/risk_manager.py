from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, Protocol, Iterable, List

from utils.risk_config import load_risk_config
from state.daily_limits import (
    load_state, ensure_day, can_open_more_trades, register_new_trade,
    register_realized_pnl, daily_loss_hit, start_of_day_equity
)

log = logging.getLogger(__name__)

# ── Исполнительный интерфейс, чтобы не привязывать код к конкретной бирже ──
class IExecutor(Protocol):
    def get_equity(self) -> float: ...
    def get_price(self, symbol: str) -> float: ...
    def round_qty(self, symbol: str, qty: float) -> float: ...
    def place_entry(self, symbol: str, side: str, qty: float, order_type: str, price: Optional[float] = None) -> dict: ...
    def place_sl_tp(self, symbol: str, side: str, sl_price: float, tp_price: Optional[float], qty: float) -> None: ...
    def close_all_positions(self, reason: str) -> None: ...
    # Необязательный метод: список текущих позиций для расчёта портфельного риска
    def get_open_positions(self) -> Iterable["PositionLike"]: ...  # type: ignore[override]


# Представление открытой позиции — то, что нужно для оценки риска.
class PositionLike(Protocol):
    symbol: str
    side: str                # "BUY" / "SELL" (для спота риск симметричен, знак не важен)
    qty: float
    entry_price: float
    stop_loss_price: Optional[float]


@dataclass
class CheckResult:
    allowed: bool
    reason: str = ""          # короткий код причины отказа
    suggested_qty: float = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Н И З К О У Р О В Н Е В Ы Е   Р А С Ч Ё Т Ы
# ──────────────────────────────────────────────────────────────────────────────
def calc_position_size(
    equity: float,
    risk_pct: float,
    stop_loss_distance: float,
    *,
    min_qty: float = 0.0
) -> float:
    """
    Риск на сделку: equity * risk_pct = максимально допустимый убыток.
    Размер позиции ≈ макс.убыток / дистанция до SL (в ценовых единицах).
    Для спота: qty ≈ (equity * risk_pct) / |entry - SL|.
    """
    if risk_pct <= 0 or stop_loss_distance <= 0:
        return 0.0
    max_loss_abs = equity * risk_pct
    qty = max_loss_abs / float(stop_loss_distance)
    if min_qty and qty < min_qty:
        return 0.0
    return qty


def position_risk_fraction_from_params(
    *,
    equity: float,
    qty: float,
    entry_price: float,
    stop_loss_price: Optional[float],
    min_sl_distance_pct: float = 0.0
) -> float:
    """
    Оценивает долю капитала под риском для конкретной позиции.
    risk_fraction = |entry - SL| * qty / equity.

    Если SL не задан или дистанция < минимально допустимой — возвращает 1.0 (консервативно),
    чтобы заблокировать открытие/учесть максимальный риск.
    """
    if equity <= 0 or qty <= 0:
        return 0.0
    if stop_loss_price is None:
        return 1.0  # нет SL = считаем как 100% риск (консервативно)

    distance_abs = abs(entry_price - float(stop_loss_price))
    # минимально допустимая дистанция в абсолюте (из процента)
    min_dist_abs = float(min_sl_distance_pct) * float(entry_price)
    if distance_abs <= 0 or distance_abs < min_dist_abs:
        # слишком «тонкий» стоп = считаем его как неподходящий → риск максимальный
        return 1.0

    risk_abs = distance_abs * float(qty)
    return max(0.0, min(1.0, risk_abs / float(equity)))


def portfolio_risk_used(positions: Iterable[PositionLike], *, equity: float, min_sl_distance_pct: float) -> float:
    """
    Суммарная доля капитала под риском по всем открытым позициям.
    """
    total = 0.0
    for p in positions:
        try:
            total += position_risk_fraction_from_params(
                equity=equity,
                qty=float(p.qty),
                entry_price=float(p.entry_price),
                stop_loss_price=p.stop_loss_price,
                min_sl_distance_pct=min_sl_distance_pct,
            )
        except Exception:
            # если при чтении полей что-то пошло не так — лучше учесть максимум
            total += 1.0
    return max(0.0, min(1.0, total))


def can_open_new(
    *,
    new_position_risk_fraction: float,
    positions: Iterable[PositionLike],
    equity: float,
    portfolio_max_risk_pct: float,
    max_open_positions: int,
    min_sl_distance_pct: float,
) -> bool:
    """
    Проверка совокупного риска ≤ portfolio_max_risk_pct и лимита открытых позиций.
    """
    # лимит по числу открытых позиций
    try:
        open_count = sum(1 for _ in positions)
    except TypeError:
        # если positions — генератор-одноразовый, превратим в список
        positions = list(positions)
        open_count = len(list(positions))
    if open_count >= max_open_positions:
        return False

    used = portfolio_risk_used(positions, equity=equity, min_sl_distance_pct=min_sl_distance_pct)
    return (used + float(new_position_risk_fraction)) <= float(portfolio_max_risk_pct)


# ──────────────────────────────────────────────────────────────────────────────
# В Ы С О К О У Р О В Н Е В Ы Е   Х Е Л П Е Р Ы
# ──────────────────────────────────────────────────────────────────────────────
def pre_trade_check(
    executor: IExecutor,
    *,
    symbol: str,
    stop_loss_price: float,
    entry_price: float,
    side: str
) -> CheckResult:
    """
    Комплексная предторговая проверка:
      1) дневной стоп по equity,
      2) лимит сделок в день,
      3) минимально допустимая дистанция SL,
      4) расчёт размера позиции (per-trade risk),
      5) проверка портфельного риска ≤ 6% и лимита открытых позиций.
    """
    cfg = load_risk_config()
    equity = float(executor.get_equity())

    # 1–2) дневные лимиты
    state = ensure_day(load_state(cfg.tz_name), cfg.tz_name, current_equity=equity)
    if daily_loss_hit(state, cfg.daily_max_loss_pct, current_equity=equity):
        return CheckResult(False, reason=f"daily_loss_hit:{cfg.daily_max_loss_pct*100:.2f}%")
    if not can_open_more_trades(state, cfg.max_trades_per_day):
        return CheckResult(False, reason=f"trades_limit_reached:{state.trades_count}/{cfg.max_trades_per_day}")

    # 3) минимальная дистанция SL
    stop_distance_abs = abs(entry_price - stop_loss_price)
    min_dist_abs = cfg.min_sl_distance_pct * float(entry_price)
    if stop_distance_abs <= 0:
        return CheckResult(False, reason="invalid_stop_distance")
    if stop_distance_abs < min_dist_abs:
        return CheckResult(False, reason=f"sl_distance_below_min:{cfg.min_sl_distance_pct:.4f}")

    # 4) размер позиции из per-trade риска
    raw_qty = calc_position_size(
        equity=equity,
        risk_pct=cfg.risk_pct_per_trade,
        stop_loss_distance=stop_distance_abs,
    )
    qty = executor.round_qty(symbol, raw_qty)
    if qty <= 0:
        return CheckResult(False, reason="qty_too_small")

    # 5) проверка портфельного риска и лимита позиций
    #    доля риска новой позиции:
    new_risk_frac = position_risk_fraction_from_params(
        equity=equity,
        qty=qty,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        min_sl_distance_pct=cfg.min_sl_distance_pct,
    )

    # Получаем список открытых позиций, если исполнитель его предоставляет
    positions: Iterable[PositionLike] = []
    if hasattr(executor, "get_open_positions"):
        try:
            positions = list(executor.get_open_positions())  # type: ignore[attr-defined]
        except Exception:
            positions = []

    if not can_open_new(
        new_position_risk_fraction=new_risk_frac,
        positions=positions,
        equity=equity,
        portfolio_max_risk_pct=cfg.portfolio_max_risk_pct,
        max_open_positions=cfg.max_open_positions,
        min_sl_distance_pct=cfg.min_sl_distance_pct,
    ):
        return CheckResult(
            False,
            reason=f"portfolio_risk_or_positions_limit:{cfg.portfolio_max_risk_pct*100:.2f}%/{cfg.max_open_positions}"
        )

    return CheckResult(True, suggested_qty=qty)


def on_trade_opened() -> None:
    """Вызываем сразу после подтверждения входа в сделку (успешного исполнения entry)."""
    cfg = load_risk_config()
    state = ensure_day(load_state(cfg.tz_name), cfg.tz_name, current_equity=0.0)  # equity не нужен для регистрации сделки
    register_new_trade(state)


def on_trade_closed(realized_pnl: float, current_equity: Optional[float] = None) -> None:
    """
    Вызываем при закрытии позиции (реализованный PnL).
    current_equity — если доступен, позволит сразу перепроверить дневной стоп.
    """
    cfg = load_risk_config()
    state = load_state(cfg.tz_name)
    if state:
        register_realized_pnl(state, realized_pnl)
        if current_equity is not None and daily_loss_hit(state, cfg.daily_max_loss_pct, current_equity=current_equity):
            log.warning(
                "Daily max loss hit after trade close. start=%.2f, eq=%.2f",
                start_of_day_equity(state), current_equity
            )
