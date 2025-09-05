# state/daily_limits.py
from __future__ import annotations

import json
import math
import os
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime, date
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any, Tuple

log = logging.getLogger("ai_trader.daily_limits")

__all__ = [
    "DailyState",
    "load_state",
    "save_state",
    "ensure_day",
    "can_open_more_trades",
    "register_new_trade",
    "register_realized_pnl",
    "daily_loss_hit",
    "start_of_day_equity",
]

# ──────────────────────────────────────────────────────────────────────────────
# Storage
# ──────────────────────────────────────────────────────────────────────────────

STATE_DIR = Path(__file__).resolve().parents[1] / "data" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_FILE = STATE_DIR / "daily_limits.json"
STATE_BAK_FILE = STATE_DIR / "daily_limits.json.bak"

SCHEMA_VERSION = 1  # на будущее для возможных миграций


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


@dataclass
class DailyState:
    """
    Состояние дневных лимитов в локальной таймзоне:
      • day           — локальный день в формате YYYY-MM-DD
      • start_equity  — оценка equity на начало этого дня
      • trades_count  — сколько сделок уже совершено сегодня
      • realized_pnl  — накопленный реализованный PnL за день
      • version       — версия схемы хранения
    """
    day: str                      # YYYY-MM-DD локального дня
    start_equity: float           # equity на начало дня
    trades_count: int             # число сделок за день
    realized_pnl: float           # суммарный реализованный PnL за день
    version: int = field(default=SCHEMA_VERSION)

    @classmethod
    def new(cls, today: date, start_equity: float) -> "DailyState":
        return cls(
            day=today.isoformat(),
            start_equity=safe_float(start_equity, 0.0),
            trades_count=0,
            realized_pnl=0.0,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Time helpers
# ──────────────────────────────────────────────────────────────────────────────

def now_local(tz_name: str) -> datetime:
    return datetime.now(ZoneInfo(tz_name))

def today_local(tz_name: str) -> date:
    return now_local(tz_name).date()


# ──────────────────────────────────────────────────────────────────────────────
# IO helpers (robust load/save)
# ──────────────────────────────────────────────────────────────────────────────

def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            return None
        return json.loads(text)
    except Exception as e:
        log.warning("DailyState read failed (%s): %r", path.name, e)
        return None

def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        # atomically replace
        os.replace(tmp, path)
    except Exception as e:
        log.error("DailyState atomic write failed: %r", e)
        # best-effort cleanup
        try:
            if tmp.exists():
                tmp.unlink(missing_ok=True)  # type: ignore[arg-type]
        except Exception:
            pass
        raise

def load_state(tz_name: str) -> Optional[DailyState]:
    """
    Пытается загрузить состояние. При битом JSON — восстановит из .bak.
    """
    raw = _read_json(STATE_FILE)
    if raw is None:
        # попытаться восстановить из бэкапа
        bak = _read_json(STATE_BAK_FILE)
        if bak is not None:
            try:
                state = DailyState(**bak)
                # перепишем основной файл подтверждённым бэкапом
                try:
                    _atomic_write_json(STATE_FILE, asdict(state))
                except Exception:
                    pass
                return state
            except Exception:
                pass
        return None

    try:
        return DailyState(**raw)
    except Exception as e:
        log.warning("DailyState schema mismatch, trying backup: %r", e)
        # схема изменилась или файл повреждён → попытаться взять бэкап
        bak = _read_json(STATE_BAK_FILE)
        if bak:
            try:
                return DailyState(**bak)
            except Exception:
                pass
        return None

def save_state(state: DailyState) -> None:
    """
    Атомарно сохраняет состояние и поддерживает .bak.
    """
    data = asdict(state)
    # создаём бэкап текущего файла (если есть)
    try:
        if STATE_FILE.exists():
            STATE_BAK_FILE.write_text(STATE_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    except Exception as e:
        log.warning("DailyState backup write failed: %r", e)
    # основная запись
    _atomic_write_json(STATE_FILE, data)


# ──────────────────────────────────────────────────────────────────────────────
# Public API expected by routers/trading_exec.py
# ──────────────────────────────────────────────────────────────────────────────

def ensure_day(state: Optional[DailyState], tz_name: str, current_equity: float) -> DailyState:
    """
    Гарантирует, что состояние относится к сегодняшнему локальному дню.
    Если день сменился — инициализирует новый дневной стейт с новым start_equity.
    """
    today = today_local(tz_name)
    if state is None or state.day != today.isoformat():
        state = DailyState.new(today=today, start_equity=current_equity)
        save_state(state)
        return state

    # если стартовый equity подозрительно <=0 и есть оценка текущего — подстрахуемся
    if state.start_equity <= 0 and current_equity > 0:
        state.start_equity = safe_float(current_equity, state.start_equity)
        save_state(state)
    return state

def can_open_more_trades(state: DailyState, max_trades_per_day: int) -> bool:
    """
    True, если дневной лимит сделок не превышен.
    """
    limit = max(0, int(max_trades_per_day))
    count = max(0, int(state.trades_count))
    return count < limit

def register_new_trade(state: DailyState) -> None:
    """
    Инкремент счётчика сделок за день.
    """
    state.trades_count = max(0, int(state.trades_count)) + 1
    save_state(state)

def register_realized_pnl(state: DailyState, pnl: float) -> None:
    """
    Добавляет реализованный PnL к дневной сумме (может быть отрицательным).
    """
    state.realized_pnl = safe_float(state.realized_pnl, 0.0) + safe_float(pnl, 0.0)
    save_state(state)

def _compute_drawdown(start_equity: float, current_equity: float) -> float:
    """
    Возвращает относительную просадку (0..+inf) как долю от стартового equity.
    """
    s = safe_float(start_equity, 0.0)
    c = safe_float(current_equity, 0.0)
    if s <= 0:
        return 0.0
    dd = (s - c) / s
    # нормируем: отрицательную «просадку» (рост) в 0
    return max(0.0, dd)

def daily_loss_hit(state: DailyState, daily_max_loss_pct: float, current_equity: float) -> bool:
    """
    True, если фактическая просадка от стартового equity дня превысила лимит.
    """
    limit = max(0.0, float(daily_max_loss_pct))
    dd = _compute_drawdown(state.start_equity, current_equity)
    return dd >= limit

def start_of_day_equity(state: DailyState) -> float:
    return safe_float(state.start_equity, 0.0)
