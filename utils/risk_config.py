from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Python 3.9+: zoneinfo для локального времени дневного стопа и учёта
try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# helpers: чтение ENV/файлов и клампинг значений
# ──────────────────────────────────────────────────────────────────────────────
def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _as_float(name: str, default: float, lo: float = 0.0, hi: float = 1.0) -> float:
    v = os.getenv(name)
    if v is None:
        return _clamp(float(default), lo, hi)
    try:
        return _clamp(float(v), lo, hi)
    except Exception:
        return _clamp(float(default), lo, hi)

def _as_int(name: str, default: int, lo: int = 0, hi: int = 10_000) -> int:
    v = os.getenv(name)
    if v is None:
        return max(lo, min(hi, int(default)))
    try:
        return max(lo, min(hi, int(v)))
    except Exception:
        return max(lo, min(hi, int(default)))

def _load_from_file(path: Path) -> Dict[str, Any]:
    """Читает exec.yaml/exec.json, возвращает dict (может быть вложенным)."""
    if not path.exists():
        return {}
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return {}
        if path.suffix in (".yaml", ".yml"):
            import yaml  # pyyaml
            return yaml.safe_load(text) or {}
        if path.suffix == ".json":
            return json.loads(text) or {}
    except Exception:
        return {}
    return {}

def _find_config_file() -> Optional[Path]:
    base = Path(__file__).resolve().parents[1] / "configs"  # ai_trader/configs
    for name in ("exec.yaml", "exec.yml", "exec.json"):
        f = base / name
        if f.exists():
            return f
    return None

def _get_nested(cfg: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Безопасно вытаскивает cfg[a][b]..., если есть; иначе default."""
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


# ──────────────────────────────────────────────────────────────────────────────
# Dataclass конфигурации рисков
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RiskConfig:
    """
    Конфигурация риск-лимитов.
    Значения храним в долях (0.02 = 2%). Валидируем и клампим при загрузке.
    """
    # Пер-сделка и портфель
    risk_pct_per_trade: float = 0.01         # ≤1% риска на сделку (по умолчанию)
    portfolio_max_risk_pct: float = 0.06     # ≤6% совокупно по открытым позициям
    max_open_positions: int = 5              # ограничение на кол-во одновременных позиций

    # Дневные лимиты/паузы
    daily_max_loss_pct: float = 0.02         # ≤2% потеря за день → автопауза
    day_reset_hour_local: int = 0            # во сколько «переводить день» для дневных метрик (час локальной TZ)

    # Лимиты на активность
    max_trades_per_day: int = 15             # лимит сделок/день (доп. защита)

    # Dead-man / heartbeat
    deadman_max_stale_sec: int = 90          # если нет тиков/ивентов дольше — стоп/пауза

    # Трейлинг и стопы
    enable_trailing: bool = True
    atr_mult_trailing: float = 2.0           # множитель ATR для трейлинга
    min_sl_distance_pct: float = 0.002       # минимальная дистанция SL (0.2%) на случай тонкого рынка

    # Часовой пояс для дневного учёта
    tz_name: str = "Asia/Almaty"

    # ── утиль ──────────────────────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def tz(self):
        if ZoneInfo is None:
            return None
        try:
            return ZoneInfo(self.tz_name)
        except Exception:
            return None

    def validate(self) -> "RiskConfig":
        """
        Возвращает скорректированную (клампинг) копию, если параметры вне диапазона.
        """
        return replace(
            self,
            risk_pct_per_trade=_clamp(self.risk_pct_per_trade, 0.0, 0.2),
            portfolio_max_risk_pct=_clamp(self.portfolio_max_risk_pct, 0.0, 1.0),
            daily_max_loss_pct=_clamp(self.daily_max_loss_pct, 0.0, 0.5),
            atr_mult_trailing=_clamp(self.atr_mult_trailing, 0.1, 20.0),
            min_sl_distance_pct=_clamp(self.min_sl_distance_pct, 0.0, 0.2),
            max_open_positions=max(0, min(1000, int(self.max_open_positions))),
            max_trades_per_day=max(0, min(10000, int(self.max_trades_per_day))),
            deadman_max_stale_sec=max(5, min(3600, int(self.deadman_max_stale_sec))),
            day_reset_hour_local=max(0, min(23, int(self.day_reset_hour_local))),
        )

    # Удобный хелпер для вычисления риска в долях на конкретную сделку
    def risk_fraction_for_trade(self, stop_distance_pct: float) -> float:
        """
        Возвращает долю капитала, которую можно задействовать под риск этой сделки,
        исходя из risk_pct_per_trade и дистанции до стопа.
        Например: при 1% риска и стопе 0.5% → позицию можно построить на ~2R.
        """
        stop_pct = max(self.min_sl_distance_pct, float(stop_distance_pct))
        if stop_pct <= 0:
            return 0.0
        # сколько «единиц риска» помещается в заданный стоп
        return _clamp(self.risk_pct_per_trade / stop_pct, 0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Загрузка: ENV → файл (ENV имеет приоритет). Поддержка вложенных ключей.
# ──────────────────────────────────────────────────────────────────────────────
def load_risk_config() -> RiskConfig:
    """
    Загружает RiskConfig из ENV и (опционально) configs/exec.yaml|json.
    Поддерживаются два стиля ключей в файле:
      1) Плоский: risk_pct_per_trade: 0.01
      2) Вложенный: risk: { per_trade_max: 0.01, portfolio_max: 0.06, ... }

    ENV имеет приоритет. Переменные окружения:
      RISK_PCT_PER_TRADE
      PORTFOLIO_MAX_RISK_PCT
      MAX_OPEN_POSITIONS
      DAILY_MAX_LOSS_PCT
      DAY_RESET_HOUR_LOCAL
      MAX_TRADES_PER_DAY
      DEADMAN_MAX_STALE_SEC
      ENABLE_TRAILING
      ATR_MULT_TRAILING
      MIN_SL_DISTANCE_PCT
      TZ_NAME
    """
    # 1) читаем файл (если есть)
    file_path = _find_config_file()
    file_cfg = _load_from_file(file_path) if file_path else {}

    # 2) плоские значения (с бэкапом из вложенных)
    flat = {
        "risk_pct_per_trade": _get_nested(file_cfg, "risk_pct_per_trade", default=None)
                               or _get_nested(file_cfg, "risk", "per_trade_max", default=0.01),
        "portfolio_max_risk_pct": _get_nested(file_cfg, "portfolio_max_risk_pct", default=None)
                               or _get_nested(file_cfg, "risk", "portfolio_max", default=0.06),
        "max_open_positions": _get_nested(file_cfg, "max_open_positions", default=None)
                               or _get_nested(file_cfg, "risk", "max_positions", default=5),
        "daily_max_loss_pct": _get_nested(file_cfg, "daily_max_loss_pct", default=None)
                               or _get_nested(file_cfg, "risk", "daily_stop", default=0.02),
        "day_reset_hour_local": _get_nested(file_cfg, "day_reset_hour_local", default=None)
                               or _get_nested(file_cfg, "risk", "day_reset_hour_local", default=0),
        "max_trades_per_day": _get_nested(file_cfg, "max_trades_per_day", default=None)
                               or _get_nested(file_cfg, "risk", "max_trades_per_day", default=15),
        "deadman_max_stale_sec": _get_nested(file_cfg, "deadman_max_stale_sec", default=None)
                               or _get_nested(file_cfg, "risk", "deadman_max_stale_sec", default=90),
        "enable_trailing": _get_nested(file_cfg, "enable_trailing", default=None)
                               or _get_nested(file_cfg, "risk", "enable_trailing", default=True),
        "atr_mult_trailing": _get_nested(file_cfg, "atr_mult_trailing", default=None)
                               or _get_nested(file_cfg, "risk", "atr_mult_trailing", default=2.0),
        "min_sl_distance_pct": _get_nested(file_cfg, "min_sl_distance_pct", default=None)
                               or _get_nested(file_cfg, "risk", "min_sl_distance_pct", default=0.002),
        "tz_name": _get_nested(file_cfg, "tz_name", default=None)
                               or _get_nested(file_cfg, "risk", "tz_name", default="Asia/Almaty"),
    }

    # 3) ENV override (с клампингом)
    rc = RiskConfig(
        risk_pct_per_trade=_as_float("RISK_PCT_PER_TRADE", float(flat["risk_pct_per_trade"]), 0.0, 0.2),
        portfolio_max_risk_pct=_as_float("PORTFOLIO_MAX_RISK_PCT", float(flat["portfolio_max_risk_pct"]), 0.0, 1.0),
        max_open_positions=_as_int("MAX_OPEN_POSITIONS", int(flat["max_open_positions"]), 0, 1000),
        daily_max_loss_pct=_as_float("DAILY_MAX_LOSS_PCT", float(flat["daily_max_loss_pct"]), 0.0, 0.5),
        day_reset_hour_local=_as_int("DAY_RESET_HOUR_LOCAL", int(flat["day_reset_hour_local"]), 0, 23),
        max_trades_per_day=_as_int("MAX_TRADES_PER_DAY", int(flat["max_trades_per_day"]), 0, 10_000),
        deadman_max_stale_sec=_as_int("DEADMAN_MAX_STALE_SEC", int(flat["deadman_max_stale_sec"]), 5, 3600),
        enable_trailing=str(os.getenv("ENABLE_TRAILING", str(flat["enable_trailing"]))).strip().lower() in {"1","true","yes","on"},
        atr_mult_trailing=_as_float("ATR_MULT_TRAILING", float(flat["atr_mult_trailing"]), 0.1, 20.0),
        min_sl_distance_pct=_as_float("MIN_SL_DISTANCE_PCT", float(flat["min_sl_distance_pct"]), 0.0, 0.2),
        tz_name=os.getenv("TZ_NAME", str(flat["tz_name"])),
    )

    # 4) Возврат валидированной копии
    return rc.validate()
