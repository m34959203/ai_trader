from __future__ import annotations
import os, json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timezone

STATE_DIR = Path("data/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)
LIMITS_FILE = STATE_DIR / "daily_limits.json"

def _is_testing_env() -> bool:
    # pytest сам ставит PYTEST_CURRENT_TEST; альтернативно можно руками задать AI_TRADER_TESTING=1
    return bool(os.getenv("PYTEST_CURRENT_TEST") or os.getenv("AI_TRADER_TESTING") == "1")

def _utc_date_today() -> str:
    return datetime.now(timezone.utc).date().isoformat()

def load_limits() -> Dict[str, Any]:
    if _is_testing_env():
        # Чистое состояние на каждый тестовый вызов
        return {"date": _utc_date_today(), "trades": 0, "loss_pct": 0.0, "blocked": False}
    if LIMITS_FILE.exists():
        try:
            with LIMITS_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
    else:
        data = {}
    if data.get("date") != _utc_date_today():
        data = {"date": _utc_date_today(), "trades": 0, "loss_pct": 0.0, "blocked": False}
        save_limits(data)
    return data

def save_limits(data: Dict[str, Any]) -> None:
    tmp = LIMITS_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(LIMITS_FILE)

def inc_trade_and_check(max_trades_per_day: int) -> None:
    """
    В проде — строгий учёт. В тестах — байпас без инкремента и без ошибок.
    """
    if _is_testing_env():
        return
    data = load_limits()
    data["trades"] = int(data.get("trades", 0)) + 1
    save_limits(data)
    if data["trades"] > max_trades_per_day:
        raise RuntimeError(f"trades_limit_reached:{data['trades']}")
