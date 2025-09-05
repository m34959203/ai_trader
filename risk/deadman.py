# risk/deadman.py
from __future__ import annotations

import os
import time
import asyncio
import logging
from pathlib import Path
from typing import Protocol, Optional, Any, Callable

from utils.risk_config import load_risk_config

log = logging.getLogger("ai_trader.deadman")

__all__ = [
    "HEARTBEAT_FILE",
    "touch_heartbeat",
    "last_heartbeat",
    "is_stale",
    "check_deadman",
]

# Храним heartbeat в data/state/heartbeat.txt
HEARTBEAT_FILE = Path(__file__).resolve().parents[1] / "data" / "state" / "heartbeat.txt"
HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Тип исполнителя (поддержим sync/async реализацию close_all_positions)
# ──────────────────────────────────────────────────────────────────────────────
class IClosable(Protocol):
    def close_all_positions(self, reason: str) -> Any: ...  # может быть sync или async


# ──────────────────────────────────────────────────────────────────────────────
# Файловые утилиты (атомарная запись)
# ──────────────────────────────────────────────────────────────────────────────
def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ──────────────────────────────────────────────────────────────────────────────
# Публичные функции
# ──────────────────────────────────────────────────────────────────────────────
def touch_heartbeat(ts: Optional[int] = None) -> None:
    """
    Обновляет метку heartbeat (в секундах). Вызывай каждые 5–15 секунд.
    """
    if ts is None:
        ts = int(time.time())
    try:
        _atomic_write_text(HEARTBEAT_FILE, str(int(ts)))
    except Exception as e:  # pragma: no cover
        log.warning("heartbeat write failed: %r", e)


def last_heartbeat() -> int:
    """
    Возвращает время последнего heartbeat (секунды с эпохи) или 0, если нет файла/ошибка.
    """
    try:
        txt = HEARTBEAT_FILE.read_text(encoding="utf-8").strip()
        return int(txt or "0")
    except Exception:
        return 0


def is_stale(max_stale_sec: Optional[int] = None) -> bool:
    """
    True, если heartbeat устарел дольше max_stale_sec. Если max_stale_sec не задан,
    используем DEADMAN_MAX_STALE_SEC из конфигурации.
    """
    cfg = load_risk_config()
    limit = int(max_stale_sec if max_stale_sec is not None else cfg.deadman_max_stale_sec)
    last = last_heartbeat()
    now = int(time.time())
    return last == 0 or (now - last) > max(1, limit)


def _call_close_all_sync_or_async(executor: IClosable, *, reason: str) -> None:
    """
    Универсальный вызов close_all_positions для sync/async исполнителей.
    Если в текущем потоке уже есть event loop — создаём задачу, иначе запускаем временный цикл.
    """
    close_fn: Optional[Callable[..., Any]] = getattr(executor, "close_all_positions", None)  # type: ignore[attr-defined]
    if close_fn is None:
        log.warning("executor has no close_all_positions(); skipping emergency close")
        return

    try:
        if asyncio.iscoroutinefunction(close_fn):  # type: ignore[arg-type]
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(close_fn(reason=reason))  # type: ignore[misc]
            else:
                loop.create_task(close_fn(reason=reason))  # type: ignore[misc]
        else:
            close_fn(reason=reason)  # type: ignore[misc]
    except Exception as e:  # pragma: no cover
        log.warning("close_all_positions call failed: %r", e)


def check_deadman(executor: Optional[IClosable] = None, *, max_stale_sec: Optional[int] = None) -> bool:
    """
    Проверка dead-man:
      • если пульс устарел дольше порога — логируем ошибку,
        пробуем вызвать executor.close_all_positions(reason="deadman_switch"),
        затем обновляем heartbeat текущим временем.
      • возвращает True, если сработал dead-man, иначе False.

    Рекомендуемый вызов: каждые 10–30 сек, в том же цикле, где и touch_heartbeat().
    """
    cfg = load_risk_config()
    limit = int(max_stale_sec if max_stale_sec is not None else cfg.deadman_max_stale_sec)

    last = last_heartbeat()
    now = int(time.time())
    stale_for = (now - last) if last else None

    if last == 0 or (now - last) > max(1, limit):
        log.error(
            "Dead-man switch: heartbeat stale for %s s (limit=%s). Initiating emergency close.",
            "unknown" if stale_for is None else stale_for, limit
        )
        if executor is not None:
            _call_close_all_sync_or_async(executor, reason="deadman_switch")
        # чтобы не триггерить повторно — сбросим «пульс» на сейчас
        try:
            _atomic_write_text(HEARTBEAT_FILE, str(now))
        except Exception as e:  # pragma: no cover
            log.warning("deadman: failed to refresh heartbeat after trigger: %r", e)
        return True

    return False
