# services/reconcile.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from threading import Lock
from typing import Dict, Any, List, Optional, Literal, Iterable, Tuple

from db import crud_orders
from db.session import get_session

from executors.api_binance import BinanceExecutor

try:  # optional simulated executor
    from executors.simulated import SimulatedExecutor  # type: ignore
    _HAS_SIM = True
except Exception:  # pragma: no cover
    _HAS_SIM = False

from utils.structured_logging import get_logger


LOG = get_logger("ai_trader.reconcile")

__all__ = [
    "reconcile_positions",
    "reconcile_on_start",
    "reconcile_periodic",
    "get_periodic_config_from_env",
]


@dataclass(slots=True)
class WormJournalEntry:
    sequence: int
    timestamp: int
    prev_checksum: str
    payload: Dict[str, Any]
    checksum: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "prev_checksum": self.prev_checksum,
            "payload": dict(self.payload),
            "checksum": self.checksum,
        }


class WormJournal:
    """Append-only trade ledger with chained checksums."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    @property
    def path(self) -> Path:
        return self._path

    def _read_last(self) -> Optional[Dict[str, Any]]:
        if not self._path.exists():
            return None
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                last_line = ""
                for line in handle:
                    line = line.strip()
                    if line:
                        last_line = line
                if not last_line:
                    return None
                return json.loads(last_line)
        except Exception as exc:  # pragma: no cover - diagnostic only
            LOG.warning("Failed to read WORM journal tail: %s", exc)
            return None

    def _make_entry(self, payload: Dict[str, Any]) -> WormJournalEntry:
        last = self._read_last()
        prev_checksum = str(last.get("checksum")) if isinstance(last, dict) else "0" * 64
        sequence = int(last.get("sequence", 0)) + 1 if isinstance(last, dict) else 1
        base = {
            "sequence": sequence,
            "timestamp": int(time.time() * 1000),
            "prev_checksum": prev_checksum,
            "payload": dict(payload),
        }
        checksum = sha256(json.dumps(base, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        return WormJournalEntry(sequence=sequence, timestamp=base["timestamp"], prev_checksum=prev_checksum, payload=payload, checksum=checksum)

    def append(self, payload: Dict[str, Any]) -> WormJournalEntry:
        with self._lock:
            entry = self._make_entry(payload)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry.to_dict(), sort_keys=True) + "\n")
            return entry


def _worm_journal_path() -> Path:
    override = os.getenv("WORM_JOURNAL_PATH")
    if override:
        return Path(override).expanduser()
    root = Path(__file__).resolve().parents[1]
    return root / "state" / "worm" / "trade_journal.log"


_WORM_JOURNAL: Optional[WormJournal] = None


def _get_worm_journal() -> WormJournal:
    global _WORM_JOURNAL
    if _WORM_JOURNAL is None:
        _WORM_JOURNAL = WormJournal(_worm_journal_path())
    return _WORM_JOURNAL

# ──────────────────────────────────────────────────────────────────────────────
# Executor selection (robust to different Sim signatures)
# ──────────────────────────────────────────────────────────────────────────────

def _select_executor(mode: Literal["binance", "sim"], testnet: bool):
    """
    Единая точка выбора исполнителя.
    SimulatedExecutor используем, если доступен; иначе — BinanceExecutor в тестнете.
    """
    if mode == "sim":
        if _HAS_SIM:
            try:
                # Вариант 1: современный конструктор
                return SimulatedExecutor(testnet=True, config={})  # type: ignore
            except TypeError:
                # Вариант 2: обратная совместимость (как у BinanceExecutor)
                try:
                    return SimulatedExecutor(api_key="x", api_secret="y", testnet=True)  # type: ignore
                except Exception:
                    LOG.warning("Simulated executor ctor failed, fallback to BinanceExecutor testnet")
        else:
            LOG.warning("Simulated executor not found, falling back to BinanceExecutor (testnet=%s)", testnet)
        return BinanceExecutor(testnet=True, config={})
    return BinanceExecutor(testnet=testnet, config={})


# ──────────────────────────────────────────────────────────────────────────────
# Journal aggregation
# ──────────────────────────────────────────────────────────────────────────────

async def _fetch_last_orders(session, *, limit: int) -> List[Dict[str, Any]]:
    """
    Унифицированная выборка последних записей журнала ордеров.
    Поддерживает обе сигнатуры:
      - crud_orders.get_last_orders(session, limit=..): List[Dict]
      - crud_orders.last_orders(session, limit=..): List[OrderLog]
    Возвращает список словарей с ключами: symbol, side, qty.
    """
    # Вариант 1: уже есть удобная функция, отдающая словари
    get_last = getattr(crud_orders, "get_last_orders", None)
    if callable(get_last):
        rows = await get_last(session, limit=limit)
        out: List[Dict[str, Any]] = []
        for r in rows or []:
            try:
                out.append({
                    "symbol": (r.get("symbol") if isinstance(r, dict) else getattr(r, "symbol", None)),
                    "side": (r.get("side") if isinstance(r, dict) else getattr(r, "side", None)),
                    "qty": (r.get("qty") if isinstance(r, dict) else getattr(r, "qty", None)),
                })
            except Exception:
                continue
        return out

    # Вариант 2: старое API last_orders -> объекты ORM
    last_orders_fn = getattr(crud_orders, "last_orders", None)
    if callable(last_orders_fn):
        rows = await last_orders_fn(session, limit=limit)
        out = []
        for r in rows or []:
            try:
                out.append({
                    "symbol": getattr(r, "symbol", None),
                    "side": getattr(r, "side", None),
                    "qty": getattr(r, "qty", None),
                })
            except Exception:
                continue
        return out

    LOG.warning("No suitable 'get_last_orders/last_orders' in crud_orders; journal aggregation will be empty.")
    return []


async def _build_journal_net_positions(limit_orders: int = 200, symbols: Optional[Iterable[str]] = None) -> Dict[str, float]:
    """
    Нетто-позиции по символам из локального журнала (последние N записей):
      net = sum(+qty for buy) + sum(-qty for sell)
    Служебные записи (cancel/reconcile/None side) игнорируем.
    Если передан symbols — агрегируем только по ним.
    """
    filt: Optional[set[str]] = None
    if symbols:
        filt = {str(s).upper() for s in symbols if s}

    journal_map: Dict[str, float] = {}
    async with get_session() as session:
        last_orders = await _fetch_last_orders(session, limit=limit_orders)

    for o in last_orders:
        sym = str(o.get("symbol") or "").upper()
        if not sym:
            continue
        if filt and sym not in filt:
            continue
        qty = o.get("qty")
        side = (o.get("side") or "").lower()
        if qty is None or side not in ("buy", "sell"):
            continue
        journal_map[sym] = journal_map.get(sym, 0.0) + (float(qty) if side == "buy" else -float(qty))

    return journal_map


# ──────────────────────────────────────────────────────────────────────────────
# Tolerances & helpers
# ──────────────────────────────────────────────────────────────────────────────

async def _symbol_tolerance(executor, symbol: str, default_abs_tol: float) -> float:
    """
    Эвристически оцениваем шаг лота через round_qty, чтобы вычислить разумный допуск.
    Если не получилось — возвращаем default_abs_tol.
    """
    try:
        probe = 1e-10  # достаточно маленький объём
        round_qty = getattr(executor, "round_qty", None)
        if callable(round_qty):
            q1 = await round_qty(symbol, probe)
            q2 = await round_qty(symbol, 2.0 * probe)
            step_est = abs(float(q2) - float(q1))
            if step_est > 0:
                # половина шага — консервативно, но не меньше базового допуска
                return max(default_abs_tol, step_est * 0.5)
    except Exception:
        pass
    return default_abs_tol


def _diff(a: float, b: float) -> float:
    return float(a) - float(b)


def _comp_side_for_delta(delta: float) -> Optional[str]:
    """
    Возвращает сторону служебной записи, чтобы скомпенсировать дельту журнала до факта:
      delta = actual - journal.
      >0 → надо ДОБАВИТЬ к журналу покупку (buy),
      <0 → надо ДОБАВИТЬ к журналу продажу (sell),
       0 → ничего.
    """
    if delta > 0:
        return "buy"
    if delta < 0:
        return "sell"
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Reconcile core
# ──────────────────────────────────────────────────────────────────────────────

async def _write_reconcile_entry(
    session,
    *,
    mode: str,
    testnet: bool,
    symbol: str,
    side: Optional[str],
    qty: Optional[float],
    reason: str,
    extra_raw: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Пишет служебную запись RECONCILE в журнал.
    """
    raw = {"ts_ms": int(time.time() * 1000), **(extra_raw or {})}
    await crud_orders.log_order(
        session=session,
        exchange=mode,
        testnet=testnet,
        symbol=symbol,
        side=side,
        type=None,
        qty=qty,
        price=None,
        status="RECONCILE",
        order_id=None,
        client_order_id=None,
        reason=reason,
        sl_pct=None,
        tp_pct=None,
        raw=raw,
    )


async def reconcile_positions(
    *,
    mode: Literal["binance", "sim"] = "binance",
    testnet: bool = True,
    auto_fix: bool = False,
    abs_tol: float = 1e-8,
    journal_limit: int = 200,
    symbols: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Сверка фактических позиций и локального журнала.

    Ключевой сценарий: «позицию закрыли вручную на бирже» →
    фактическая позиция 0, в журнале остаток ≠ 0 → добавляем служебную запись,
    чтобы нетто-журнал совпал с фактом.

    Args:
        mode: "binance" или "sim".
        testnet: True для тестнета.
        auto_fix: если True — вносит компенсационные служебные записи в журнал.
        abs_tol: базовый абсолютный допуск по qty (если нельзя оценить шаг).
        journal_limit: сколько последних записей журнала агрегировать.
        symbols: список тикеров для частичной сверки (None = все, что найдены).

    Returns:
        dict: подробный отчёт со списком расхождений и предпринятыми действиями.
    """
    ex = _select_executor(mode, testnet)

    try:
        # 1) Фактический баланс/позиции на исполнителе
        actual_balance = await ex.fetch_balance()
        # get_positions поддерживается всеми нашими исполнителями
        actual_positions = await ex.get_positions(symbols=[s for s in symbols] if symbols else None)  # type: ignore[arg-type]
        actual_map: Dict[str, float] = {
            (p.get("symbol") or "").upper(): float(p.get("qty") or 0.0) for p in (actual_positions or [])
        }

        # 2) Нетто-позиции из локального журнала
        journal_map = await _build_journal_net_positions(limit_orders=journal_limit, symbols=symbols)

        # 3) Полный список символов к сверке
        symbols_all = sorted(set(actual_map.keys()) | set(journal_map.keys()))

        mismatches: List[Dict[str, Any]] = []
        per_symbol: List[Dict[str, Any]] = []
        planned_actions: List[Dict[str, Any]] = []
        performed_actions: List[Dict[str, Any]] = []
        worm_detected: List[Dict[str, Any]] = []
        worm_adjustments: List[Dict[str, Any]] = []
        try:
            worm_journal = _get_worm_journal()
        except Exception as exc:  # pragma: no cover - defensive guard
            LOG.error("Failed to initialise WORM journal: %s", exc)
            worm_journal = None

        for sym in symbols_all:
            a = float(actual_map.get(sym, 0.0))
            j = float(journal_map.get(sym, 0.0))

            tol = await _symbol_tolerance(ex, sym, abs_tol)
            delta = _diff(a, j)  # насколько нужно подвинуть журнал, чтобы сравняться с фактом
            match = abs(delta) <= tol

            item = {
                "symbol": sym,
                "actual_qty": a,
                "journal_qty": j,
                "delta": delta,
                "tolerance": tol,
                "match": match,
                "only_in_actual": sym in actual_map and sym not in journal_map,
                "only_in_journal": sym in journal_map and sym not in actual_map,
            }
            per_symbol.append(item)
            if not match:
                mismatches.append(item)

                # план действия (компенсация дельты в журнале)
                side = _comp_side_for_delta(delta)
                action = {
                    "symbol": sym,
                    "side": side,
                    "qty": abs(delta) if side else 0.0,
                    "reason": "reconcile_compensate",
                    "note": "compensate journal to match actual",
                }
                planned_actions.append(action)

                if worm_journal is not None:
                    try:
                        entry = worm_journal.append(
                            {
                                "event": "reconcile_mismatch",
                                "symbol": sym,
                                "actual_qty": a,
                                "journal_qty": j,
                                "delta": delta,
                                "tolerance": tol,
                                "auto_fix_enabled": bool(auto_fix),
                            }
                        )
                        worm_detected.append(entry.to_dict())
                    except Exception as exc:  # pragma: no cover - diagnostics only
                        LOG.error("WORM journal append failed (mismatch %s): %s", sym, exc)

        # 4) Авто-фиксация журнала (без торгов на бирже!)
        if mismatches and auto_fix:
            async with get_session() as session:
                for act in planned_actions:
                    if not act["side"] or act["qty"] <= 0:
                        # нечего фиксировать (теоретически не должно случиться)
                        continue
                    await _write_reconcile_entry(
                        session,
                        mode=mode,
                        testnet=testnet,
                        symbol=act["symbol"],
                        side=act["side"],
                        qty=float(act["qty"]),
                        reason=act["reason"],
                        extra_raw={"note": act["note"]},
                    )
                    performed_actions.append(act)
                    if worm_journal is not None:
                        try:
                            entry = worm_journal.append(
                                {
                                    "event": "auto_adjustment",
                                    "symbol": act["symbol"],
                                    "side": act["side"],
                                    "qty": float(act["qty"]),
                                    "reason": act["reason"],
                                    "note": act["note"],
                                }
                            )
                            worm_adjustments.append(entry.to_dict())
                        except Exception as exc:  # pragma: no cover - diagnostics only
                            LOG.error("WORM journal append failed (adjustment %s): %s", act["symbol"], exc)
            LOG.info("Reconcile auto-fix applied (%d records)", len(performed_actions))
        elif mismatches:
            LOG.warning("RECONCILE: %d mismatches found (auto_fix=%s)", len(mismatches), auto_fix)

        # 5) Итоговый отчёт
        report: Dict[str, Any] = {
            "ok": True,
            "summary": {
                "symbols_checked": len(symbols_all),
                "mismatch_count": len(mismatches),
                "auto_fix": bool(auto_fix),
                "actions_planned": len(planned_actions),
                "actions_performed": len(performed_actions),
            },
            "mismatches": mismatches,
            "per_symbol": per_symbol,
            "actions": {
                "planned": planned_actions,
                "performed": performed_actions,
            },
            "positions": actual_positions,
            "balance": actual_balance,
            "worm_journal": {
                "path": str(worm_journal.path) if worm_journal is not None else None,
                "mismatches": worm_detected,
                "adjustments": worm_adjustments,
            },
            "meta": {
                "mode": mode,
                "testnet": testnet,
                "journal_limit": journal_limit,
                "abs_tol_base": abs_tol,
                "symbols_filter": sorted({s for s in symbols}) if symbols else None,
            },
        }
        return report

    except Exception as e:
        LOG.exception("Reconcile failed: %r", e)
        return {"ok": False, "error": str(e)}

    finally:
        try:
            await ex.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# Boot-time hook via ENV
# ──────────────────────────────────────────────────────────────────────────────

def _env_true(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on"}

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


async def reconcile_on_start() -> Optional[Dict[str, Any]]:
    """
    Если выставлены флаги ENV, выполняет сверку при старте приложения.

    ENV:
      • RECONCILE_ON_START=true|1        — включить сверку при старте
      • RECONCILE_AUTOFIX=true|1         — вносить компенсационные записи
      • RECONCILE_MODE=binance|sim       — выбрать исполнителя
      • RECONCILE_TESTNET=true|1         — Binance testnet
      • RECONCILE_JOURNAL_LIMIT=500      — глубина журнала
      • RECONCILE_ABS_TOL=1e-8           — базовый допуск
      • RECONCILE_SYMBOLS=BTCUSDT,ETHUSDT — фильтр символов (необязательно)
    """
    if not _env_true("RECONCILE_ON_START", False):
        return None

    mode_env = os.getenv("RECONCILE_MODE", "binance").strip().lower()
    mode: Literal["binance", "sim"] = "sim" if mode_env == "sim" else "binance"
    testnet = _env_true("RECONCILE_TESTNET", True)
    auto_fix = _env_true("RECONCILE_AUTOFIX", False)
    journal_limit = _env_int("RECONCILE_JOURNAL_LIMIT", 500)
    abs_tol = _env_float("RECONCILE_ABS_TOL", 1e-8)
    symbols_env = os.getenv("RECONCILE_SYMBOLS")
    symbols = [s.strip().upper() for s in symbols_env.split(",")] if symbols_env else None

    LOG.info(
        "Reconcile on start: mode=%s testnet=%s auto_fix=%s limit=%d abs_tol=%g symbols=%s",
        mode, testnet, auto_fix, journal_limit, abs_tol, symbols
    )
    return await reconcile_positions(
        mode=mode,
        testnet=testnet,
        auto_fix=auto_fix,
        abs_tol=abs_tol,
        journal_limit=journal_limit,
        symbols=symbols,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Periodic reconcile loop (optional)
# ──────────────────────────────────────────────────────────────────────────────

async def reconcile_periodic(
    *,
    interval_sec: int = 60,
    mode: Literal["binance", "sim"] = "binance",
    testnet: bool = True,
    auto_fix: bool = False,
    abs_tol: float = 1e-8,
    journal_limit: int = 200,
    symbols: Optional[Iterable[str]] = None,
) -> None:
    """
    Простой фоновой цикл сверки. Вызывать из startup-сценария:
        asyncio.create_task(reconcile_periodic(interval_sec=300, auto_fix=True))

    Останавливается только по отмене задачи.
    """
    import asyncio

    LOG.info("Reconcile loop started: every %ds (mode=%s, testnet=%s, auto_fix=%s)",
             interval_sec, mode, testnet, auto_fix)
    try:
        while True:
            try:
                report = await reconcile_positions(
                    mode=mode,
                    testnet=testnet,
                    auto_fix=auto_fix,
                    abs_tol=abs_tol,
                    journal_limit=journal_limit,
                    symbols=symbols,
                )
                if not report.get("ok", False):
                    LOG.warning("Reconcile iteration returned error: %s", report.get("error"))
                else:
                    LOG.debug("Reconcile iteration ok: mismatches=%d",
                              int(report.get("summary", {}).get("mismatch_count", 0)))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                LOG.exception("Reconcile loop iteration failed: %r", e)

            await asyncio.sleep(max(1, int(interval_sec)))
    except asyncio.CancelledError:
        LOG.info("Reconcile loop cancelled; exiting")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers to configure periodic loop from ENV (opt-in)
# ──────────────────────────────────────────────────────────────────────────────

def get_periodic_config_from_env() -> Optional[Dict[str, Any]]:
    """
    Читает ENV и, если включено, возвращает kwargs для reconcile_periodic().
    Иначе — None.

    ENV:
      • RECONCILE_PERIODIC=true|1
      • RECONCILE_INTERVAL_SEC=60
      • (прочие переменные — см. reconcile_on_start)
    """
    if not _env_true("RECONCILE_PERIODIC", False):
        return None

    mode_env = os.getenv("RECONCILE_MODE", "binance").strip().lower()
    mode: Literal["binance", "sim"] = "sim" if mode_env == "sim" else "binance"
    testnet = _env_true("RECONCILE_TESTNET", True)
    auto_fix = _env_true("RECONCILE_AUTOFIX", False)
    journal_limit = _env_int("RECONCILE_JOURNAL_LIMIT", 200)
    abs_tol = _env_float("RECONCILE_ABS_TOL", 1e-8)
    interval_sec = _env_int("RECONCILE_INTERVAL_SEC", 60)
    symbols_env = os.getenv("RECONCILE_SYMBOLS")
    symbols = [s.strip().upper() for s in symbols_env.split(",")] if symbols_env else None

    return {
        "interval_sec": interval_sec,
        "mode": mode,
        "testnet": testnet,
        "auto_fix": auto_fix,
        "abs_tol": abs_tol,
        "journal_limit": journal_limit,
        "symbols": symbols,
    }

