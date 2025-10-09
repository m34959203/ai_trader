# ai_trader/services/trading_service.py
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Literal, Optional, Set, Tuple, TypeVar, cast

# Совместимые интерфейсы (ожидаемые типы/классы)
from executors import (  # type: ignore
    Executor,            # базовый интерфейс исполнителя
    BinanceExecutor,     # реальный спот через Binance REST (testnet/prod)
    SimulatedExecutor,   # симулятор
    UIExecutorAgent,     # UI-агент с DOM/OCR
    UIExecutorStub,      # совместимость
    OrderResult,         # тип результата ордера
    Position,            # тип позиции
)

# Риск-конфиг и утилиты портфельного риска
from monitoring.observability import OBSERVABILITY
from services.auto_heal import AutoHealingOrchestrator, StateSnapshot
from services.model_router import router_singleton
from services.security import AccessController, Role, SecretVault, TwoFactorGuard
from utils.assets import load_assets
from utils.logging_utils import install_sensitive_filter
from utils.risk_config import load_risk_config
from src.models.base import MarketFeatures
from src.utils_math import clamp01
try:
    # Необязательная зависимость (если добавляли ранее)
    from risk.risk_manager import (
        position_risk_fraction_from_params,
        portfolio_risk_used,
        can_open_new,
    )
except Exception:  # pragma: no cover
    # Локальные fallback-реализации (минимум для портфельной проверки)
    def position_risk_fraction_from_params(
        *, equity: float, qty: float, entry_price: float, stop_loss_price: Optional[float], min_sl_distance_pct: float = 0.0
    ) -> float:
        if equity <= 0 or qty <= 0 or stop_loss_price is None:
            return 1.0
        dist = abs(entry_price - float(stop_loss_price))
        min_dist_abs = float(min_sl_distance_pct) * float(entry_price)
        if dist <= 0 or dist < min_dist_abs:
            return 1.0
        return max(0.0, min(1.0, (dist * float(qty)) / float(equity)))

    def portfolio_risk_used(positions: Iterable[Dict[str, Any]], *, equity: float, min_sl_distance_pct: float) -> float:
        tot = 0.0
        for p in positions:
            try:
                tot += position_risk_fraction_from_params(
                    equity=equity,
                    qty=float(p.get("qty", 0.0)),
                    entry_price=float(p.get("entry_price", 0.0)),
                    stop_loss_price=p.get("stop_loss_price"),
                    min_sl_distance_pct=min_sl_distance_pct,
                )
            except Exception:
                tot += 1.0
        return max(0.0, min(1.0, tot))

    def can_open_new(
        *, new_position_risk_fraction: float, positions: Iterable[Dict[str, Any]], equity: float,
        portfolio_max_risk_pct: float, max_open_positions: int, min_sl_distance_pct: float
    ) -> bool:
        try:
            pos_list = list(positions)
        except TypeError:
            pos_list = [*positions]
        if len(pos_list) >= max_open_positions:
            return False
        used = portfolio_risk_used(pos_list, equity=equity, min_sl_distance_pct=min_sl_distance_pct)
        return (used + float(new_position_risk_fraction)) <= float(portfolio_max_risk_pct)

LOG = logging.getLogger("ai_trader.trading_service")

T = TypeVar("T")


# ──────────────────────────────────────────────────────────────────────────────
# Конфиг (локальный слой — оставлен для обратной совместимости UI/API)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskLimits:
    """
    Локальные риски для UI/старого кода (per-trade/daily).
    Портфельные лимиты берём из utils.risk_config (≤6%, max_open_positions, min SL distance).
    """
    per_trade_risk_pct: float = 0.01
    daily_loss_limit_pct: float = 0.02
    daily_max_trades: int = 50
    auto_adjust_amount: bool = True
    default_quote_ccy: str = "USDT"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "per_trade_risk_pct": self.per_trade_risk_pct,
            "daily_loss_limit_pct": self.daily_loss_limit_pct,
            "daily_max_trades": self.daily_max_trades,
            "auto_adjust_amount": self.auto_adjust_amount,
            "default_quote_ccy": self.default_quote_ccy,
        }


@dataclass(frozen=True)
class TradingConfig:
    """Обёртка конфигурации сервиса."""
    mode: Literal["sim", "binance", "ui"] = "sim"
    testnet: bool = True
    # подконфиги конкретных исполнителей
    binance: Optional[Dict[str, Any]] = None
    sim: Optional[Dict[str, Any]] = None
    ui: Optional[Dict[str, Any]] = None
    # риск
    risk: RiskLimits = RiskLimits()
    security: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_mapping(raw: Optional[Dict[str, Any]]) -> "TradingConfig":
        raw = raw or {}
        risk_raw = raw.get("risk") or {}
        risk = RiskLimits(
            per_trade_risk_pct=float(risk_raw.get("per_trade_risk_pct", 0.01)),
            daily_loss_limit_pct=float(risk_raw.get("daily_loss_limit_pct", 0.02)),
            daily_max_trades=int(risk_raw.get("daily_max_trades", 50)),
            auto_adjust_amount=bool(risk_raw.get("auto_adjust_amount", True)),
            default_quote_ccy=str(risk_raw.get("default_quote_ccy", "USDT")),
        )
        return TradingConfig(
            mode=raw.get("mode", "sim"),
            testnet=bool(raw.get("testnet", True)),
            binance=(raw.get("binance") or {}),
            sim=(raw.get("sim") or {}),
            ui=(raw.get("ui") or {}),
            risk=risk,
            security=dict(raw.get("security") or {}),
        )

    def to_mapping(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "testnet": self.testnet,
            "binance": dict(self.binance or {}),
            "sim": dict(self.sim or {}),
            "ui": dict(self.ui or {}),
            "risk": self.risk.to_dict(),
            "security": dict(self.security or {}),
        }


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


def evaluate_pre_trade_controls(
    *,
    symbol: str,
    side: Literal["buy", "sell"],
    quantity: float,
    price: float,
    decision: Dict[str, Any],
    features: Dict[str, Any],
    order_type: str = "market",
) -> Dict[str, Any]:
    """Run broker-level guardrails before an :class:`OrderRequest` is created.

    The evaluation covers three safety nets required by the production
    checklist:

    • Margin availability — ensure the notional (with buffer) fits into the
      deployable equity/margin bucket.
    • Broker stop constraints — block orders whose stop distance violates the
      configured minimum enforced by the broker.
    • Volume/volatility guardrails — avoid oversized participation when the
      venue liquidity or volatility is outside approved ranges.

    Returns a diagnostic payload containing violations (if any) together with
    contextual metrics so that callers can attach the information to audit
    logs or API responses.
    """

    limits = decision.get("limits", {}) if isinstance(decision, dict) else {}
    available_margin = float(
        limits.get("equity")
        or features.get("available_margin")
        or features.get("margin_available")
        or features.get("equity")
        or 0.0
    )
    required_margin = max(0.0, float(quantity) * float(price))

    margin_buffer_pct = _env_float("PRECHECK_MARGIN_BUFFER_PCT", 0.05)
    min_margin = required_margin * (1.0 + margin_buffer_pct)

    rc = load_risk_config()
    min_stop_pct = max(0.0, _env_float("BROKER_MIN_STOP_PCT", rc.min_sl_distance_pct))

    inferred_stop_pct_sources = [
        (decision.get("signal", {}) if isinstance(decision, dict) else {}).get("stop_loss_pct"),
        features.get("stop_loss_pct"),
        features.get("stop_distance_pct"),
    ]
    inferred_stop_pct = next((float(s) for s in inferred_stop_pct_sources if s), None)
    if (inferred_stop_pct is None or inferred_stop_pct <= 0) and price > 0:
        atr = float(features.get("atr", 0.0) or 0.0)
        inferred_stop_pct = atr / float(price) if atr > 0 else None
    if inferred_stop_pct is None or inferred_stop_pct <= 0:
        inferred_stop_pct = rc.min_sl_distance_pct

    max_order_equity_pct = _env_float("MAX_ORDER_EQUITY_PCT", 0.25)
    order_equity_ratio = (required_margin / available_margin) if available_margin > 0 else None

    avg_volume = float(
        features.get("avg_volume")
        or features.get("average_volume")
        or features.get("volume")
        or 0.0
    )
    max_volume_fraction = _env_float("MAX_ORDER_VOLUME_FRACTION", 0.1)
    max_volume_quantity: Optional[float]
    if avg_volume > 0:
        max_volume_quantity = avg_volume * max_volume_fraction
    else:
        max_volume_quantity = None

    volatility_measure = float(
        features.get("volatility")
        or features.get("vol")
        or limits.get("atr_pct")
        or 0.0
    )
    max_volatility = _env_float("MAX_ALLOWED_VOLATILITY", 0.2)

    violations: List[Dict[str, Any]] = []
    advisories: List[str] = []

    if available_margin <= 0:
        violations.append(
            {
                "code": "margin_unavailable",
                "message": "Available margin is unknown or zero; refuse to route order.",
                "details": {"available_margin": available_margin},
            }
        )
    elif available_margin < min_margin:
        violations.append(
            {
                "code": "margin_shortfall",
                "message": "Required margin exceeds allowed buffer.",
                "details": {
                    "available_margin": available_margin,
                    "required_margin": required_margin,
                    "buffer_pct": margin_buffer_pct,
                    "min_margin": min_margin,
                },
            }
        )

    if inferred_stop_pct < min_stop_pct:
        violations.append(
            {
                "code": "broker_stop_distance",
                "message": "Planned stop distance violates broker minimum.",
                "details": {
                    "stop_pct": inferred_stop_pct,
                    "min_stop_pct": min_stop_pct,
                    "order_type": order_type,
                },
            }
        )

    if order_equity_ratio is not None and order_equity_ratio > max_order_equity_pct:
        violations.append(
            {
                "code": "equity_participation_limit",
                "message": "Order notional breaches equity participation cap.",
                "details": {
                    "order_equity_ratio": order_equity_ratio,
                    "max_order_equity_pct": max_order_equity_pct,
                },
            }
        )

    if max_volume_quantity is None:
        advisories.append("volume_data_unavailable")
    elif quantity > max_volume_quantity:
        violations.append(
            {
                "code": "volume_limit",
                "message": "Requested quantity exceeds configured share of venue volume.",
                "details": {
                    "quantity": quantity,
                    "max_volume_quantity": max_volume_quantity,
                    "avg_volume": avg_volume,
                    "max_volume_fraction": max_volume_fraction,
                },
            }
        )

    if volatility_measure <= 0:
        advisories.append("volatility_data_unavailable")
    elif volatility_measure > max_volatility:
        violations.append(
            {
                "code": "volatility_limit",
                "message": "Market volatility above governance threshold.",
                "details": {
                    "volatility": volatility_measure,
                    "max_volatility": max_volatility,
                },
            }
        )

    return {
        "ok": not violations,
        "violations": violations,
        "advisories": advisories,
        "context": {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "required_margin": required_margin,
            "available_margin": available_margin,
            "margin_buffer_pct": margin_buffer_pct,
            "order_equity_ratio": order_equity_ratio,
            "max_order_equity_pct": max_order_equity_pct,
            "stop_distance_pct": inferred_stop_pct,
            "min_stop_pct": min_stop_pct,
            "avg_volume": avg_volume,
            "max_volume_fraction": max_volume_fraction,
            "volatility": volatility_measure,
            "max_volatility": max_volatility,
        },
    }


# ──────────────────────────────────────────────────────────────────────────────
# TradingService
# ──────────────────────────────────────────────────────────────────────────────

class TradingService:
    """
    Высокоуровневый сервис исполнения + риск-контроль.
    Теперь с портфельной проверкой риска перед каждым открытием:
      • вычисление риска новой позиции по SL-дистанции,
      • запрос открытых позиций из services.reconcile (или фоллбек к executor),
      • блокировка, если (суммарный риск + новый риск) > лимита портфеля
        или достигнут лимит числа позиций.
    """

    def __init__(
        self,
        *,
        mode: Literal["sim", "binance", "ui"] = "sim",
        testnet: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        install_sensitive_filter(LOG, fields=("api_key", "api_secret"))
        self._assets = load_assets()
        self._auto_healer = AutoHealingOrchestrator()
        self._auto_heal_tasks: Set[asyncio.Task[Any]] = set()
        self._auto_heal_pending_replay = False
        self._auto_healer.register_restore("trading_executor", self._restore_executor_from_snapshot)
        self._schedule_auto_heal_task(
            self._auto_healer.replay(name="trading_executor"),
            allow_defer=True,
        )
        self._ui_executor: Optional[UIExecutorAgent] = None
        self._failover_count = 0
        cfg = TradingConfig.from_mapping(config)
        # приоритет: явные аргументы > конфиг
        self._mode: Literal["sim", "binance", "ui"] = mode or cfg.mode or "sim"
        self._testnet: bool = testnet if testnet is not None else cfg.testnet
        self._raw_config: TradingConfig = TradingConfig(
            mode=self._mode,
            testnet=self._testnet,
            binance=cfg.binance,
            sim=cfg.sim,
            ui=cfg.ui,
            risk=cfg.risk,
            security=cfg.security,
        )
        self._vault = SecretVault()
        self._twofactor = TwoFactorGuard(self._vault)
        self._security_cfg: Dict[str, Any] = {}
        self._access = AccessController(roles={}, assignments={})
        self._require_2fa = False
        self._reconfigure_security()
        # ENV override (полезно в Docker/CI)
        if self._mode == "binance":
            self._testnet = _env_bool("BINANCE_TESTNET", self._testnet)

        self._executor: Executor = self._make_executor()
        if isinstance(self._executor, UIExecutorAgent):
            self._ui_executor = self._executor
        self._lock = asyncio.Lock()

        # Состояние риск-счётчиков
        self._day: date = datetime.now(timezone.utc).date()
        self._day_start_equity: Optional[float] = None
        self._day_trades_count: int = 0

    # ------------ lifecycle ------------
    async def close(self) -> None:
        """Освобождает ресурсы текущего исполнителя (HTTP-сессии и т.п.)."""
        for task in list(self._auto_heal_tasks):
            if not task.done():
                task.cancel()
        for task in list(self._auto_heal_tasks):
            try:
                await task
            except (asyncio.CancelledError, Exception):  # pragma: no cover - best effort cleanup
                pass
        self._auto_heal_tasks.clear()
        try:
            await self._executor.close()  # type: ignore[attr-defined]
        except Exception:
            pass
        if self._ui_executor is not None and self._ui_executor is not self._executor:
            try:
                await self._ui_executor.close()
            except Exception:
                pass

    async def __aenter__(self) -> "TradingService":
        if self._auto_heal_pending_replay:
            await self._auto_healer.replay(name="trading_executor")
            self._auto_heal_pending_replay = False
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ------------ internal ------------
    def _make_executor(self) -> Executor:
        if self._mode == "binance":
            return BinanceExecutor(testnet=self._testnet, config=self._raw_config.binance or {})  # type: ignore
        if self._mode == "ui":
            return UIExecutorAgent(testnet=self._testnet, config=self._raw_config.ui or {})  # type: ignore
        # по умолчанию — симулятор
        sim = SimulatedExecutor(testnet=self._testnet)
        return sim  # type: ignore

    def _get_ui_executor(self) -> UIExecutorAgent:
        if self._ui_executor is None:
            self._ui_executor = UIExecutorAgent(testnet=self._testnet, config=self._raw_config.ui or {})
        return self._ui_executor

    def _schedule_auto_heal_task(self, coro: Awaitable[Any], *, allow_defer: bool = False) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            if allow_defer:
                self._auto_heal_pending_replay = True
            return

        task: asyncio.Task[Any] = loop.create_task(coro)  # type: ignore[arg-type]

        def _on_done(done: asyncio.Task[Any]) -> None:
            self._auto_heal_tasks.discard(done)
            try:
                done.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:  # pragma: no cover - diagnostic logging
                LOG.debug("Auto-heal task finished with error: %r", exc)

        task.add_done_callback(_on_done)
        self._auto_heal_tasks.add(task)

    async def _restore_executor_from_snapshot(self, payload: Dict[str, Any]) -> None:
        try:
            await self._swap_executor_if_needed(
                mode=payload.get("mode"),
                testnet=payload.get("testnet"),
                config=payload.get("config"),
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            LOG.debug("Auto-heal restore failed: %r", exc)

    async def _with_failover(
        self,
        func: Callable[[Executor], Awaitable[T]],
        *,
        context: str,
        allow_ui: bool = True,
    ) -> T:
        try:
            return await func(self._executor)
        except Exception as exc:
            LOG.warning("Primary executor failure during %s: %r", context, exc)
            snapshot_payload = {
                "context": context,
                "mode": self._mode,
                "testnet": self._testnet,
                "config": self._raw_config.to_mapping(),
                "error": repr(exc),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "failovers": self._failover_count + 1,
            }
            await self._auto_healer.write_snapshot(
                StateSnapshot(name="trading_executor", payload=snapshot_payload)
            )
            self._schedule_auto_heal_task(self._auto_healer.trigger("trading_executor", snapshot_payload))
            if not allow_ui or self._mode == "ui":
                OBSERVABILITY.record_failover(context=context, recovered=False)
                raise
            self._failover_count += 1
            ui = self._get_ui_executor()
            recovered = False
            try:
                result = await func(ui)
                recovered = True
                return result
            except Exception as ui_exc:
                LOG.error("UI failover failed during %s: %r", context, ui_exc)
                raise
            finally:
                OBSERVABILITY.record_failover(context=context, recovered=recovered)

    async def _swap_executor_if_needed(
        self,
        *,
        mode: Optional[Literal["sim", "binance", "ui"]] = None,
        testnet: Optional[bool] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Горячее переключение режима без остановки сервиса."""
        new_mode = mode or self._mode
        new_testnet = self._testnet if testnet is None else bool(testnet)
        must_swap = (new_mode != self._mode) or (new_testnet != self._testnet) or (config is not None)
        if not must_swap:
            return

        async with self._lock:
            await self.close()
            self._mode = new_mode
            self._testnet = new_testnet
            if config is not None:
                self._raw_config = TradingConfig.from_mapping(config)
                self._reconfigure_security()
            self._executor = self._make_executor()
            if isinstance(self._executor, UIExecutorAgent):
                self._ui_executor = self._executor
            # сброс дневных счётчиков при смене режима
            self._reset_day_counters(force=True)

    # ------------ helpers ------------

    @property
    def risk(self) -> RiskLimits:
        return self._raw_config.risk

    @property
    def available_assets(self) -> Dict[str, List[str]]:
        return self._assets

    @property
    def failover_count(self) -> int:
        return self._failover_count

    @staticmethod
    def _symbols_arg(symbols: Optional[Iterable[str]]) -> Optional[List[str]]:
        if symbols is None:
            return None
        out = [str(s).upper() for s in symbols if s]
        return out or None

    def _reset_day_counters(self, *, force: bool = False) -> None:
        today = datetime.now(timezone.utc).date()
        if force or today != self._day:
            self._day = today
            self._day_trades_count = 0
            self._day_start_equity = None  # лениво вычислим при первом использовании
            LOG.info("Risk counters reset for new day: %s", self._day.isoformat())

    def _reconfigure_security(self) -> None:
        security = getattr(self._raw_config, "security", {}) or {}
        self._security_cfg = dict(security)
        roles_cfg = self._security_cfg.get("roles") or {}
        roles = {name: Role.from_permissions(name, perms) for name, perms in roles_cfg.items()}
        assignments_cfg = self._security_cfg.get("assignments") or {}
        assignments = {user: role for user, role in assignments_cfg.items() if role in roles}
        self._access = AccessController(roles=roles, assignments=assignments)
        self._require_2fa = bool(self._security_cfg.get("require_2fa", False))

    def _authorize(self, security_ctx: Optional[Dict[str, str]], permission: str) -> None:
        if not self._require_2fa and not self._access.roles:
            return
        ctx = security_ctx or {}
        user = ctx.get("user")
        otp = ctx.get("otp")
        if self._access.roles:
            if not user:
                raise PermissionError("User identifier required for RBAC checks")
            self._access.require(user, permission)
        if self._require_2fa:
            if not otp:
                raise PermissionError("Two-factor token required")
            subject = user or "default"
            self._twofactor.require(subject, otp)

    def enroll_twofactor(self, user_id: str) -> str:
        return self._twofactor.enroll(user_id)

    def assign_role(self, user_id: str, role: str) -> None:
        self._access.assign(user_id, role)

    async def _ensure_day_equity(self) -> None:
        """Лениво фиксируем equity на начало дня."""
        if self._day_start_equity is None:
            bal = await self._executor.fetch_balance()  # type: ignore[attr-defined]
            self._day_start_equity = self._estimate_equity(bal, self.risk.default_quote_ccy)
            LOG.info("Day start equity fixed: %.8f", self._day_start_equity)

    @staticmethod
    def _estimate_equity(balance: Dict[str, Any], preferred_quote: str = "USDT") -> float:
        """
        Пытаемся оценить общий equity:
          1) total (число) → берём
          2) total (dict)  → суммируем числа
          3) equity        → берём
          4) balances[]    → берём free+locked по preferred_quote
          5) иначе → 0.0
        """
        try:
            if isinstance(balance, dict):
                # 1/2/3
                if isinstance(balance.get("total"), (int, float)):
                    return float(balance["total"])
                if isinstance(balance.get("total"), dict):
                    vals = [float(v) for v in balance["total"].values() if isinstance(v, (int, float))]
                    if vals:
                        return float(sum(vals))
                if isinstance(balance.get("equity"), (int, float)):
                    return float(balance["equity"])
                # 4) спот BinanceExecutor — balances: [{asset, free, locked}]
                bals = balance.get("balances")
                if isinstance(bals, list):
                    pref = preferred_quote.upper()
                    for b in bals:
                        if str(b.get("asset")).upper() == pref:
                            free = float(b.get("free", 0.0) or 0.0)
                            locked = float(b.get("locked", 0.0) or 0.0)
                            return free + locked
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _quote_ccy_from_symbol(symbol: str, default_quote: str) -> str:
        try:
            if "/" in symbol:
                return symbol.split("/", 1)[1].upper()
        except Exception:
            pass
        return default_quote.upper()

    async def _get_est_price(self, symbol: str) -> Optional[float]:
        """Пытаемся взять текущую цену у исполнителя, если поддерживается."""
        try:
            getter = getattr(self._executor, "get_last_price", None)
            if callable(getter):
                p = await getter(symbol)
                return float(p) if p is not None else None
        except Exception as e:
            LOG.warning("get_last_price failed for %s: %r", symbol, e)
        return None

    async def _round_amount(self, symbol: str, amount: float) -> float:
        """Подгоняем объём под шаг лота, если исполнитель поддерживает round_qty."""
        try:
            rq = getattr(self._executor, "round_qty", None)
            if callable(rq):
                adj = await rq(symbol, float(amount))
                return float(adj)
        except Exception as e:
            LOG.debug("round_qty failed for %s: %r", symbol, e)
        return float(amount)

    # ── portfolio state (через reconcile) ─────────────────────────────────
    async def _load_open_positions_snapshot(self) -> List[Dict[str, Any]]:
        """
        Пробуем взять открытые позиции через services.reconcile (если есть),
        иначе — через исполнителя. Нормализуем в формат со следующими полями:
          {symbol, qty, entry_price, stop_loss_price}
        """
        # 1) services.reconcile.* (опционально)
        try:
            from services import reconcile  # type: ignore
            # пытаемся найти одно из expected API
            for fn_name in ("get_open_positions", "open_positions_snapshot", "positions_snapshot"):
                fn = getattr(reconcile, fn_name, None)
                if callable(fn):
                    raw = await fn()  # ожидаем list[dict]
                    return [
                        {
                            "symbol": str(p.get("symbol", "")).upper(),
                            "qty": float(p.get("qty", 0.0) or 0.0),
                            "entry_price": float(p.get("entry_price", p.get("avg_price", 0.0)) or 0.0),
                            "stop_loss_price": p.get("stop_loss_price"),
                        }
                        for p in (raw or [])
                    ]
        except Exception as e:
            LOG.debug("reconcile snapshot unavailable: %r", e)

        # 2) fallback: executor.get_positions / list_positions
        try:
            getter = getattr(self._executor, "get_positions", None) or getattr(self._executor, "list_positions", None)
            if callable(getter):
                raw = await getter(symbols=None)  # type: ignore[misc]
                out: List[Dict[str, Any]] = []
                for p in (raw or []):
                    out.append({
                        "symbol": str(p.get("symbol", "")).upper(),
                        "qty": float(p.get("qty", 0.0) or 0.0),
                        "entry_price": float(p.get("entry_price", p.get("avg_price", 0.0)) or 0.0),
                        "stop_loss_price": p.get("stop_loss_price"),
                    })
                return out
        except Exception as e:
            LOG.debug("executor positions snapshot unavailable: %r", e)

        return []

    def _calc_sl_tp_prices_for_market(
        self,
        *,
        side: Literal["buy", "sell"],
        last_price: Optional[float],
        sl_pct: Optional[float],
        tp_pct: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Для рыночного входа оцениваем цены SL/TP от last_price:
          long:  SL = P*(1-sl_pct), TP = P*(1+tp_pct)
          short: SL = P*(1+sl_pct), TP = P*(1-tp_pct)
        Если last_price отсутствует — возвращаем (None, None).
        """
        if last_price is None:
            return None, None
        p = float(last_price)
        sl = None
        tp = None
        if sl_pct and sl_pct > 0:
            sl = p * (1.0 - sl_pct) if side == "buy" else p * (1.0 + sl_pct)
        if tp_pct and tp_pct > 0:
            tp = p * (1.0 + tp_pct) if side == "buy" else p * (1.0 - tp_pct)
        return sl, tp

    # ── risk sizing + portfolio block ─────────────────────────────────────
    async def _check_risk_and_size(
        self,
        *,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
        price_hint: Optional[float],
        sl_pct: Optional[float],
    ) -> Tuple[float, Optional[str]]:
        """
        Возвращает (approved_amount, warning_msg).
        1) Проверяет дневные лимиты (count / daily DD).
        2) Делает per-trade sizing по risk.per_trade_risk_pct + sl_pct (если задан).
        3) Делает портфельную проверку:
           - вытаскивает позиции из services.reconcile (или исполнителя),
           - считает долю риска новой позиции,
           - если (used + new) > portfolio_max_risk_pct → либо авто-уменьшает объём,
             либо бросает ValueError (если auto_adjust_amount=False).
        """
        self._reset_day_counters()
        await self._ensure_day_equity()

        cfg_local = self.risk
        cfg_risk = load_risk_config()  # содержит portfolio_max_risk_pct, max_open_positions, min_sl_distance_pct

        # 1) лимит кол-ва сделок/день
        if self._day_trades_count >= int(cfg_local.daily_max_trades):
            raise ValueError(f"Daily trade limit exceeded: {self._day_trades_count} >= {cfg_local.daily_max_trades}")

        # 2) дневной стоп по equity
        if self._day_start_equity and self._day_start_equity > 0:
            cur_bal = await self._executor.fetch_balance()  # type: ignore[attr-defined]
            cur_eq = self._estimate_equity(cur_bal, cfg_local.default_quote_ccy)
            dd = (cur_eq - self._day_start_equity) / self._day_start_equity
            if dd <= -abs(cfg_local.daily_loss_limit_pct):
                raise ValueError(
                    f"Daily loss limit hit: drawdown={dd:.4f}, limit={-abs(cfg_local.daily_loss_limit_pct):.4f}"
                )

        # 3) per-trade sizing (если sl_pct дан). Если нет — просто возвращаем amount (но портфельную проверку пропустим).
        if sl_pct is None or sl_pct <= 0:
            return float(amount), "no_sl_pct"

        bal = await self._executor.fetch_balance()  # type: ignore[attr-defined]
        equity = self._estimate_equity(bal, cfg_local.default_quote_ccy)
        if equity <= 0:
            LOG.warning("Risk check: equity unknown, skip strict sizing & portfolio check")
            return float(amount), "equity_unknown"

        # оценочная цена инструмента
        est_price = price_hint if price_hint is not None else await self._get_est_price(symbol)
        if est_price is None:
            LOG.warning("Risk check: no price for %s; cannot size by risk nor portfolio check", symbol)
            return float(amount), "no_price_for_risk_check"

        # максимально допустимый нотиционал по per-trade:
        max_notional = equity * float(cfg_local.per_trade_risk_pct) / float(sl_pct)
        wanted_notional = float(amount) * float(est_price)

        approved_amount = float(amount)
        warn: Optional[str] = None

        if wanted_notional > max_notional + 1e-12:
            if not cfg_local.auto_adjust_amount:
                reason = f"per_trade_risk_exceeded:notional={wanted_notional:.8f}>allowed={max_notional:.8f}"
                LOG.warning("BLOCK open: %s %s — %s", symbol, side, reason)
                raise ValueError(reason)
            # авто-подгонка под per-trade
            approved_amount = max_notional / float(est_price)
            warn = "amount_adjusted_by_per_trade_risk"

        # 4) портфельная проверка
        try:
            positions = await self._load_open_positions_snapshot()
        except Exception as e:
            LOG.warning("Portfolio check: failed to load positions snapshot: %r", e)
            return float(approved_amount), warn or "positions_snapshot_unavailable"

        # доля риска новой позиции с учётом рассчитанного approved_amount
        # SL-цена для оценки (в абсолюте), из sl_pct и est_price:
        sl_abs = est_price * (1.0 - sl_pct) if side == "buy" else est_price * (1.0 + sl_pct)

        new_risk_frac = position_risk_fraction_from_params(
            equity=equity,
            qty=float(approved_amount),
            entry_price=float(est_price),
            stop_loss_price=float(sl_abs),
            min_sl_distance_pct=cfg_risk.min_sl_distance_pct,
        )

        # проверка (число позиций и суммарный риск)
        if not can_open_new(
            new_position_risk_fraction=new_risk_frac,
            positions=positions,
            equity=equity,
            portfolio_max_risk_pct=cfg_risk.portfolio_max_risk_pct,
            max_open_positions=cfg_risk.max_open_positions,
            min_sl_distance_pct=cfg_risk.min_sl_distance_pct,
        ):
            # Попытка авто-уменьшить объём, если включено авто-подгонка
            used = portfolio_risk_used(positions, equity=equity, min_sl_distance_pct=cfg_risk.min_sl_distance_pct)
            remaining = max(0.0, float(cfg_risk.portfolio_max_risk_pct) - float(used))

            if remaining <= 0.0 or not cfg_local.auto_adjust_amount:
                reason = (
                    f"portfolio_risk_or_positions_limit:"
                    f"used={used*100:.2f}% new={new_risk_frac*100:.2f}% "
                    f"limit={cfg_risk.portfolio_max_risk_pct*100:.2f}% "
                    f"positions={len(positions)}/{cfg_risk.max_open_positions}"
                )
                LOG.warning("BLOCK open: %s %s — %s", symbol, side, reason)
                raise ValueError(reason)

            # можно ли ужаться под оставшийся риск?
            # remaining_risk * equity = допустимый риск в деньгах
            # notional_allowed = (remaining_risk * equity) / sl_pct
            notional_allowed = (remaining * float(equity)) / float(sl_pct)
            amount_allowed = max(0.0, notional_allowed / float(est_price))

            if amount_allowed <= 0.0:
                reason = (
                    f"portfolio_risk_remaining_zero: used={used*100:.2f}% limit={cfg_risk.portfolio_max_risk_pct*100:.2f}%"
                )
                LOG.warning("BLOCK open: %s %s — %s", symbol, side, reason)
                raise ValueError(reason)

            new_amount = min(float(approved_amount), float(amount_allowed))
            if new_amount <= 0.0:
                reason = "portfolio_adjustment_zero_amount"
                LOG.warning("BLOCK open: %s %s — %s", symbol, side, reason)
                raise ValueError(reason)

            LOG.info(
                "Auto-adjust amount (portfolio): %.8f -> %.8f "
                "[used=%.4f, rem=%.4f, limit=%.4f, sl_pct=%.4f, price=%.8f]",
                approved_amount, new_amount, used, remaining, cfg_risk.portfolio_max_risk_pct, sl_pct, est_price
            )
            approved_amount = float(new_amount)
            warn = (warn or "") + "|amount_adjusted_by_portfolio"
        return float(approved_amount), warn

    def _bump_day_trades(self) -> None:
        self._day_trades_count += 1
        LOG.debug("Day trades count: %d", self._day_trades_count)

    # ------------ public API ------------

    async def configure(
        self,
        *,
        mode: Optional[Literal["sim", "binance", "ui"]] = None,
        testnet: Optional[bool] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Переключить режим/тестовую среду во время работы.
        Можно также передать новые риск-лимиты через config['risk'].
        """
        await self._swap_executor_if_needed(mode=mode, testnet=testnet, config=config)
        rc = load_risk_config()
        return {
            "mode": self._mode,
            "testnet": self._testnet,
            "failovers": self._failover_count,
            "risk": {
                "per_trade_risk_pct": self.risk.per_trade_risk_pct,
                "daily_loss_limit_pct": self.risk.daily_loss_limit_pct,
                "daily_max_trades": self.risk.daily_max_trades,
                "auto_adjust_amount": self.risk.auto_adjust_amount,
                "default_quote_ccy": self.risk.default_quote_ccy,
                # портфельные (read-only из utils.risk_config)
                "portfolio_max_risk_pct": rc.portfolio_max_risk_pct,
                "max_open_positions": rc.max_open_positions,
                "min_sl_distance_pct": rc.min_sl_distance_pct,
            },
            "assets": self.available_assets,
            "security": {
                "require_2fa": self._require_2fa,
                "roles": list(self._access.roles.keys()),
            },
        }

    # --- Market ---
    async def open_market(
        self,
        *,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
        sl_pct: Optional[float] = None,
        tp_pct: Optional[float] = None,
        client_tag: Optional[str] = None,
        security_ctx: Optional[Dict[str, str]] = None,
    ) -> OrderResult:
        """
        MARKET-ордер. amount — базовое количество (qty).
        Риск-проверка, портфельная проверка, подгонка объёма к шагу,
        защита через open_with_protection (если есть).
        """
        self._authorize(security_ctx, "trade:open")
        # риск-проверка (для рынка цену возьмём из тикера)
        approved_amount, warn = await self._check_risk_and_size(
            symbol=symbol, side=side, amount=amount, price_hint=None, sl_pct=sl_pct
        )
        # подгонка к шагу
        approved_amount = await self._round_amount(symbol, approved_amount)

        # вычислим цены SL/TP от текущей (если возможно)
        last_price = await self._get_est_price(symbol)
        sl_price, tp_price = self._calc_sl_tp_prices_for_market(
            side=side, last_price=last_price, sl_pct=sl_pct, tp_pct=tp_pct
        )

        async def _execute(ex: Executor) -> OrderResult:
            owp = getattr(ex, "open_with_protection", None)
            if callable(owp) and (sl_price is not None or tp_price is not None):
                return await owp(  # type: ignore[misc]
                    symbol=symbol,
                    side=side,
                    qty=approved_amount,
                    entry_type="market",
                    entry_price=None,
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_tag,
                )
            om = getattr(ex, "open_market", None)
            if callable(om):
                return await om(  # type: ignore[misc]
                    symbol=symbol,
                    side=side,
                    amount=approved_amount,
                    sl_pct=sl_pct,
                    tp_pct=tp_pct,
                    client_tag=client_tag,
                )
            oo = getattr(ex, "open_order")
            return await oo(  # type: ignore[misc]
                symbol=symbol,
                side=side,
                type="market",
                qty=approved_amount,
                client_order_id=client_tag,
            )

        started = time.perf_counter()
        res = await self._with_failover(_execute, context=f"open_market:{symbol}")

        success = True
        try:
            status = getattr(res, "status", None) or (res.get("status") if isinstance(res, dict) else None)
            normalized = str(status).lower() if status is not None else ""
            success = normalized not in {"rejected", "expired"}
            if success:
                self._bump_day_trades()
        except Exception:
            self._bump_day_trades()
            success = True

        OBSERVABILITY.record_order(
            symbol=symbol,
            success=success,
            latency=time.perf_counter() - started,
        )

        if warn:
            LOG.info("open_market(%s %s) warn=%s qty=%.8f", symbol, side, warn, approved_amount)
        return res

    # --- Limit ---
    async def open_limit(
        self,
        *,
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
        price: float,
        tif: Literal["GTC", "IOC", "FOK"] = "GTC",
        client_tag: Optional[str] = None,
        sl_pct: Optional[float] = None,
        tp_pct: Optional[float] = None,
        security_ctx: Optional[Dict[str, str]] = None,
    ) -> OrderResult:
        """
        LIMIT-ордер. Риск-проверка использует limit-цену как hint.
        Если исполнитель поддерживает open_with_protection и заданы sl/tp — добавляем защиту.
        """
        self._authorize(security_ctx, "trade:open")

        approved_amount, warn = await self._check_risk_and_size(
            symbol=symbol, side=side, amount=amount, price_hint=float(price), sl_pct=sl_pct
        )
        approved_amount = await self._round_amount(symbol, approved_amount)

        async def _execute(ex: Executor) -> OrderResult:
            owp = getattr(ex, "open_with_protection", None)
            if callable(owp) and (sl_pct or tp_pct):
                sl_price, tp_price = None, None
                if sl_pct and sl_pct > 0:
                    sl_price = price * (1.0 - sl_pct) if side == "buy" else price * (1.0 + sl_pct)
                if tp_pct and tp_pct > 0:
                    tp_price = price * (1.0 + tp_pct) if side == "buy" else price * (1.0 - tp_pct)

                return await owp(  # type: ignore[misc]
                    symbol=symbol,
                    side=side,
                    qty=approved_amount,
                    entry_type="limit",
                    entry_price=float(price),
                    sl_price=sl_price,
                    tp_price=tp_price,
                    client_order_id=client_tag,
                    timeInForce=tif,
                )
            ol = getattr(ex, "open_limit", None)
            if callable(ol):
                return await ol(  # type: ignore[misc]
                    symbol=symbol,
                    side=side,
                    amount=approved_amount,
                    price=float(price),
                    time_in_force=tif,
                    client_tag=client_tag,
                )
            oo = getattr(ex, "open_order")
            return await oo(  # type: ignore[misc]
                symbol=symbol,
                side=side,
                type="limit",
                qty=approved_amount,
                price=float(price),
                timeInForce=tif,
                client_order_id=client_tag,
            )

        started = time.perf_counter()
        res = await self._with_failover(_execute, context=f"open_limit:{symbol}")

        success = True
        try:
            status = getattr(res, "status", None) or (res.get("status") if isinstance(res, dict) else None)
            normalized = str(status).lower() if status is not None else ""
            success = normalized not in {"rejected", "expired"}
            if success:
                self._bump_day_trades()
        except Exception:
            self._bump_day_trades()
            success = True

        OBSERVABILITY.record_order(
            symbol=symbol,
            success=success,
            latency=time.perf_counter() - started,
        )

        if warn:
            LOG.info("open_limit(%s %s @ %.8f) warn=%s qty=%.8f", symbol, side, price, warn, approved_amount)
        return res

    # --- Cancel / Close ---
    async def cancel_order(
        self,
        *,
        symbol: str,
        order_id: Optional[str] = None,
        client_order_id: Optional[str] = None,
    ) -> OrderResult:
        """
        Отмена отложенного ордера по order_id или client_order_id.
        """
        # прямой метод, если есть
        co = getattr(self._executor, "cancel_order", None)
        if callable(co):
            return await co(symbol=symbol, order_id=order_id, client_order_id=client_order_id)  # type: ignore[misc]
        # фоллбек — delete_order (Binance)
        do = getattr(self._executor, "client", None)
        if do and hasattr(do, "delete_order"):
            return await do.delete_order(symbol=symbol, orderId=order_id, origClientOrderId=client_order_id)  # type: ignore[misc]
        raise NotImplementedError("Executor does not support cancel_order")

    async def close_all(self, *, symbol: str, security_ctx: Optional[Dict[str, str]] = None) -> List[OrderResult]:
        """
        «Закрыть всё» по инструменту: для спота — SELL рыночным на весь свободный BASE.
        В симуляторе — мгновенно.
        """
        self._authorize(security_ctx, "trade:close")
        async def _execute(ex: Executor) -> List[OrderResult]:
            ca = getattr(ex, "close_all", None)
            if callable(ca):
                return await ca(symbol=symbol)  # type: ignore[misc]

            get_pos = getattr(ex, "get_positions", None) or getattr(ex, "list_positions", None)
            pos_list: List[Dict[str, Any]] = []
            if callable(get_pos):
                pos_list = await get_pos(symbols=[symbol])  # type: ignore[misc]

            qty_to_close = 0.0
            for p in (pos_list or []):
                if isinstance(p, dict):
                    sym = str(p.get("symbol", ""))
                    qty_candidate = p.get("qty") or p.get("total_base") or p.get("amount") or 0.0
                else:
                    sym = str(getattr(p, "symbol", ""))
                    qty_candidate = getattr(p, "qty", getattr(p, "total_base", 0.0))
                if sym.upper() == symbol.upper():
                    qty_to_close = float(qty_candidate)
                    break

            if qty_to_close == 0.0:
                return []

            side_close = "sell" if qty_to_close > 0 else "buy"
            qty_abs = abs(qty_to_close)
            close_order = getattr(ex, "close_order", None)
            if callable(close_order):
                res = await close_order(symbol=symbol, qty=qty_abs, type="market")  # type: ignore[misc]
                return [res]  # type: ignore[return-value]

            om = getattr(ex, "open_market", None)
            if callable(om):
                res = await om(  # type: ignore[misc]
                    symbol=symbol,
                    side=side_close,
                    amount=qty_abs,
                    sl_pct=None,
                    tp_pct=None,
                    client_tag="auto-close",
                )
                return [res]

            open_order = getattr(ex, "open_order", None)
            if callable(open_order):
                res = await open_order(  # type: ignore[misc]
                    symbol=symbol,
                    side=side_close,
                    type="market",
                    qty=qty_abs,
                    client_order_id="auto-close",
                )
                return [res]

            raise NotImplementedError("Executor does not support close_all for this mode")

        return await self._with_failover(_execute, context=f"close_all:{symbol}")

    # --- Read only ---
    async def get_positions(
        self,
        *,
        symbols: Optional[List[str]] = None,
        security_ctx: Optional[Dict[str, str]] = None,
    ) -> List[Position]:
        self._authorize(security_ctx, "trade:view")

        async def _execute(ex: Executor) -> List[Position]:
            gp = getattr(ex, "get_positions", None) or getattr(ex, "list_positions", None)
            if callable(gp):
                return await gp(symbols=self._symbols_arg(symbols))  # type: ignore[misc]
            return []  # type: ignore[return-value]

        return await self._with_failover(_execute, context="get_positions", allow_ui=True)

    async def get_balance(self) -> Dict[str, Any]:
        """Баланс счёта/кошелька (free/locked/total), если исполнитель поддерживает."""
        fb = getattr(self._executor, "fetch_balance", None)
        if callable(fb):
            return await fb()  # type: ignore[misc]
        return {}

    # --- Унифицированная «ручка» (удобно дергать из контроллера) ---
    async def open(
        self,
        *,
        type: Literal["market", "limit"],
        symbol: str,
        side: Literal["buy", "sell"],
        amount: float,
        price: Optional[float] = None,
        sl_pct: Optional[float] = None,
        tp_pct: Optional[float] = None,
        tif: Literal["GTC", "IOC", "FOK"] = "GTC",
        client_tag: Optional[str] = None,
        security_ctx: Optional[Dict[str, str]] = None,
    ) -> OrderResult:
        """Единая точка входа: type == market|limit, с риск-проверками и защитами."""
        if type == "market":
            return await self.open_market(
                symbol=symbol,
                side=side,
                amount=amount,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                client_tag=client_tag,
                security_ctx=security_ctx,
            )
        if price is None:
            raise ValueError("price is required for limit orders")
        return await self.open_limit(
            symbol=symbol,
            side=side,
            amount=amount,
            price=float(price),
            tif=tif,
            client_tag=client_tag,
            sl_pct=sl_pct,
            tp_pct=tp_pct,
            security_ctx=security_ctx,
        )

    # --- Информация о текущем режиме/рисках (для /health, /exec/*) ---
    def info(self) -> Dict[str, Any]:
        rc = load_risk_config()
        return {
            "mode": self._mode,
            "testnet": self._testnet,
            "executor": getattr(self._executor, "name", "unknown"),
            "risk": {
                "per_trade_risk_pct": self.risk.per_trade_risk_pct,
                "daily_loss_limit_pct": self.risk.daily_loss_limit_pct,
                "daily_max_trades": self.risk.daily_max_trades,
                "auto_adjust_amount": self.risk.auto_adjust_amount,
                "default_quote_ccy": self.risk.default_quote_ccy,
                # портфельные (read-only)
                "portfolio_max_risk_pct": rc.portfolio_max_risk_pct,
                "max_open_positions": rc.max_open_positions,
                "min_sl_distance_pct": rc.min_sl_distance_pct,
            },
            "day": self._day.isoformat(),
            "day_trades": self._day_trades_count,
            "day_start_equity": self._day_start_equity,
            "failovers": self._failover_count,
            "assets": self.available_assets,
            "security": {
                "require_2fa": self._require_2fa,
                "roles": list(self._access.roles.keys()),
            },
        }


def _to_market_features(symbol: str, feats: Dict[str, Any]) -> MarketFeatures:
    data: Dict[str, Any] = {k: v for k, v in (feats or {}).items() if v is not None}
    data.setdefault("symbol", symbol)
    return cast(MarketFeatures, data)


def decide_and_execute(
    symbol: str,
    feats: Dict[str, Any],
    news_text: Optional[str] | None = None,
    *,
    router: Optional["ModelRouter"] = None,
) -> Dict[str, Any]:
    """Generate a model-driven decision with Kelly-capped risk sizing."""

    active_router = router or router_singleton
    if active_router is None:
        raise RuntimeError("Model router is not initialised")

    features = _to_market_features(symbol, feats)
    signal = active_router.signal(features)
    sentiment = active_router.sentiment(news_text or "")
    regime = active_router.regime(features)

    risk_cfg = active_router.risk_config
    per_trade_cap = float(risk_cfg.get("per_trade_cap", 0.02))
    daily_loss_limit = float(risk_cfg.get("daily_loss_limit", 0.06))

    base_fraction = clamp01(signal["confidence"] * 0.5)
    risk_fraction = min(per_trade_cap, per_trade_cap * base_fraction)
    if signal["signal"] == "hold":
        risk_fraction = 0.0

    if sentiment["label"] == "neg" or regime["regime"] == "storm":
        risk_fraction *= 0.5

    price = float(features.get("price", 0.0) or 0.0)
    atr = float(features.get("atr", 0.0) or 0.0)
    atr_pct = clamp01(atr / price) if price > 0 else 0.0
    if atr_pct > 0:
        risk_fraction *= clamp01(1.0 - min(0.9, atr_pct))

    equity = float(features.get("equity", feats.get("account_equity", 0.0)) or 0.0)
    day_pnl = float(features.get("day_pnl", feats.get("daily_pnl", 0.0)) or 0.0)

    trading_blocked = False
    if equity > 0 and day_pnl <= -daily_loss_limit * equity:
        trading_blocked = True
        risk_fraction = 0.0

    return {
        "symbol": symbol,
        "signal": signal,
        "sentiment": sentiment,
        "regime": regime,
        "risk_fraction": float(risk_fraction),
        "trading_blocked": trading_blocked,
        "limits": {
            "per_trade_cap": per_trade_cap,
            "daily_loss_limit": daily_loss_limit,
            "atr_pct": atr_pct,
            "equity": equity,
            "day_pnl": day_pnl,
        },
    }
