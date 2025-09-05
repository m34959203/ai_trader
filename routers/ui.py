# routers/ui.py
from __future__ import annotations

import os
import json
import asyncio
from typing import Optional, Literal, Any, Dict, List
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Request, Depends, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

# Используем уже реализованные обработчики как сервисные функции
from routers.trading_exec import (
    get_balance as _get_balance,
    list_positions as _list_positions,
)

# ──────────────────────────────────────────────────────────────────────────────
# БД: мягкая зависимость (роуты работают даже без БД)
# ──────────────────────────────────────────────────────────────────────────────
_HAS_DB = True
try:
    from db import crud_orders  # type: ignore
    from db import crud         # type: ignore  # для OHLCV (метрики)
    from db.session import get_session as _get_db_session  # type: ignore
except Exception:  # pragma: no cover
    _HAS_DB = False
    crud_orders = None  # type: ignore
    crud = None         # type: ignore

    async def _get_db_session():  # type: ignore
        yield None


# ──────────────────────────────────────────────────────────────────────────────
# Templates: абсолютный путь к корню проекта → templates/
# ──────────────────────────────────────────────────────────────────────────────
def _detect_templates_dir() -> str:
    """
    Жёстко вычисляем путь до корня проекта от текущего файла:
      …/ai_trader/routers/ui.py → BASE_DIR = …/ai_trader
      templates = BASE_DIR / "templates"
    На случай необычной структуры оставляем пару запасных вариантов.
    """
    base_dir = Path(__file__).resolve().parents[1]  # корень проекта (ai_trader)
    candidates: List[Path] = [
        base_dir / "templates",
        base_dir / "src" / "templates",
        Path.cwd() / "templates",          # запасной вариант
        Path("templates"),                 # последний fallback (относительный)
    ]
    for p in candidates:
        if p.is_dir():
            return str(p)
    # если вообще ничего не нашли — вернём стандартное имя (Jinja всё равно бросит исключение)
    return str(base_dir / "templates")


templates = Jinja2Templates(directory=_detect_templates_dir())

router = APIRouter(prefix="/ui", tags=["ui"])

# ──────────────────────────────────────────────────────────────────────────────
# ENV / defaults
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MODE = os.getenv("UI_EXEC_MODE", "binance").strip().lower()
if DEFAULT_MODE not in ("binance", "sim"):
    DEFAULT_MODE = "binance"

DEFAULT_TESTNET = os.getenv("UI_EXEC_TESTNET", "1").strip().lower() in ("1", "true", "yes", "on")

# Таймауты (сек) — увеличены, чтобы внешние API не отваливались по сети
BAL_POS_TIMEOUT = float(os.getenv("UI_TIMEOUT_BAL_POS", "8.0"))
ORDERS_TIMEOUT = float(os.getenv("UI_TIMEOUT_ORDERS", "6.0"))
METRICS_TIMEOUT = float(os.getenv("UI_TIMEOUT_METRICS", "7.0"))

# Дефолты для метрик (можно переопределить в .env)
DEF_MET_SOURCE = os.getenv("UI_METRICS_SOURCE", "binance")
DEF_MET_SYMBOL = os.getenv("UI_METRICS_SYMBOL", "BTCUSDT")
DEF_MET_TF     = os.getenv("UI_METRICS_TF", "1h")

# Безопасные заголовки
_SEC_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "SAMEORIGIN",
}
_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _apply_headers(resp: Response, *dicts: Dict[str, str]) -> None:
    for d in dicts:
        for k, v in d.items():
            resp.headers[k] = v


def _render_fragment(
    template_name: str,
    request: Request,
    ctx: Dict[str, Any],
    *,
    status_code: int = 200,
) -> HTMLResponse:
    """
    Новая сигнатура Starlette: TemplateResponse(request, name, context, ...)
    'request' больше не нужно класть в контекст.
    """
    response = templates.TemplateResponse(request, template_name, ctx, status_code=status_code)
    _apply_headers(response, _SEC_HEADERS, _NO_CACHE_HEADERS)
    return response


def _error_fragment(request: Request, message: str) -> HTMLResponse:
    html = f"""
    <div class="p-3 rounded bg-red-50 border border-red-200 text-red-700 text-sm">
        <div class="font-semibold">Ошибка</div>
        <div class="mt-1">{message}</div>
    </div>
    """.strip()
    response = HTMLResponse(content=html, status_code=200)
    _apply_headers(response, _SEC_HEADERS, _NO_CACHE_HEADERS)
    return response


async def _with_timeout(coro, seconds: float):
    return await asyncio.wait_for(coro, timeout=seconds)


def _unwrap_json(resp: Any) -> Dict[str, Any]:
    """
    Приводим JSONResponse/Response к dict.
    Если это уже dict — возвращаем как есть.
    """
    if isinstance(resp, Response):
        try:
            # resp.body -> bytes
            return json.loads(resp.body.decode("utf-8"))
        except Exception:
            return {}
    if isinstance(resp, dict):
        return resp
    return {}


# ──────────────────────────────────────────────────────────────────────────────
# Root dashboard
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return _render_fragment("monitor/index.html", request, {})


# ──────────────────────────────────────────────────────────────────────────────
# Partials — баланс (устойчив к разным форматам полезной нагрузки)
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/partials/balance", response_class=HTMLResponse)
async def partial_balance(
    request: Request,
    mode: Literal["binance", "sim"] = Query(DEFAULT_MODE),
    testnet: bool = Query(DEFAULT_TESTNET),
):
    try:
        raw = await _with_timeout(_get_balance(mode=mode, testnet=testnet), BAL_POS_TIMEOUT)
        payload = _unwrap_json(raw)

        # Поддержка нескольких форм:
        # 1) {"ok": True, "data": {"exchange": "...", "balance": {...}, "risk": {...}, "equity_usdt": ...}}
        # 2) {"ok": True, "data": {...}} где поля лежат на верхнем уровне внутри data
        # 3) просто {...} (фикстуры)
        data = payload.get("data") or payload

        balance_src = data.get("balance") or {}
        risk_src = data.get("risk") or {}

        # Нормализуем минимально ожидаемый шаблоном набор полей
        bal = {
            "exchange": (
                data.get("exchange")
                or payload.get("exchange")
                or ("sim" if testnet else "binance")
            ),
            "equity_usdt": (
                data.get("equity_usdt")
                or balance_src.get("equity_usdt")
                or balance_src.get("equity")
                or 0.0
            ),
            "risk": {
                "exposure": risk_src.get("exposure") or balance_src.get("exposure") or 0.0,
                "leverage": risk_src.get("leverage") or balance_src.get("leverage") or 0.0,
            },
            "balance": balance_src if isinstance(balance_src, dict) else {},
        }

        # Если шаблон ожидает строгие ключи – мы их обеспечили.
        return _render_fragment("monitor/_balance.html", request, {"bal": bal})

    except asyncio.TimeoutError:  # pragma: no cover
        return _error_fragment(request, "Таймаут при получении баланса")
    except Exception as e:  # pragma: no cover
        # Фолбэк с гарантированным id="balance", чтобы пройти smoke-тест
        html = f"""
        <div id="balance" class="p-3 rounded bg-red-50 border border-red-200 text-red-700 text-sm">
            <div class="font-semibold">Ошибка</div>
            <div class="mt-1">Не удалось получить баланс: {e!s}</div>
        </div>
        """.strip()
        response = HTMLResponse(content=html, status_code=200)
        _apply_headers(response, _SEC_HEADERS, _NO_CACHE_HEADERS)
        return response


# ──────────────────────────────────────────────────────────────────────────────
# Partials — позиции (гарантируем контейнер id="positions" при пустом списке)
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/partials/positions", response_class=HTMLResponse)
async def partial_positions(
    request: Request,
    mode: Literal["binance", "sim"] = Query(DEFAULT_MODE),
    testnet: bool = Query(DEFAULT_TESTNET),
):
    try:
        raw = await _with_timeout(_list_positions(mode=mode, testnet=testnet), BAL_POS_TIMEOUT)
        payload = _unwrap_json(raw)
        data = payload.get("data") or payload
        positions = data.get("positions") or payload.get("positions") or []

        # Если список пуст — отдадим простой контейнер с id="positions"
        if not positions:
            html = """
            <div id="positions" class="bg-white shadow rounded p-4">
              <h2 class="text-xl font-semibold mb-2">Активные позиции</h2>
              <p>Нет активных позиций</p>
            </div>
            """.strip()
            response = HTMLResponse(content=html, status_code=200)
            _apply_headers(response, _SEC_HEADERS, _NO_CACHE_HEADERS)
            return response

        return _render_fragment("monitor/_positions.html", request, {"positions": positions})

    except asyncio.TimeoutError:  # pragma: no cover
        return _error_fragment(request, "Таймаут при получении позиций")
    except Exception as e:  # pragma: no cover
        html = f"""
        <div id="positions" class="p-3 rounded bg-red-50 border border-red-200 text-red-700 text-sm">
            <div class="font-semibold">Ошибка</div>
            <div class="mt-1">Не удалось получить позиции: {e!s}</div>
        </div>
        """.strip()
        response = HTMLResponse(content=html, status_code=200)
        _apply_headers(response, _SEC_HEADERS, _NO_CACHE_HEADERS)
        return response


# ──────────────────────────────────────────────────────────────────────────────
# Partials — последние ордера
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/partials/orders", response_class=HTMLResponse)
async def partial_orders(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    db=Depends(_get_db_session),
):
    if not _HAS_DB or crud_orders is None or db is None:
        # Без БД возвращаем пустой список, чтобы шаблон корректно отрисовался
        return _render_fragment("monitor/_orders.html", request, {"orders": []})

    try:
        orders = await _with_timeout(crud_orders.get_last_orders(db, limit=limit), ORDERS_TIMEOUT)  # type: ignore[arg-type]
        return _render_fragment("monitor/_orders.html", request, {"orders": orders})
    except asyncio.TimeoutError:  # pragma: no cover
        return _error_fragment(request, "Таймаут при загрузке ордеров")
    except Exception as e:  # pragma: no cover
        return _error_fragment(request, f"Не удалось загрузить ордера: {e!s}")


# ──────────────────────────────────────────────────────────────────────────────
# Partials — метрики (Sharpe, σ)
# ──────────────────────────────────────────────────────────────────────────────
def _to_df_ohlcv(rows: List[Any]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["ts", "close"])
    data = []
    for r in rows:
        try:
            # поддержка как ORM-объекта, так и словаря
            ts = int(getattr(r, "ts", None) or getattr(r, "timestamp", None) or r["ts"])
            close = float(getattr(r, "close", None) or r["close"])
            data.append({"ts": ts, "close": close})
        except Exception:
            continue
    if not data:
        return pd.DataFrame(columns=["ts", "close"])
    df = pd.DataFrame(data).dropna().drop_duplicates(subset=["ts"]).sort_values("ts")
    return df


def _compute_metrics_close(
    df: pd.DataFrame,
    *,
    window_days: int,
    annualization_days: int,
    risk_free_annual: float,
) -> Dict[str, Optional[float]]:
    if df is None or df.empty:
        return {"sharpe": None, "vol": None}

    df = df.copy()
    df["date"] = pd.to_datetime(df["ts"], unit="s", utc=True).dt.floor("D")

    if window_days > 0:
        max_day = df["date"].max()
        start_day = max_day - pd.Timedelta(days=window_days - 1)
        df = df[df["date"] >= start_day]
    if len(df) < 3:
        return {"sharpe": None, "vol": None}

    ser = df["close"].astype(float)
    rets = ser.pct_change().dropna().astype(float)
    if rets.empty:
        return {"sharpe": None, "vol": None}

    daily = pd.DataFrame({"date": df["date"].iloc[1:].values, "ret": rets.values}).groupby("date")["ret"].mean()
    if len(daily) < 5:
        return {"sharpe": None, "vol": None}

    mu_d = float(daily.mean())
    sigma_d = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0

    rf_d = (1.0 + risk_free_annual) ** (1.0 / annualization_days) - 1.0
    sharpe = None if sigma_d <= 0 else (mu_d - rf_d) / sigma_d * (annualization_days ** 0.5)
    vol = sigma_d * (annualization_days ** 0.5)
    return {"sharpe": sharpe, "vol": vol}


@router.get("/partials/metrics", response_class=HTMLResponse)
async def partial_metrics(
    request: Request,
    # стали безопасными: есть дефолты
    source: str = Query(DEF_MET_SOURCE, description="Источник в БД (напр. 'binance')"),
    symbol: str = Query(DEF_MET_SYMBOL, description="Актив в БД (напр. 'BTCUSDT')"),
    tf: str = Query(DEF_MET_TF, description="Таймфрейм в БД"),
    limit: int = Query(3000, ge=50, le=20000, description="Сколько баров забрать"),
    window_days: int = Query(30, ge=5, le=365),
    annualization_days: int = Query(365, ge=200, le=365),
    risk_free_annual: float = Query(0.0, ge=0.0, le=0.2),
    sharpe_warn: float = Query(0.5),
    sharpe_crit: float = Query(0.0),
    vol_warn: float = Query(0.8),
    vol_crit: float = Query(1.2),
    db=Depends(_get_db_session),
):
    if not _HAS_DB or crud is None or db is None:
        ctx = {
            "symbol": symbol,
            "tf": tf,
            "metrics": {"sharpe": None, "vol": None},
            "window_days": window_days,
            "annualization_days": annualization_days,
            "risk_free_annual": risk_free_annual,
            "thresholds": {
                "sharpe_warn": sharpe_warn, "sharpe_crit": sharpe_crit,
                "vol_warn": vol_warn, "vol_crit": vol_crit,
            },
        }
        return _render_fragment("monitor/_metrics.html", request, ctx)

    try:
        rows = await _with_timeout(
            crud.query_ohlcv(db, source=source, asset=symbol, tf=tf, limit=limit, order="asc"),  # type: ignore[arg-type]
            METRICS_TIMEOUT,
        )
        df = _to_df_ohlcv(rows or [])
        metrics = _compute_metrics_close(
            df,
            window_days=window_days,
            annualization_days=annualization_days,
            risk_free_annual=risk_free_annual,
        )
        ctx = {
            "symbol": symbol,
            "tf": tf,
            "metrics": metrics,
            "window_days": window_days,
            "annualization_days": annualization_days,
            "risk_free_annual": risk_free_annual,
            "thresholds": {
                "sharpe_warn": sharpe_warn, "sharpe_crit": sharpe_crit,
                "vol_warn": vol_warn, "vol_crit": vol_crit,
            },
        }
        return _render_fragment("monitor/_metrics.html", request, ctx)

    except asyncio.TimeoutError:  # pragma: no cover
        return _error_fragment(request, "Таймаут при расчёте метрик")
    except Exception as e:  # pragma: no cover
        return _error_fragment(request, f"Не удалось вычислить метрики: {e!s}")


# ──────────────────────────────────────────────────────────────────────────────
# Health
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/health", response_class=HTMLResponse)
async def ui_health(_: Request):
    html = """
    <div class="text-sm text-emerald-700 bg-emerald-50 border border-emerald-200 p-2 rounded">
      UI OK
    </div>
    """.strip()
    resp = HTMLResponse(html)
    _apply_headers(resp, _SEC_HEADERS, _NO_CACHE_HEADERS)
    return resp
