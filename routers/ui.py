from __future__ import annotations

import os
import json
import asyncio
import hashlib
import csv
import io
from typing import Optional, Literal, Any, Dict, List, Tuple
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, Request, Depends, Query, Response, status, HTTPException
from fastapi.responses import HTMLResponse
from sqlalchemy import select
from fastapi.templating import Jinja2Templates

# Используем уже реализованные обработчики как сервисные функции
from routers.trading_exec import (
    get_balance as _get_balance,
    list_positions as _list_positions,
)
from monitoring.reporting import build_dashboard_state
from utils.risk_config import load_risk_config

# ──────────────────────────────────────────────────────────────────────────────
# БД: мягкая зависимость (роуты работают даже без БД)
# ──────────────────────────────────────────────────────────────────────────────
_HAS_DB = True
try:
    from db import crud_orders  # type: ignore
    from db import crud         # type: ignore
    from db import crud_news    # type: ignore
    from db.models_orders import OrderLog  # type: ignore
    from db.session import get_session as _get_db_session  # type: ignore
except Exception:  # pragma: no cover
    _HAS_DB = False
    crud_orders = None  # type: ignore
    crud = None         # type: ignore
    crud_news = None    # type: ignore
    OrderLog = None     # type: ignore

    async def _get_db_session():  # type: ignore
        yield None

try:
    from services.news_pipeline import load_latest_sentiment, SentimentContext
    _HAS_NEWS_INTEL = True
except Exception:  # pragma: no cover
    load_latest_sentiment = None  # type: ignore
    SentimentContext = None  # type: ignore
    _HAS_NEWS_INTEL = False

# ──────────────────────────────────────────────────────────────────────────────
# Templates
# ──────────────────────────────────────────────────────────────────────────────
def _detect_templates_dir() -> str:
    base_dir = Path(__file__).resolve().parents[1]
    for p in [
        base_dir / "templates",
        base_dir / "src" / "templates",
        Path.cwd() / "templates",
        Path("templates"),
    ]:
        if p.is_dir():
            return str(p)
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

BAL_POS_TIMEOUT = float(os.getenv("UI_TIMEOUT_BAL_POS", "8.0"))
ORDERS_TIMEOUT  = float(os.getenv("UI_TIMEOUT_ORDERS", "6.0"))
METRICS_TIMEOUT = float(os.getenv("UI_TIMEOUT_METRICS", "7.0"))
INTEL_TIMEOUT = float(os.getenv("UI_TIMEOUT_INTEL", "7.0"))

DEF_MET_SOURCE = os.getenv("UI_METRICS_SOURCE", "binance")
DEF_MET_SYMBOL = os.getenv("UI_METRICS_SYMBOL", "BTCUSDT")
DEF_MET_TF     = os.getenv("UI_METRICS_TF", "1h")

_SEC_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "SAMEORIGIN",
}
_NO_CACHE_HEADERS = {
    # no-store гарантирует отсутствие кэширования HTML-фрагментов при polling
    "Cache-Control": "no-store",
}

# ──────────────────────────────────────────────────────────────────────────────
# Stateless hash-dedupe (per key)
# ──────────────────────────────────────────────────────────────────────────────
_LAST_DIGEST: Dict[str, str] = {}

def _sha1_of(obj: Any) -> str:
    try:
        data = json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:
        data = str(obj)
    return hashlib.sha1(data.encode("utf-8")).hexdigest()

def _apply_headers(resp: Response, *dicts: Dict[str, str]) -> None:
    for d in dicts:
        for k, v in d.items():
            resp.headers[k] = v

def _is_hx(request: Request) -> bool:
    # HTMX помечает запрос заголовком HX-Request: true
    return request.headers.get("HX-Request", "").lower() == "true"

def _render_fragment(
    template_name: str,
    request: Request,
    ctx: Dict[str, Any],
    *,
    status_code: int = 200,
) -> HTMLResponse:
    context = {"request": request, **ctx}
    response = templates.TemplateResponse(template_name, context, status_code=status_code)
    _apply_headers(response, _SEC_HEADERS, _NO_CACHE_HEADERS)
    return response

def _error_fragment(element_id: str, message: str) -> HTMLResponse:
    """
    Возвращает фрагмент ошибки с гарантированным id — важно для HTMX swap.
    """
    html = f"""
    <div id="{element_id}" class="p-3 rounded bg-red-50 border border-red-200 text-red-700 text-sm">
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
    if isinstance(resp, Response):
        try:
            # JSONResponse/Response.body доступно сразу
            return json.loads(resp.body.decode("utf-8"))
        except Exception:
            return {}
    if isinstance(resp, dict):
        return resp
    return {}

def _to_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

def _to_int(v: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default

def _maybe_204(request: Request, key: str, signature_obj: Any) -> Optional[Response]:
    """
    Если это HTMX-запрос и подпись не изменилась — возвращаем 204 No Content,
    иначе обновляем подпись и продолжаем нормальный рендеринг.
    """
    if not _is_hx(request):
        return None
    new_digest = _sha1_of(signature_obj)
    old_digest = _LAST_DIGEST.get(key)
    if old_digest == new_digest:
        r = Response(status_code=status.HTTP_204_NO_CONTENT)
        _apply_headers(r, _SEC_HEADERS, _NO_CACHE_HEADERS)
        return r
    _LAST_DIGEST[key] = new_digest
    return None

# ──────────────────────────────────────────────────────────────────────────────
# Root dashboard
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return _render_fragment("monitor/index.html", request, {})

# ──────────────────────────────────────────────────────────────────────────────
# Partials — баланс
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/partials/balance", response_class=HTMLResponse)
async def partial_balance(
    request: Request,
    mode: Literal["binance", "sim"] = Query(DEFAULT_MODE),
    testnet: bool = Query(DEFAULT_TESTNET),
):
    def _demo_balance() -> Dict[str, Any]:
        return {
            "exchange": "sim" if not testnet else "sim:testnet",
            "equity_usdt": 10000.0,
            "risk": {
                "daily_start_equity": 10000.0,
                "daily_max_loss_pct": 0.02,
                "max_trades_per_day": 50,
                "exposure": 0.0,
                "leverage": 1.0,
            },
            "balances": [
                {"asset": "USDT", "free": 10000.0, "locked": 0.0, "total": 10000.0}
            ],
            "_demo": True,
        }

    try:
        raw = await _with_timeout(_get_balance(mode=mode, testnet=testnet), BAL_POS_TIMEOUT)
        payload = _unwrap_json(raw)
        data = payload.get("data") or payload or {}

        balance_src = data.get("balance") or {}
        risk_src    = data.get("risk") or {}
        balances_src = data.get("balances") or []
        if isinstance(balance_src, dict) and "balances" in balance_src:
            balances_src = balance_src["balances"]

        equity_candidates = [
            data.get("equity_usdt"),
            balance_src.get("equity_usdt") if isinstance(balance_src, dict) else None,
            balance_src.get("equity") if isinstance(balance_src, dict) else None,
            data.get("equity"),
            data.get("total"),
            balance_src.get("total") if isinstance(balance_src, dict) else None,
            balance_src.get("free") if isinstance(balance_src, dict) else None,
        ]
        equity_usdt = next((v for v in equity_candidates if v is not None), None)
        equity_usdt = _to_float(equity_usdt)

        if equity_usdt is None and mode == "sim":
            # Демо-данные
            demo = _demo_balance()
            # сигнатура для 204
            if (r := _maybe_204(request, f"balance:{mode}:{testnet}", {"equity": demo["equity_usdt"], "n": len(demo["balances"])})) is not None:
                return r
            return _render_fragment("monitor/_balance.html", request, {"bal": demo})

        bal = {
            "exchange": (
                data.get("exchange")
                or payload.get("exchange")
                or (f"{mode}:testnet" if testnet else mode)
            ),
            "equity_usdt": equity_usdt,
            "risk": {
                "daily_start_equity": _to_float(risk_src.get("daily_start_equity")),
                "daily_max_loss_pct": _to_float(risk_src.get("daily_max_loss_pct")),
                "max_trades_per_day": _to_int(risk_src.get("max_trades_per_day")),
                "exposure": _to_float(
                    (risk_src.get("exposure") if isinstance(risk_src, dict) else None)
                    or (balance_src.get("exposure") if isinstance(balance_src, dict) else None),
                    0.0,
                ),
                "leverage": _to_float(
                    (risk_src.get("leverage") if isinstance(risk_src, dict) else None)
                    or (balance_src.get("leverage") if isinstance(balance_src, dict) else None),
                    0.0,
                ),
            },
            "balances": balances_src,
            "balance": balance_src if isinstance(balance_src, dict) else {},
            "_demo": False,
        }

        # Подпись «существенных» полей для 204
        sig = {
            "exchange": bal["exchange"],
            "equity": bal["equity_usdt"],
            "balances_n": len(balances_src or []),
        }
        if (r := _maybe_204(request, f"balance:{mode}:{testnet}", sig)) is not None:
            return r

        return _render_fragment("monitor/_balance.html", request, {"bal": bal})

    except asyncio.TimeoutError:
        if mode == "sim":
            demo = _demo_balance()
            if (r := _maybe_204(request, f"balance:{mode}:{testnet}", {"equity": demo["equity_usdt"], "n": len(demo["balances"])})) is not None:
                return r
            return _render_fragment("monitor/_balance.html", request, {"bal": demo})
        return _error_fragment("balance", "Таймаут при получении баланса")
    except Exception as e:
        if mode == "sim":
            demo = _demo_balance()
            if (r := _maybe_204(request, f"balance:{mode}:{testnet}", {"equity": demo["equity_usdt"], "n": len(demo["balances"])})) is not None:
                return r
            return _render_fragment("monitor/_balance.html", request, {"bal": demo})
        return _error_fragment("balance", f"Не удалось получить баланс: {e!s}")

# ──────────────────────────────────────────────────────────────────────────────
# Partials — позиции
# ──────────────────────────────────────────────────────────────────────────────
def _positions_signature(positions: List[Dict[str, Any] | Any]) -> Dict[str, Any]:
    n = len(positions or [])
    # Строим компактную подпись по ключевым полям, если они есть
    sample: List[Tuple] = []
    keys = ("symbol", "asset", "position", "qty", "size", "leverage", "side", "entryPrice", "avgPrice", "unrealizedPnl", "updateTime")
    for p in (positions or [])[:10]:
        row = []
        if isinstance(p, dict):
            for k in keys:
                row.append(p.get(k))
        else:
            for k in keys:
                row.append(getattr(p, k, None))
        sample.append(tuple(row))
    return {"n": n, "sample": sample}

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

        # 204, если состав позиций не изменился
        sig = _positions_signature(positions)
        if (r := _maybe_204(request, f"positions:{mode}:{testnet}", sig)) is not None:
            return r

        return _render_fragment("monitor/_positions.html", request, {"positions": positions})

    except asyncio.TimeoutError:
        return _error_fragment("positions", "Таймаут при получении позиций")
    except Exception as e:
        return _error_fragment("positions", f"Не удалось получить позиции: {e!s}")

# ──────────────────────────────────────────────────────────────────────────────
# Partials — последние ордера
# ──────────────────────────────────────────────────────────────────────────────
def _orders_signature(orders: List[Any]) -> Dict[str, Any]:
    n = len(orders or [])
    sample: List[Tuple] = []
    keys = ("id", "order_id", "client_order_id", "symbol", "side", "type", "status", "price", "origQty", "executedQty", "updateTime", "transactTime", "created_at", "updated_at")
    for o in (orders or [])[:10]:
        row = []
        if isinstance(o, dict):
            for k in keys:
                row.append(o.get(k))
        else:
            for k in keys:
                row.append(getattr(o, k, None))
        sample.append(tuple(row))
    return {"n": n, "sample": sample}

@router.get("/partials/orders", response_class=HTMLResponse)
async def partial_orders(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    db=Depends(_get_db_session),
):
    if not _HAS_DB or crud_orders is None or db is None:
        # 204 имеет смысл только для HTMX, но при пустых данных можно сразу 200 с пустым блоком
        return _render_fragment("monitor/_orders.html", request, {"orders": [], "updated_at": None})

    try:
        orders = await _with_timeout(crud_orders.get_last_orders(db, limit=limit), ORDERS_TIMEOUT)  # type: ignore[arg-type]
        # 204, если список ордеров по сути не изменился
        if (r := _maybe_204(request, f"orders:{limit}", _orders_signature(orders))) is not None:
            return r
        ctx = {"orders": orders, "updated_at": pd.Timestamp.utcnow().isoformat()}
        return _render_fragment("monitor/_orders.html", request, ctx)
    except asyncio.TimeoutError:
        return _error_fragment("orders", "Таймаут при загрузке ордеров")
    except Exception as e:
        return _error_fragment("orders", f"Не удалось загрузить ордера: {e!s}")

# ──────────────────────────────────────────────────────────────────────────────
# Partials — метрики
# ──────────────────────────────────────────────────────────────────────────────
def _to_df_ohlcv(rows: List[Any]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["ts", "close"])
    data = []
    for r in rows:
        try:
            ts = int(getattr(r, "ts", None) or getattr(r, "timestamp", None) or r["ts"])
            close = float(getattr(r, "close", None) or r["close"])
            data.append({"ts": ts, "close": close})
        except Exception:
            continue
    if not data:
        return pd.DataFrame(columns=["ts", "close"])
    return (
        pd.DataFrame(data)
        .dropna()
        .drop_duplicates(subset=["ts"])
        .sort_values("ts")
    )

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

def _metrics_signature(metrics: Dict[str, Optional[float]], symbol: str, tf: str) -> Dict[str, Any]:
    return {
        "symbol": symbol,
        "tf": tf,
        "sharpe": None if metrics.get("sharpe") is None else round(float(metrics["sharpe"]), 6),
        "vol": None if metrics.get("vol") is None else round(float(metrics["vol"]), 6),
    }

@router.get("/partials/metrics", response_class=HTMLResponse)
async def partial_metrics(
    request: Request,
    source: str = Query(DEF_MET_SOURCE),
    symbol: str = Query(DEF_MET_SYMBOL),
    tf: str = Query(DEF_MET_TF),
    limit: int = Query(3000, ge=50, le=20000),
    window_days: int = Query(30, ge=5, le=365),
    annualization_days: int = Query(365, ge=200, le=365),
    risk_free_annual: float = Query(0.0, ge=0.0, le=0.2),
    sharpe_warn: float = Query(0.5),
    sharpe_crit: float = Query(0.0),
    vol_warn: float = Query(0.8),
    vol_crit: float = Query(1.2),
    db=Depends(_get_db_session),
):
    calculated_at = pd.Timestamp.utcnow().isoformat()

    if not _HAS_DB or crud is None or db is None:
        metrics = {"sharpe": None, "vol": None}
        # 204 на пустых метриках не критичен — рендерим статично
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
            "calculated_at": calculated_at,
            "error": None,
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

        # 204, если метрики не изменились
        if (r := _maybe_204(request, f"metrics:{symbol}:{tf}", _metrics_signature(metrics, symbol, tf))) is not None:
            return r

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
            "calculated_at": calculated_at,
            "error": None,
        }
        return _render_fragment("monitor/_metrics.html", request, ctx)

    except asyncio.TimeoutError:
        return _error_fragment("metrics", "Таймаут при расчёте метрик")
    except Exception as e:
        return _error_fragment("metrics", f"Не удалось вычислить метрики: {e!s}")


@router.get("/partials/intel", response_class=HTMLResponse)
async def partial_intel(
    request: Request,
    mode: Literal["binance", "sim"] = Query(DEFAULT_MODE),
    testnet: bool = Query(DEFAULT_TESTNET),
    db=Depends(_get_db_session),
):
    if not _HAS_DB or db is None:
        ctx = {
            "pnl": {"trades": 0, "filled": 0, "rejected": 0, "volume": 0.0, "timeline": [], "total_pnl": 0.0},
            "risk": load_risk_config().to_dict(),
            "sentiment": None,
            "news": [],
            "error": "База данных недоступна",
        }
        return _render_fragment("monitor/_intel.html", request, ctx)

    try:
        dashboard_state = await _with_timeout(build_dashboard_state(db), INTEL_TIMEOUT)
        news_rows = []
        if crud_news is not None:
            news_rows = await _with_timeout(crud_news.latest_news(db, limit=8), INTEL_TIMEOUT)  # type: ignore[arg-type]
        sentiment_ctx = None
        if _HAS_NEWS_INTEL and load_latest_sentiment is not None:
            sentiment_ctx = await _with_timeout(load_latest_sentiment(), INTEL_TIMEOUT)

        ctx = {
            "pnl": dashboard_state.get("pnl", {}),
            "risk": dashboard_state.get("risk", {}),
            "sentiment": _sentiment_to_dict(sentiment_ctx),
            "news": [_news_item_to_dict(item) for item in news_rows],
            "error": None,
            "mode": mode,
            "testnet": testnet,
        }
        return _render_fragment("monitor/_intel.html", request, ctx)
    except asyncio.TimeoutError:
        return _error_fragment("intel", "Таймаут при загрузке панели интеллекта")
    except Exception as e:
        return _error_fragment("intel", f"Не удалось построить панель: {e!s}")

# ──────────────────────────────────────────────────────────────────────────────
# Reports
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/reports/orders.csv")
async def export_orders_csv(
    limit: int = Query(500, ge=10, le=2000),
    db=Depends(_get_db_session),
):
    if not _HAS_DB or db is None or OrderLog is None:
        raise HTTPException(status_code=503, detail="База данных недоступна")
    stmt = select(OrderLog).order_by(OrderLog.created_at.desc()).limit(limit)
    rows = list(await db.scalars(stmt))

    headers = [
        "id",
        "created_at",
        "exchange",
        "testnet",
        "symbol",
        "side",
        "type",
        "status",
        "price",
        "qty",
        "quote_qty",
        "cummulative_quote_qty",
        "commission",
        "commission_asset",
    ]

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(headers)
    for row in rows:
        data = row.as_dict() if hasattr(row, "as_dict") else {}
        writer.writerow([
            data.get("id"),
            data.get("created_at"),
            data.get("exchange"),
            data.get("testnet"),
            data.get("symbol"),
            data.get("side"),
            data.get("type"),
            data.get("status"),
            data.get("price"),
            data.get("qty"),
            data.get("quote_qty"),
            data.get("cummulative_quote_qty"),
            data.get("commission"),
            data.get("commission_asset"),
        ])

    content = buf.getvalue()
    return Response(
        content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="orders.csv"'},
    )

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

# ──────────────────────────────────────────────────────────────────────────────
# Back-compat aliases (короткие пути без /partials/*)
# ──────────────────────────────────────────────────────────────────────────────
@router.get("/balance", response_class=HTMLResponse)
async def balance_alias(
    request: Request,
    mode: Literal["binance", "sim"] = Query(DEFAULT_MODE),
    testnet: bool = Query(DEFAULT_TESTNET),
):
    return await partial_balance(request, mode=mode, testnet=testnet)

@router.get("/positions", response_class=HTMLResponse)
async def positions_alias(
    request: Request,
    mode: Literal["binance", "sim"] = Query(DEFAULT_MODE),
    testnet: bool = Query(DEFAULT_TESTNET),
):
    return await partial_positions(request, mode=mode, testnet=testnet)

@router.get("/orders", response_class=HTMLResponse)
async def orders_alias(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    db=Depends(_get_db_session),
):
    return await partial_orders(request, limit=limit, db=db)

@router.get("/metrics", response_class=HTMLResponse)
async def metrics_alias(
    request: Request,
    source: str = Query(DEF_MET_SOURCE),
    symbol: str = Query(DEF_MET_SYMBOL),
    tf: str = Query(DEF_MET_TF),
    limit: int = Query(3000, ge=50, le=20000),
    window_days: int = Query(30, ge=5, le=365),
    annualization_days: int = Query(365, ge=200, le=365),
    risk_free_annual: float = Query(0.0, ge=0.0, le=0.2),
    sharpe_warn: float = Query(0.5),
    sharpe_crit: float = Query(0.0),
    vol_warn: float = Query(0.8),
    vol_crit: float = Query(1.2),
    db=Depends(_get_db_session),
):
    return await partial_metrics(
        request,
        source=source,
        symbol=symbol,
        tf=tf,
        limit=limit,
        window_days=window_days,
        annualization_days=annualization_days,
        risk_free_annual=risk_free_annual,
        sharpe_warn=sharpe_warn,
        sharpe_crit=sharpe_crit,
        vol_warn=vol_warn,
        vol_crit=vol_crit,
        db=db,
    )
def _news_item_to_dict(item: Any) -> Dict[str, Any]:
    published = None
    if getattr(item, "published_at", None):
        try:
            published = item.published_at.isoformat()
        except Exception:
            published = str(item.published_at)
    return {
        "title": getattr(item, "title", ""),
        "summary": (getattr(item, "summary", "") or "")[:240],
        "impact": getattr(item, "impact", "low"),
        "sentiment": getattr(item, "sentiment", 0),
        "importance": getattr(item, "importance", "low"),
        "link": getattr(item, "link", None),
        "source": getattr(item, "source", None),
        "published": published,
    }


def _sentiment_to_dict(ctx: Any) -> Optional[Dict[str, Any]]:
    if ctx is None:
        return None
    if SentimentContext is not None and isinstance(ctx, SentimentContext):
        news_score = ctx.news_score
        social_score = ctx.social_score
        fear_greed = ctx.fear_greed
        composite = ctx.composite_score
        methodology = getattr(ctx, "methodology", "v1")
    else:
        news_score = float(getattr(ctx, "news_score", 0.0))
        social_score = float(getattr(ctx, "social_score", 0.0))
        fear_greed = float(getattr(ctx, "fear_greed", 50.0))
        composite = float(getattr(ctx, "composite_score", 0.0))
        methodology = getattr(ctx, "methodology", "v1")
    return {
        "news_score": round(news_score, 3),
        "social_score": round(social_score, 3),
        "fear_greed": round(fear_greed, 1),
        "composite": round(composite, 3),
        "methodology": methodology,
    }
