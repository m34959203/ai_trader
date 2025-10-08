# executors/api_binance.py
from __future__ import annotations

import hmac
import json
import time
import hashlib
import logging
import random
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Literal, Tuple
from urllib.parse import urlencode
from decimal import Decimal, getcontext

import httpx

# Повышенная точность Decimal для корректной работы с фильтрами биржи
getcontext().prec = 40

# Проектные ключи (ENV/.env), бросает RuntimeError с понятным текстом при отсутствии
try:
    from utils.secrets import get_binance_keys
except Exception:
    # безопасный фолбэк, если модуль ещё не добавлен
    def get_binance_keys(*, testnet: bool = True) -> tuple[str, str]:
        api_key = os.getenv("BINANCE_TESTNET_API_KEY" if testnet else "BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_TESTNET_API_SECRET" if testnet else "BINANCE_API_SECRET", "")
        if not api_key or not api_secret:
            raise RuntimeError("Binance API keys not configured in environment")
        return api_key, api_secret


LOG = logging.getLogger("executors.api_binance")

BINANCE_DOMAIN_MAIN = "https://api.binance.com"
BINANCE_DOMAIN_TEST = "https://testnet.binance.vision"

# Все пути внизу вызываются с префиксом "/api/v3/..." (base_url = DOMAIN без /api)
API_PREFIX = "/api/v3"

# ──────────────────────────────────────────────────────────────────────────────
# Базовый протокол исполнителя
# ──────────────────────────────────────────────────────────────────────────────
class Executor:
    name: str

    async def open_order(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    async def close_order(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    async def list_positions(self, **kwargs) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def fetch_balance(self) -> Dict[str, Any]:
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Ошибка Binance
# ──────────────────────────────────────────────────────────────────────────────
class BinanceAPIError(RuntimeError):
    def __init__(
        self,
        status_code: int,
        code: Optional[int],
        msg: str,
        payload: Any = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.status_code = status_code
        self.code = code
        self.msg = msg
        self.payload = payload
        self.headers = headers or {}
        super().__init__(f"Binance API error {status_code} (code={code}): {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# Низкоуровневый Spot/Testnet клиент
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class _Creds:
    api_key: str
    api_secret: str


class _BinanceSpotClient:
    """
    Лёгкий клиент для Binance Spot/Testnet с ретраями и кэшами публичных методов.

    Private:
      - POST   /api/v3/order
      - DELETE /api/v3/order
      - DELETE /api/v3/openOrders        (cancel all open orders on a symbol)
      - POST   /api/v3/order/oco
      - GET    /api/v3/account
      - GET    /api/v3/openOrders
      - GET    /api/v3/order             (get order)

    Public:
      - GET    /api/v3/ping
      - GET    /api/v3/time
      - GET    /api/v3/exchangeInfo
      - GET    /api/v3/ticker/price
    """

    def __init__(
        self,
        creds: _Creds,
        base_domain: str,
        recv_window: int = 5000,
        timeout: float = 20.0,
        max_retries: int = 5,
        backoff_base: float = 0.5,
        backoff_cap: float = 8.0,
        user_agent: str = "AI-Trader/1.0 (+binance-spot)",
        trust_env: bool = True,
    ):
        self.creds = creds
        # domain без /api; пути будут начинаться с /api/v3/...
        self.base_url = base_domain.rstrip("/")
        self.recv_window = max(1, int(recv_window))
        self.max_retries = max(0, int(max_retries))
        self.backoff_base = float(backoff_base)
        self.backoff_cap = float(backoff_cap)

        # Безопасный таймаут (0.0 заменяем на дефолт)
        _t = float(timeout) if float(timeout) > 0 else 20.0
        # Раздельные таймауты и лимиты пула, чтобы не залипать на connect/pool
        self._timeout = httpx.Timeout(
            connect=min(10.0, _t),
            read=_t,
            write=min(10.0, _t),
            pool=min(10.0, _t),
        )
        self._limits = httpx.Limits(
            max_keepalive_connections=10,
            max_connections=20,
            keepalive_expiry=30.0,
        )
        self._headers_default = {"User-Agent": user_agent}

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self._timeout,
            limits=self._limits,
            headers=self._headers_default,
            http2=False,
            trust_env=trust_env,
        )

        # serverTime - local_time_ms drift
        self._time_offset_ms: int = 0
        self._last_sync_ts: float = 0.0
        self._sync_interval_sec: float = 60.0

        # Public caches
        self._ex_info_cache: Dict[str, Any] = {}
        self._ex_info_cache_ts: float = 0.0
        self._ex_info_ttl: float = 6 * 60 * 60  # 6h

        self._ticker_cache: Dict[str, Tuple[Decimal, float]] = {}  # symbol -> (price, ts)
        self._ticker_ttl: float = 5.0  # 5s

        # Батч-кэш всех тикеров
        self._all_tickers_cache: Dict[str, Tuple[Decimal, float]] = {}
        self._all_tickers_ttl: float = 2.0

    async def close(self) -> None:
        try:
            await self._client.aclose()
        except Exception:
            pass

    def _recreate_client(self) -> None:
        try:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self._timeout,
                limits=self._limits,
                headers=self._headers_default,
                http2=False,
                trust_env=True,
            )
        except Exception as e:
            LOG.warning("Failed to recreate httpx client: %r", e)

    # ---- классификация ошибок ----
    @staticmethod
    def _is_rate_limit(err: Exception, code: Optional[int], status_code: Optional[int]) -> bool:
        if isinstance(err, BinanceAPIError):
            c = code or err.code or 0
            sc = status_code or err.status_code or 0
            return sc == 429 or c in (-1003, -1015)
        return False

    @staticmethod
    def _is_invalid_timestamp(err: Exception, code: Optional[int]) -> bool:
        if isinstance(err, BinanceAPIError):
            c = code or err.code or 0
            if c == -1021:
                return True
            msg = (err.msg or "").lower()
            return "timestamp" in msg
        return False

    @staticmethod
    def _is_network_error(err: Exception) -> bool:
        return isinstance(err, httpx.HTTPError)

    # ---- синхронизация времени ----
    async def _sync_server_time(self) -> None:
        now = time.time()
        if (now - self._last_sync_ts) < self._sync_interval_sec:
            return
        try:
            resp = await self._client.get(f"{API_PREFIX}/time")
            resp.raise_for_status()
            data = resp.json()
            server_time = int(data.get("serverTime"))
            local = int(time.time() * 1000)
            self._time_offset_ms = server_time - local
            self._last_sync_ts = now
            LOG.debug("Binance time synced, offset_ms=%d", self._time_offset_ms)
        except Exception as e:
            LOG.warning("Binance time sync failed: %r", e)

    # ---- подпись ----
    def _ts_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        p = dict(params or {})
        local_ms = int(time.time() * 1000)
        p.setdefault("timestamp", local_ms + self._time_offset_ms)
        p.setdefault("recvWindow", self.recv_window)
        return p

    def _sign(self, params: Dict[str, Any]) -> str:
        q = urlencode(params, doseq=True)
        return hmac.new(self.creds.api_secret.encode("utf-8"), q.encode("utf-8"), hashlib.sha256).hexdigest()

    @staticmethod
    def _parse_error(status_code: int, text: str, headers: Optional[Dict[str, str]] = None) -> BinanceAPIError:
        try:
            data = json.loads(text)
            return BinanceAPIError(status_code, data.get("code"), data.get("msg") or text, payload=data, headers=headers)
        except Exception:
            return BinanceAPIError(status_code, None, text, headers=headers)

    async def _req_once(
        self,
        method: Literal["GET", "POST", "DELETE"],
        path: str,
        params: Optional[Dict[str, Any]] = None,
        sign: bool = True,
    ) -> Dict[str, Any]:
        headers = {"X-MBX-APIKEY": self.creds.api_key} if sign else None

        t0 = time.perf_counter()
        try:
            if method == "GET":
                qp = dict(params or {})
                if sign:
                    qp = self._ts_params(qp)
                    qp["signature"] = self._sign(qp)
                resp = await self._client.get(path, params=qp, headers=headers)
            elif method == "POST":
                body = dict(params or {})
                if sign:
                    body = self._ts_params(body)
                    body["signature"] = self._sign(body)
                resp = await self._client.post(path, data=body, headers=headers)
            elif method == "DELETE":
                qp = dict(params or {})
                if sign:
                    qp = self._ts_params(qp)
                    qp["signature"] = self._sign(qp)
                resp = await self._client.delete(path, params=qp, headers=headers)
            else:
                raise ValueError("Unsupported HTTP method")
        finally:
            dt = time.perf_counter() - t0
            LOG.debug("Binance %s %s took %.3fs", method, path, dt)

        if resp.status_code >= 400:
            raise self._parse_error(resp.status_code, resp.text, headers=dict(resp.headers))

        try:
            return resp.json()
        except Exception:
            return {"raw": resp.text}

    async def _request_with_retries(
        self,
        method: Literal["GET", "POST", "DELETE"],
        path: str,
        params: Optional[Dict[str, Any]] = None,
        sign: bool = True,
    ) -> Dict[str, Any]:
        """
        Политика ретраев:
          - (-1021) INVALID_TIMESTAMP → ресинк времени и повтор
          - 429/-1003/-1015           → экспоненциальный бэкофф (+ Retry-After)
          - сетевые ошибки            → экспоненциальный бэкофф
        """
        await self._sync_server_time()

        attempt = 0
        while True:
            attempt += 1
            try:
                return await self._req_once(method, path, params=params, sign=sign)

            except BinanceAPIError as e:
                code = e.code or 0
                status = e.status_code or 0

                if self._is_invalid_timestamp(e, code) and attempt <= self.max_retries:
                    LOG.warning("(-1021) INVALID_TIMESTAMP: resync & retry (attempt=%d)", attempt)
                    self._last_sync_ts = 0.0
                    await self._sync_server_time()
                    continue

                if self._is_rate_limit(e, code, status) and attempt <= self.max_retries:
                    retry_after = 0.0
                    try:
                        ra = e.headers.get("Retry-After") if isinstance(e.headers, dict) else None
                        if ra:
                            retry_after = float(ra)
                    except Exception:
                        retry_after = 0.0

                    backoff = min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_cap)
                    sleep_s = max(retry_after, backoff) * (1.0 + random.uniform(0, 0.15))
                    LOG.warning("Rate limit (%s/%s), sleep %.2fs (attempt=%d)", status, code, sleep_s, attempt)
                    await _sleep_async(sleep_s)
                    continue

                LOG.error("Binance API error (no-retry): %s", e)
                raise

            except httpx.HTTPError as e:
                if attempt <= self.max_retries and self._is_network_error(e):
                    backoff = min(self.backoff_base * (2 ** (attempt - 1)), self.backoff_cap)
                    sleep_s = backoff * (1.0 + random.uniform(0, 0.25))
                    LOG.warning("Network error %r, retry in %.2fs (attempt=%d)", e, sleep_s, attempt)
                    if attempt >= 2:
                        self._recreate_client()
                    await _sleep_async(sleep_s)
                    continue
                raise

    # -------- Публичные утилиты (с кэшом) --------
    async def ping(self) -> None:
        await self._request_with_retries("GET", f"{API_PREFIX}/ping", params={}, sign=False)

    async def get_exchange_info(self) -> Dict[str, Any]:
        now = time.time()
        if self._ex_info_cache and (now - self._ex_info_cache_ts) < self._ex_info_ttl:
            return self._ex_info_cache
        data = await self._request_with_retries("GET", f"{API_PREFIX}/exchangeInfo", params={}, sign=False)
        self._ex_info_cache = data or {}
        self._ex_info_cache_ts = now
        return data

    async def get_all_tickers(self) -> Dict[str, Decimal]:
        """
        Батч-метод: вернуть все цены (symbol->Decimal) с коротким TTL.
        """
        now = time.time()
        if self._all_tickers_cache:
            _, ts = next(iter(self._all_tickers_cache.values()))
            if (now - ts) < self._all_tickers_ttl:
                return {k: v for k, (v, _) in self._all_tickers_cache.items()}

        data = await self._request_with_retries("GET", f"{API_PREFIX}/ticker/price", params={}, sign=False)
        prices: Dict[str, Decimal] = {}
        for t in data or []:
            try:
                s = str(t.get("symbol")).upper()
                p = Decimal(str(t.get("price")))
                prices[s] = p
            except Exception:
                continue

        self._all_tickers_cache = {k: (v, now) for k, v in prices.items()}
        return prices

    async def get_all_tickers_list(self) -> List[Dict[str, Any]]:
        """
        Альтернатива: список словарей {'symbol','price'} — удобно для потребителей,
        которые ожидают list-формат.
        """
        d = await self.get_all_tickers()
        return [{"symbol": k, "price": float(v)} for k, v in d.items()]

    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        info = await self.get_exchange_info()
        for s in info.get("symbols", []):
            if s.get("symbol") == symbol.upper():
                return s
        return None

    async def get_ticker_price(self, symbol: str) -> Optional[Decimal]:
        """
        Возвращает цену символа с приоритетом:
        1) батч-кэш get_all_tickers,
        2) локальный кэш конкретного символа,
        3) сетевой запрос.
        """
        now = time.time()
        sym = symbol.upper()

        # 1) батч-кэш
        if sym in self._all_tickers_cache:
            price, ts = self._all_tickers_cache[sym]
            if (now - ts) < self._all_tickers_ttl:
                return price

        # 2) локальный кэш
        if sym in self._ticker_cache:
            price, ts = self._ticker_cache[sym]
            if (now - ts) < self._ticker_ttl:
                return price

        # 3) сеть
        data = await self._request_with_retries("GET", f"{API_PREFIX}/ticker/price", params={"symbol": sym}, sign=False)
        p = data.get("price")
        try:
            price = Decimal(str(p))
            self._ticker_cache[sym] = (price, now)
            return price
        except Exception:
            return None

    async def get_time(self) -> Dict[str, Any]:
        return await self._request_with_retries("GET", f"{API_PREFIX}/time", params={}, sign=False)

    async def get_order(self, symbol: str, orderId: Optional[int] = None, origClientOrderId: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {"symbol": symbol.upper()}
        if orderId is not None:
            params["orderId"] = orderId
        if origClientOrderId is not None:
            params["origClientOrderId"] = origClientOrderId
        return await self._request_with_retries("GET", f"{API_PREFIX}/order", params=params, sign=True)

    # ---- ордера/баланс ----
    async def post_order(
        self,
        *,
        symbol: str,
        side: Literal["BUY", "SELL"],
        type: Literal["MARKET", "LIMIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"],
        quantity: Optional[float] = None,
        quoteOrderQty: Optional[float] = None,
        price: Optional[float] = None,
        stopPrice: Optional[float] = None,
        timeInForce: Optional[str] = None,
        newClientOrderId: Optional[str] = None,
        newOrderRespType: Optional[str] = None,  # "ACK" | "RESULT" | "FULL"
        test: bool = False,  # /order/test
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"symbol": symbol.upper(), "side": side.upper(), "type": type.upper()}
        if quantity is not None:
            params["quantity"] = _to_str(quantity)
        if quoteOrderQty is not None:
            params["quoteOrderQty"] = _to_str(quoteOrderQty)
        if price is not None:
            params["price"] = _to_str(price)
        if stopPrice is not None:
            params["stopPrice"] = _to_str(stopPrice)
        if timeInForce:
            params["timeInForce"] = timeInForce
        if newClientOrderId:
            params["newClientOrderId"] = newClientOrderId
        if newOrderRespType:
            params["newOrderRespType"] = newOrderRespType

        path = f"{API_PREFIX}/order/test" if test else f"{API_PREFIX}/order"
        return await self._request_with_retries("POST", path, params=params, sign=True)

    async def post_oco_order(
        self,
        *,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: float,
        price: float,
        stopPrice: float,
        stopLimitPrice: float,
        stopLimitTimeInForce: str = "GTC",
        listClientOrderId: Optional[str] = None,
        limitClientOrderId: Optional[str] = None,
        stopClientOrderId: Optional[str] = None,
        newOrderRespType: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "quantity": _to_str(quantity),
            "price": _to_str(price),
            "stopPrice": _to_str(stopPrice),
            "stopLimitPrice": _to_str(stopLimitPrice),
            "stopLimitTimeInForce": stopLimitTimeInForce,
        }
        if listClientOrderId:
            params["listClientOrderId"] = listClientOrderId
        if limitClientOrderId:
            params["limitClientOrderId"] = limitClientOrderId
        if stopClientOrderId:
            params["stopClientOrderId"] = stopClientOrderId
        if newOrderRespType:
            params["newOrderRespType"] = newOrderRespType

        return await self._request_with_retries("POST", f"{API_PREFIX}/order/oco", params=params, sign=True)

    async def delete_order(
        self,
        *,
        symbol: str,
        orderId: Optional[int] = None,
        origClientOrderId: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"symbol": symbol.upper()}
        if orderId is not None:
            params["orderId"] = orderId
        if origClientOrderId is not None:
            params["origClientOrderId"] = origClientOrderId
        return await self._request_with_retries("DELETE", f"{API_PREFIX}/order", params=params, sign=True)

    async def delete_open_orders(self, *, symbol: str) -> Dict[str, Any]:
        """
        Отмена всех открытых ордеров по символу (DELETE /api/v3/openOrders).
        """
        params: Dict[str, Any] = {"symbol": symbol.upper()}
        return await self._request_with_retries("DELETE", f"{API_PREFIX}/openOrders", params=params, sign=True)

    async def get_account(self) -> Dict[str, Any]:
        return await self._request_with_retries("GET", f"{API_PREFIX}/account", params={}, sign=True)

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        if symbol:
            params["symbol"] = symbol.upper()
        data = await self._request_with_retries("GET", f"{API_PREFIX}/openOrders", params=params, sign=True)
        return data if isinstance(data, list) else []


# ──────────────────────────────────────────────────────────────────────
# Вспомогательная асинхронная пауза
# ──────────────────────────────────────────────────────────────────────
async def _sleep_async(seconds: float) -> None:
    import asyncio
    await asyncio.sleep(seconds)


def _to_str(x: float | int | str | Decimal) -> str:
    if isinstance(x, str):
        return x
    d = Decimal(str(x))
    # normalize без экспоненты
    return format(d.normalize(), "f")


def _dec(x: float | int | str | Decimal | None) -> Decimal:
    return Decimal(str(x)) if x is not None else Decimal("0")


def _floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    if step <= 0:
        return value
    return (value // step) * step


def _clamp(value: Decimal, min_v: Optional[Decimal], max_v: Optional[Decimal]) -> Decimal:
    if min_v is not None and value < min_v:
        return min_v
    if max_v is not None and value > max_v:
        return max_v
    return value


def _short_client_id(prefix: str = "cli") -> str:
    """
    Binance ограничивает ClientOrderId до 36 символов.
    Формат: <prefix>-<ms>-<hex> (влезает в лимит).
    """
    ms = int(time.time() * 1000)
    return f"{prefix}-{ms}-{os.urandom(4).hex()}"[:36]


# ──────────────────────────────────────────────────────────────────────────────
# Исполнитель для проекта
# ──────────────────────────────────────────────────────────────────────────────
class BinanceExecutor(Executor):
    """
    Исполнитель спотовых ордеров.

    testnet=True  -> https://testnet.binance.vision
    testnet=False -> https://api.binance.com

    Совместим со спринтом 4+:
      - open_order / close_order / list_positions / fetch_balance
      - защитные приказы (SL/TP/OCO) через open_with_protection
      - delete_open_orders(symbol=...), get_order(...)
    """

    name = "binance"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        *,
        testnet: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.testnet = testnet
        self.config = config or {}

        # Ключи из ENV, если явно не передали
        if not api_key or not api_secret:
            api_key, api_secret = get_binance_keys(testnet=self.testnet)

        self.creds = _Creds(api_key=api_key, api_secret=api_secret)

        base_domain = self.config.get("base_url") or (BINANCE_DOMAIN_TEST if testnet else BINANCE_DOMAIN_MAIN)
        recv_window = int(self.config.get("recv_window", 5000))
        timeout_cfg = float(self.config.get("timeout", 20.0)) or 20.0
        max_retries = int(self.config.get("max_retries", 5))
        backoff_base = float(self.config.get("backoff_base", 0.5))
        backoff_cap = float(self.config.get("backoff_cap", 8.0))
        user_agent = str(self.config.get("user_agent", "AI-Trader/1.0 (+binance-spot)"))

        self.client = _BinanceSpotClient(
            self.creds,
            base_domain=base_domain,
            recv_window=recv_window,
            timeout=timeout_cfg,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_cap=backoff_cap,
            user_agent=user_agent,
            trust_env=True,
        )

        # Опциональный локальный регистр позиций; для прод полагаться на reconcile/балансы
        self._track_local_pos: bool = bool(self.config.get("track_local_pos", False))
        self._pos: Dict[str, float] = {}

    async def close(self) -> None:
        await self.client.close()

    # ---------- Применение фильтров символа ----------
    async def _apply_symbol_rules(
        self,
        *,
        symbol: str,
        type_u: Literal["MARKET", "LIMIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"],
        side_u: Literal["BUY", "SELL"],
        qty: Optional[float],
        price: Optional[float],
        quote_qty: Optional[float],
        stop_price: Optional[float] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """
        Возвращает (qty, price, quote_qty, stop_price) приведённые к фильтрам Binance.
        """
        sym = symbol.upper()
        s_info = await self.client.get_symbol_info(sym)
        if not s_info:
            return qty, price, quote_qty, stop_price

        # filters
        price_filter = next((f for f in s_info.get("filters", []) if f.get("filterType") == "PRICE_FILTER"), None)
        lot_filter = next((f for f in s_info.get("filters", []) if f.get("filterType") == "LOT_SIZE"), None)
        mlot_filter = next((f for f in s_info.get("filters", []) if f.get("filterType") == "MARKET_LOT_SIZE"), None)
        notional_filter = next((f for f in s_info.get("filters", []) if f.get("filterType") in ("MIN_NOTIONAL", "NOTIONAL")), None)

        # steps & thresholds
        tick_size = _dec(price_filter["tickSize"]) if price_filter else None
        min_price = _dec(price_filter.get("minPrice")) if price_filter and price_filter.get("minPrice") else None
        max_price = _dec(price_filter.get("maxPrice")) if price_filter and price_filter.get("maxPrice") else None

        step_size = _dec(lot_filter["stepSize"]) if lot_filter else None
        min_qty = _dec(lot_filter["minQty"]) if lot_filter else None
        max_qty = _dec(lot_filter["maxQty"]) if lot_filter else None

        m_step_size = _dec(mlot_filter["stepSize"]) if mlot_filter else None
        m_min_qty = _dec(mlot_filter["minQty"]) if mlot_filter else None
        m_max_qty = _dec(mlot_filter["maxQty"]) if mlot_filter else None

        min_notional = _dec(notional_filter.get("minNotional")) if notional_filter else None

        # LIMIT-like
        if type_u in ("LIMIT", "STOP_LOSS_LIMIT", "TAKE_PROFIT_LIMIT"):
            d_price = _dec(price) if price is not None else None
            d_stop = _dec(stop_price) if stop_price is not None else None
            d_qty = _dec(qty) if qty is not None else None

            if d_price is not None and tick_size and tick_size > 0:
                d_price = _floor_to_step(d_price, tick_size)
                d_price = _clamp(d_price, min_price, max_price)
            if d_stop is not None and tick_size and tick_size > 0:
                d_stop = _floor_to_step(d_stop, tick_size)
                d_stop = _clamp(d_stop, min_price, max_price)
            if d_qty is not None:
                step = step_size if step_size and step_size > 0 else None
                if step:
                    d_qty = _floor_to_step(d_qty, step)
                if min_qty and d_qty < min_qty:
                    d_qty = min_qty
                if max_qty and d_qty > max_qty:
                    d_qty = max_qty

            if min_notional and d_price is not None and d_qty is not None and d_price > 0:
                notion = d_price * d_qty
                if notion < min_notional:
                    need_qty = (min_notional / d_price)
                    if step_size and step_size > 0:
                        need_qty = _floor_to_step(need_qty, step_size)
                    if min_qty and need_qty < min_qty:
                        need_qty = min_qty
                    d_qty = need_qty

            return (
                float(d_qty) if d_qty is not None else None,
                float(d_price) if d_price is not None else None,
                quote_qty,
                float(d_stop) if d_stop is not None else None,
            )

        # MARKET
        if type_u == "MARKET":
            if qty is not None:
                d_qty = _dec(qty)
                step = m_step_size or step_size
                qmin = m_min_qty or min_qty
                qmax = m_max_qty or max_qty
                if step and step > 0:
                    d_qty = _floor_to_step(d_qty, step)
                if qmin and d_qty < qmin:
                    d_qty = qmin
                if qmax and d_qty > qmax:
                    d_qty = qmax
                if min_notional:
                    last = await self.client.get_ticker_price(sym)
                    if last and last > 0:
                        notion = d_qty * last
                        if notion < min_notional:
                            need_qty = (min_notional / last)
                            if step and step > 0:
                                need_qty = _floor_to_step(need_qty, step)
                            if qmin and need_qty < qmin:
                                need_qty = qmin
                            d_qty = need_qty
                return float(d_qty), None, None, None
            # MARKET by quote — биржа сама валидирует
            return None, None, quote_qty, None

        return qty, price, quote_qty, stop_price

    # ---------- Утилиты для роутера ----------
    async def round_qty(self, symbol: str, qty: float) -> float:
        adj_qty, _, _, _ = await self._apply_symbol_rules(
            symbol=symbol, type_u="MARKET", side_u="BUY", qty=qty, price=None, quote_qty=None
        )
        return float(adj_qty or qty)

    async def get_price(self, symbol: str) -> Optional[float]:
        p = await self.client.get_ticker_price(symbol)
        return float(p) if p is not None else None

    async def price(self, symbol: str) -> Dict[str, Any]:
        """
        Совместимость с вызывающим кодом, который ожидает метод `price(...)`
        и умеет разбирать словарь c ключом 'price'.
        """
        p = await self.get_price(symbol)
        return {"symbol": symbol.upper(), "price": p} if p is not None else {"symbol": symbol.upper(), "price": None}

    async def get_ticker_price(self, symbol: str) -> Optional[float]:
        return await self.get_price(symbol)

    async def get_last_price(self, symbol: str) -> Optional[float]:
        return await self.get_price(symbol)

    async def get_all_tickers(self) -> Dict[str, float]:
        """
        Проксируем батч-цены наружу (float), чтобы потребители могли использовать executor.get_all_tickers().
        """
        d = await self.client.get_all_tickers()
        return {k: float(v) for k, v in d.items()}

    # ---------- helpers ----------
    @staticmethod
    def _avg_fill_price(resp: Dict[str, Any]) -> Optional[float]:
        """
        Пытается получить среднюю цену исполнения из FULL-ответа Binance.
        Порядок:
          1) fills: sum(p*q)/sum(q)
          2) cummulativeQuoteQty / executedQty
          3) price (как fallback)
        """
        try:
            fills = resp.get("fills") or []
            if fills:
                total_q = 0.0
                total_quote = 0.0
                for f in fills:
                    p = float(f.get("price") or 0)
                    q = float(f.get("qty") or 0)
                    total_q += q
                    total_quote += p * q
                if total_q > 0:
                    return total_quote / total_q
            ex_q = float(resp.get("executedQty") or 0)
            cq = float(resp.get("cummulativeQuoteQty") or 0)
            if ex_q > 0 and cq > 0:
                return cq / ex_q
            p = resp.get("price")
            return float(p) if p is not None else None
        except Exception:
            return None

    def _bump_local_pos(self, symbol: str, side_u: str, filled_qty: float) -> None:
        if not self._track_local_pos:
            return
        if filled_qty <= 0:
            return
        s = symbol.upper()
        self._pos[s] = self._pos.get(s, 0.0) + (filled_qty if side_u == "BUY" else -filled_qty)

    # ---------- API ----------
    async def open_order(
        self,
        *,
        symbol: str,
        side: Literal["buy", "sell"],
        type: Literal["market", "limit"] = "market",
        qty: Optional[float] = None,
        price: Optional[float] = None,
        timeInForce: Optional[str] = None,
        client_order_id: Optional[str] = None,
        quote_qty: Optional[float] = None,
        test: bool = False,
    ) -> Dict[str, Any]:
        side_u = side.upper()
        type_u = type.upper()

        # Быстрая валидация
        if type_u == "LIMIT":
            if qty is None or price is None:
                raise ValueError("LIMIT order requires both qty and price")
        elif type_u == "MARKET":
            if qty is None and quote_qty is None:
                raise ValueError("MARKET order requires qty or quote_qty")
        else:
            raise ValueError("Unsupported order type")

        # По умолчанию TIF для LIMIT
        if type_u == "LIMIT" and not timeInForce:
            timeInForce = "GTC"

        # Приведение по правилам символа
        qty_adj, price_adj, quote_qty_adj, _ = await self._apply_symbol_rules(
            symbol=symbol,
            type_u=type_u,
            side_u=side_u,
            qty=qty,
            price=price,
            quote_qty=quote_qty,
        )

        # clientOrderId <= 36
        if not client_order_id:
            client_order_id = _short_client_id("cli")

        resp = await self.client.post_order(
            symbol=symbol,
            side=side_u,
            type=type_u,
            quantity=qty_adj,
            quoteOrderQty=quote_qty_adj,
            price=price_adj,
            timeInForce=timeInForce,
            newClientOrderId=client_order_id,
            newOrderRespType="FULL",  # чтобы получить fills/комиссии
            test=test,
        )

        # Если это тестовый ордер — Binance возвращает {} при успехе
        if test:
            return {
                "exchange": "binance",
                "testnet": self.testnet,
                "test_order": True,
                "symbol": symbol.upper(),
                "side": side.lower(),
                "type": type.lower(),
                "qty": float(qty_adj or 0.0),
                "price": float(price_adj or 0.0) if price_adj else None,
                "status": "TEST_OK",
                "raw": resp,
            }

        # расчёт исполнения
        status_u = (resp.get("status") or "").upper()
        try:
            filled_qty = float(resp.get("executedQty") or 0)
        except Exception:
            filled_qty = 0.0

        # локальный (опциональный) bump
        if status_u in ("FILLED", "PARTIALLY_FILLED") and filled_qty > 0.0:
            self._bump_local_pos(symbol, side_u, filled_qty)

        # цена для отчёта: limit → price_adj, market → avg_fill_price (или price из ответа)
        avg_px = self._avg_fill_price(resp)
        out_price = None
        if type_u == "LIMIT":
            out_price = float(resp.get("price") or (price_adj or 0.0) or 0.0)
        else:
            out_price = float(avg_px or 0.0)

        return {
            "exchange": "binance",
            "testnet": self.testnet,
            "order_id": str(resp.get("orderId") or ""),
            "client_order_id": str(resp.get("clientOrderId") or client_order_id),
            "symbol": symbol.upper(),
            "side": side.lower(),
            "type": type.lower(),
            "price": out_price,
            "qty": float(resp.get("origQty") or (qty_adj or 0.0) or 0.0),
            "executed_qty": float(resp.get("executedQty") or 0.0),
            "cummulative_quote": float(resp.get("cummulativeQuoteQty") or 0.0),
            "status": resp.get("status") or "NEW",
            "raw": resp,
        }

    async def open_with_protection(
        self,
        *,
        symbol: str,
        side: Literal["buy", "sell"],
        qty: Optional[float] = None,
        entry_type: Literal["market", "limit"] = "market",
        entry_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        quote_qty: Optional[float] = None,
        timeInForce: Optional[str] = None,
    ) -> Dict[str, Any]:
        entry = await self.open_order(
            symbol=symbol,
            side=side,
            type=entry_type,
            qty=qty,
            price=entry_price,
            timeInForce=timeInForce,
            client_order_id=client_order_id,
            quote_qty=quote_qty,
        )

        raw = entry.get("raw") or {}
        status = (raw.get("status") or entry.get("status") or "NEW").upper()
        try:
            filled_qty = float(raw.get("executedQty") or entry.get("executed_qty") or entry.get("qty") or 0)
        except Exception:
            filled_qty = float(entry.get("qty") or 0)

        # защита не нужна
        if sl_price is None and tp_price is None:
            return {**entry, "protection": None, "protection_ids": None}

        # для LIMIT ждём исполнения
        if entry_type.lower() == "limit" and filled_qty <= 0 and status not in ("FILLED", "PARTIALLY_FILLED"):
            return {
                **entry,
                "protection": None,
                "protection_ids": None,
                "protection_pending": True,
                "note": "Protection not placed: entry not filled yet (LIMIT).",
            }

        # фильтры
        sym = symbol.upper()
        s_info = await self.client.get_symbol_info(sym)
        pf = next((f for f in (s_info.get("filters", []) if s_info else []) if f["filterType"] == "PRICE_FILTER"), None)
        lf = next((f for f in (s_info.get("filters", []) if s_info else []) if f["filterType"] == "LOT_SIZE"), None)
        tick_size = _dec(pf["tickSize"]) if pf else Decimal("0.00000001")
        lot_step = _dec(lf["stepSize"]) if lf else Decimal("0.00000001")
        min_price = _dec(pf.get("minPrice")) if pf and pf.get("minPrice") else None
        max_price = _dec(pf.get("maxPrice")) if pf and pf.get("maxPrice") else None

        def _round_price(p: float | None) -> Optional[float]:
            if p is None:
                return None
            d = _dec(p)
            if tick_size > 0:
                d = _floor_to_step(d, tick_size)
                d = _clamp(d, min_price, max_price)
            return float(d)

        sl_p = _round_price(sl_price)
        tp_p = _round_price(tp_price)

        # объём защиты
        protect_qty = filled_qty if filled_qty > 0 else (entry.get("qty") or qty or 0.0)
        if protect_qty and lot_step > 0:
            protect_qty = float(_floor_to_step(_dec(protect_qty), lot_step))
        if protect_qty <= 0:
            return {**entry, "protection": None, "protection_ids": None, "note": "No filled qty to protect."}

        protect_side = "SELL" if side.lower() == "buy" else "BUY"

        protection_ids: Dict[str, Any] = {}
        protection_kind: str = "none"

        # sanity для OCO
        if sl_p is not None and tp_p is not None:
            if side.lower() == "buy" and not (tp_p > sl_p):
                raise ValueError("For long OCO: tp_price must be > sl_price")
            if side.lower() == "sell" and not (tp_p < sl_p):
                raise ValueError("For short OCO: tp_price must be < sl_price")

        try:
            # OCO
            if sl_p is not None and tp_p is not None:
                list_cid = _short_client_id("oco")
                limit_cid = f"{list_cid}-tp"[:36]
                stop_cid = f"{list_cid}-sl"[:36]
                oco = await self.client.post_oco_order(
                    symbol=sym,
                    side=protect_side,
                    quantity=protect_qty,
                    price=tp_p,
                    stopPrice=sl_p,
                    stopLimitPrice=sl_p,
                    stopLimitTimeInForce="GTC",
                    listClientOrderId=list_cid,
                    limitClientOrderId=limit_cid,
                    stopClientOrderId=stop_cid,
                    newOrderRespType="RESULT",
                )
                protection_kind = "oco"
                protection_ids = {
                    "orderListId": oco.get("orderListId"),
                    "listClientOrderId": oco.get("listClientOrderId"),
                    "orders": [
                        {"orderId": it.get("orderId"), "clientOrderId": it.get("clientOrderId")}
                        for it in (oco.get("orders") or [])
                    ],
                    "orderReports": [
                        {
                            "orderId": it.get("orderId"),
                            "clientOrderId": it.get("clientOrderId"),
                            "type": it.get("type"),
                            "side": it.get("side"),
                            "status": it.get("status"),
                        }
                        for it in (oco.get("orderReports") or [])
                    ],
                }

            # только SL
            elif sl_p is not None:
                qty_adj, price_adj, _, stop_adj = await self._apply_symbol_rules(
                    symbol=sym,
                    type_u="STOP_LOSS_LIMIT",
                    side_u=protect_side,
                    qty=protect_qty,
                    price=sl_p,
                    quote_qty=None,
                    stop_price=sl_p,
                )
                sl_resp = await self.client.post_order(
                    symbol=sym,
                    side=protect_side,
                    type="STOP_LOSS_LIMIT",
                    quantity=qty_adj,
                    price=price_adj,
                    stopPrice=stop_adj,
                    timeInForce="GTC",
                    newClientOrderId=_short_client_id("sl"),
                    newOrderRespType="RESULT",
                )
                protection_kind = "stop_loss_limit"
                protection_ids = {
                    "orderId": sl_resp.get("orderId"),
                    "clientOrderId": sl_resp.get("clientOrderId"),
                    "status": sl_resp.get("status"),
                }

            # только TP
            elif tp_p is not None:
                qty_adj, price_adj, _, _ = await self._apply_symbol_rules(
                    symbol=sym,
                    type_u="TAKE_PROFIT_LIMIT",
                    side_u=protect_side,
                    qty=protect_qty,
                    price=tp_p,
                    quote_qty=None,
                    stop_price=None,
                )
                tp_resp = await self.client.post_order(
                    symbol=sym,
                    side=protect_side,
                    type="TAKE_PROFIT_LIMIT",
                    quantity=qty_adj,
                    price=price_adj,
                    timeInForce="GTC",
                    newClientOrderId=_short_client_id("tp"),
                    newOrderRespType="RESULT",
                )
                protection_kind = "take_profit_limit"
                protection_ids = {
                    "orderId": tp_resp.get("orderId"),
                    "clientOrderId": tp_resp.get("clientOrderId"),
                    "status": tp_resp.get("status"),
                }

        except BinanceAPIError as e:
            LOG.error("Failed to place protection orders: %s", e)
            return {
                **entry,
                "protection": "error",
                "protection_error": {"status_code": e.status_code, "code": e.code, "msg": e.msg},
                "protection_ids": None,
            }

        return {**entry, "protection": protection_kind, "protection_ids": protection_ids}

    async def close_order(
        self,
        *,
        symbol: str,
        qty: Optional[float] = None,
        client_order_id: Optional[str] = None,
        type: Literal["market", "limit"] = "market",
        price: Optional[float] = None,
        timeInForce: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Закрытие за счёт встречного ордера (spot):
          - если qty не задан — продаём полностью свободный BASE.
        """
        sym = symbol.upper()
        s_info = await self.client.get_symbol_info(sym)
        if not s_info:
            raise ValueError(f"Unknown symbol: {sym}")

        base_asset = s_info.get("baseAsset")
        if not base_asset:
            raise ValueError(f"No baseAsset for {sym}")

        # если qty не задан — используем свободный баланс базовой монеты
        use_qty = qty
        if use_qty is None:
            bal = await self.fetch_balance()
            free_map = {b["asset"]: float(b["free"]) for b in bal.get("balances", [])}
            use_qty = free_map.get(base_asset, 0.0)

        if use_qty is None or use_qty <= 0:
            return {
                "exchange": "binance",
                "testnet": self.testnet,
                "symbol": sym,
                "note": "Nothing to close (qty<=0).",
                "status": "NOOP",
            }

        if type.lower() == "market":
            return await self.open_order(
                symbol=sym,
                side="sell",
                type="market",
                qty=use_qty,
                client_order_id=client_order_id,
            )
        elif type.lower() == "limit":
            if price is None:
                raise ValueError("LIMIT close requires price")
            return await self.open_order(
                symbol=sym,
                side="sell",
                type="limit",
                qty=use_qty,
                price=price,
                timeInForce=timeInForce or "GTC",
                client_order_id=client_order_id,
            )
        else:
            raise ValueError("Unsupported close order type")

    # ──────────────────────────────────────────────────────────────────────
    # Балансы и «позиции»
    # ──────────────────────────────────────────────────────────────────────
    async def fetch_balance(self) -> Dict[str, Any]:
        acc = await self.client.get_account()
        raw_bal = acc.get("balances", []) or []
        balances: List[Dict[str, Any]] = []
        for b in raw_bal:
            try:
                free = float(b.get("free", 0) or 0)
                locked = float(b.get("locked", 0) or 0)
            except Exception:
                free, locked = 0.0, 0.0
            balances.append({
                "asset": b.get("asset"),
                "free": free,
                "locked": locked,
                "total": free + locked,
            })
        return {
            "exchange": "binance",
            "testnet": self.testnet,
            "updateTime": acc.get("updateTime"),
            "balances": balances,
            "raw": acc,
        }

    async def list_positions(self, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Псевдо-позиции для spot из балансов:
          - Для символа BASEQUOTE наличие BASE (>0) трактуем как лонг по этому символу.
        """
        acc = await self.fetch_balance()
        free_map = {b["asset"]: float(b["free"]) for b in acc.get("balances", [])}

        info = await self.client.get_exchange_info()
        sym_list = [s["symbol"] for s in info.get("symbols", [])]

        out: List[Dict[str, Any]] = []
        target = [s.upper() for s in symbols] if symbols else sym_list
        for sym in target:
            s_info = next((x for x in info.get("symbols", []) if x.get("symbol") == sym), None)
            if not s_info:
                continue
            base = s_info.get("baseAsset")
            quote = s_info.get("quoteAsset")
            qty = float(free_map.get(base, 0.0))
            if qty > 0.0:
                out.append({
                    "symbol": sym,
                    "side": "long",
                    "qty": qty,
                    "base": base,
                    "quote": quote,
                })
        return out

    # Алиасы
    async def get_positions(self, *, symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        return await self.list_positions(symbols=symbols)

    async def close_position(self, symbol: str, *, type: Literal["market", "limit"] = "market", price: Optional[float] = None) -> Dict[str, Any]:
        return await self.close_order(symbol=symbol, type=type, price=price)

    async def close_all_positions(self, *, quote_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Закрыть все «позиции» на spot (продать все BASE активы по символам; опционально — только символы с указанным quote).
        """
        info = await self.client.get_exchange_info()
        pos = await self.list_positions()
        results: List[Dict[str, Any]] = []
        for p in pos:
            sym = p["symbol"]
            if quote_filter:
                s_info = next((x for x in info.get("symbols", []) if x.get("symbol") == sym), None)
                if not s_info or s_info.get("quoteAsset") != quote_filter.upper():
                    continue
            try:
                res = await self.close_order(symbol=sym, qty=p["qty"], type="market")
                results.append(res)
            except Exception as e:
                results.append({"symbol": sym, "error": repr(e)})
        return results
