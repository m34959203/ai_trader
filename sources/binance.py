from __future__ import annotations

import os
import time
import math
import json
import logging
from typing import Iterable, Dict, List, Optional, Any

import requests

# Для асинхронных REST (listenKey и др.)
try:
    import httpx  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Library 'httpx' is required for BinanceREST. "
        "Add it to requirements.txt: httpx>=0.27.0"
    ) from e

from .base import SourceError

# ------------------------------------------------------------------------------
# Конфигурация
# ------------------------------------------------------------------------------
# Базовые URL для REST:
#  - prod:    https://api.binance.com/api
#  - testnet: https://testnet.binance.vision/api
_BINANCE_HOST = os.getenv("BINANCE_HOST", "https://api.binance.com")
_BINANCE_API_BASE = os.getenv("BINANCE_API_BASE", f"{_BINANCE_HOST}/api")
_BINANCE_WS_BASE = os.getenv("BINANCE_WS_BASE", "wss://stream.binance.com:9443")
_BINANCE_TESTNET = str(os.getenv("BINANCE_TESTNET", "0")).strip().lower() in ("1", "true", "yes", "on")

_LOG = logging.getLogger("ai_trader.binance")

# Соответствие таймфреймов к Binance intervals
_BI_TF = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
}

# Максимум 1000 свечей за один вызов /klines
_MAX_BINANCE_LIMIT = 1000

# Ретраи/таймауты
_REQ_TIMEOUT = 30  # секунд
_MAX_RETRIES = 4
_BACKOFF_BASE = 0.8  # экспоненциальная пауза: 0.8s, 1.6s, 3.2s, ...
_BACKOFF_MAX = 30.0

# ------------------------------------------------------------------------------
# Вспомогательные функции (sync)
# ------------------------------------------------------------------------------
def _safe_float(x: object, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _sleep_backoff(attempt: int) -> None:
    # attempt = 0..N-1
    delay = _BACKOFF_BASE * (2 ** attempt)
    time.sleep(delay)


def _parse_binance_error_resp_text(status_code: int, text: str) -> str:
    # На случай когда resp.json() падает
    return f"Binance HTTP {status_code}: {text[:200]}"


def _parse_binance_error(resp: requests.Response) -> str:
    try:
        data = resp.json()
        # Пример: {"code":-1121,"msg":"Invalid symbol."}
        code = data.get("code")
        msg = data.get("msg")
        return f"Binance error {resp.status_code} (code={code}): {msg}"
    except Exception:
        return _parse_binance_error_resp_text(resp.status_code, resp.text or "")


def _next_start_time_from_batch(batch: List[list]) -> Optional[int]:
    """
    Возвращает следующий startTime (ms) для пагинации — это (последний openTime + 1 ms).
    batch — список клайнов ([openTime, open, high, low, close, volume, closeTime, ...])
    """
    if not batch:
        return None
    last_open_ms = int(batch[-1][0])
    return last_open_ms + 1


# ------------------------------------------------------------------------------
# Публичная (sync) функция получения OHLCV (совместимость с текущим кодом)
# ------------------------------------------------------------------------------
def fetch(
    symbol: str,
    timeframe: str,
    limit: int | None,
    ts_from: int | None,
    ts_to: int | None,
) -> Iterable[Dict]:
    """
    Возвращает нормализованный список свечей dict:
    {
        "source": "binance",
        "asset":  "<SYMBOL>",
        "tf":     "<timeframe>",
        "ts":     <unix_seconds>,
        "open":   float,
        "high":   float,
        "low":    float,
        "close":  float,
        "volume": float
    }

    Параметры:
      - symbol: 'BTCUSDT', 'ETHUSDT', ...
      - timeframe: '1m','5m','15m','30m','1h','4h','1d'
      - limit: желаемое кол-во свечей (будет ограничено максимумом и/или рамками времени)
      - ts_from / ts_to: границы по времени в секундах (UTC). Можно None.
    """
    interval = _BI_TF.get(timeframe)
    if not interval:
        raise SourceError(f"timeframe '{timeframe}' не поддерживается Binance")

    # Нормализация пределов
    need = int(limit or _MAX_BINANCE_LIMIT)
    if need <= 0:
        return []

    # База для /api/v3/klines
    if _BINANCE_TESTNET:
        api_base = "https://testnet.binance.vision/api"
    else:
        # уважим переопределение через ENV, если задано
        api_base = _BINANCE_API_BASE

    url = f"{api_base}/v3/klines"

    # Binance принимает startTime/endTime в миллисекундах
    start_ms = int(ts_from * 1000) if ts_from else None
    end_ms = int(ts_to * 1000) if ts_to else None

    # Для надёжности используем общий session
    session = requests.Session()

    collected: List[Dict] = []
    params: Dict[str, object] = {
        "symbol": symbol.upper(),
        "interval": interval,
        # Ставим максимальный пакет, чтобы уменьшить число запросов
        "limit": min(_MAX_BINANCE_LIMIT, max(1, min(need, _MAX_BINANCE_LIMIT))),
    }
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms

    # Если хотим <=1000 и заданы границы, достаточно одного запроса.
    # Если хотим >1000 или нет явного endTime — пагинируем по openTime.
    while True:
        batch = _request_klines_sync(session, url, params)
        if not batch:
            break

        # Преобразуем к нормализованному виду
        for k in batch:
            # Формат k:
            # [
            #   0 openTime(ms), 1 open, 2 high, 3 low, 4 close, 5 volume,
            #   6 closeTime(ms), 7 quoteAssetVolume, 8 numberOfTrades,
            #   9 takerBuyBaseAssetVolume, 10 takerBuyQuoteAssetVolume, 11 ignore
            # ]
            open_time_ms = int(k[0])
            ts_sec = open_time_ms // 1000

            # строгая фильтрация по сек. границам, если заданы
            if ts_from and ts_sec < ts_from:
                continue
            if ts_to and ts_sec > ts_to:
                continue

            collected.append(
                {
                    "source": "binance",
                    "asset": symbol.upper(),
                    "tf": timeframe,
                    "ts": int(ts_sec),
                    "open": _safe_float(k[1]),
                    "high": _safe_float(k[2]),
                    "low": _safe_float(k[3]),
                    "close": _safe_float(k[4]),
                    "volume": _safe_float(k[5]),
                }
            )

            if len(collected) >= need:
                break

        if len(collected) >= need:
            break

        # Пагинация: двигаем окно по времени от последнего openTime
        next_start = _next_start_time_from_batch(batch)
        if next_start is None:
            break

        params["startTime"] = next_start

        # Если endTime был задан и мы уже "догнали" его — выходим
        if end_ms is not None and next_start > end_ms:
            break

        # Если Binance вернул меньше, чем спросили — дальше уже нечего выбирать
        if len(batch) < int(params["limit"]):
            break

    # Возвращаем в возрастающем порядке по времени (на всякий случай)
    collected.sort(key=lambda r: r["ts"])
    return collected


def _request_klines_sync(
    session: requests.Session,
    url: str,
    params: Dict[str, object],
) -> List[list]:
    """
    Выполняет GET /api/v3/klines c ретраями.
    Возвращает список клайнов (каждый элемент - массив из 12 значений).
    """
    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        try:
            resp = session.get(url, params=params, timeout=_REQ_TIMEOUT)
            # 2xx
            if 200 <= resp.status_code < 300:
                return resp.json()

            # 429 Too Many Requests / 418 Banned / 5xx -> ретрай
            if resp.status_code in (418, 429) or (500 <= resp.status_code < 600):
                _LOG.warning("Binance throttled or 5xx, retry %s: %s", attempt + 1, _parse_binance_error(resp))
                _sleep_backoff(attempt)
                continue

            # Остальные ошибки — сразу падаем с понятным сообщением
            raise SourceError(_parse_binance_error(resp))

        except requests.RequestException as e:
            last_exc = e
            _LOG.warning("Network error to Binance, retry %s: %r", attempt + 1, e)
            _sleep_backoff(attempt)
            continue

    # Если все ретраи исчерпаны
    if last_exc:
        raise SourceError(f"Binance request failed after retries: {last_exc!r}")
    raise SourceError("Binance request failed after retries (unknown error)")


# ------------------------------------------------------------------------------
# Асинхронный REST-клиент для listenKey и (опционально) других вызовов
# ------------------------------------------------------------------------------
class BinanceREST:
    """
    Асинхронная обёртка REST для Binance (spot):
      • create_listen_key()        POST /api/v3/userDataStream
      • keepalive_listen_key(lk)   PUT  /api/v3/userDataStream
      • close_listen_key(lk)       DELETE /api/v3/userDataStream
      • aget_klines(...)           GET  /api/v3/klines (опционально)

    Нужен только X-MBX-APIKEY для listenKey (подпись не требуется).
    """

    def __init__(
        self,
        *,
        api_key: str,
        api_secret: str | None = None,   # не требуется для userDataStream, но оставим на будущее
        testnet: bool = False,
        timeout: float = 30.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = bool(testnet or _BINANCE_TESTNET)
        self.timeout = timeout
        self._log = logger or logging.getLogger("ai_trader.binance.rest")

        if self.testnet:
            self.api_base = "https://testnet.binance.vision/api"
        else:
            self.api_base = _BINANCE_API_BASE

        # Ленивая инициализация клиента; можно сделать явный .aclose()
        self._client: httpx.AsyncClient | None = None

    # ---------------------------
    # Жизненный цикл http-клиента
    # ---------------------------
    async def _client_get(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            finally:
                self._client = None

    # ---------------------------
    # Вспомогательные методы
    # ---------------------------
    def _headers_key(self) -> dict[str, str]:
        return {"X-MBX-APIKEY": self.api_key}

    def _err_from_httpx(self, resp: httpx.Response) -> str:
        try:
            data = resp.json()
            return f"Binance error {resp.status_code} (code={data.get('code')}): {data.get('msg')}"
        except Exception:
            return _parse_binance_error_resp_text(resp.status_code, resp.text)

    async def _with_retries(self, coro_factory, *, max_retries: int = _MAX_RETRIES) -> Any:
        """
        Универсальный ретраер с backoff для httpx-вызовов.
        coro_factory: lambda -> awaitable(httpx.Response)
        """
        last_exc: Optional[Exception] = None
        backoff = _BACKOFF_BASE
        for attempt in range(max_retries):
            try:
                resp: httpx.Response = await coro_factory()
                if 200 <= resp.status_code < 300:
                    return resp
                if resp.status_code in (418, 429) or (500 <= resp.status_code < 600):
                    self._log.warning("Binance REST throttled/5xx, retry %s: %s", attempt + 1, self._err_from_httpx(resp))
                    await asyncio_sleep(backoff)
                    backoff = min(_BACKOFF_MAX, backoff * 2)
                    continue
                raise SourceError(self._err_from_httpx(resp))
            except (httpx.HTTPError, httpx.TransportError) as e:
                last_exc = e
                self._log.warning("Binance REST network error, retry %s: %r", attempt + 1, e)
                await asyncio_sleep(backoff)
                backoff = min(_BACKOFF_MAX, backoff * 2)
                continue
        if last_exc:
            raise SourceError(f"Binance REST failed after retries: {last_exc!r}")
        raise SourceError("Binance REST failed after retries (unknown error)")

    # ---------------------------
    # Listen key (userDataStream)
    # ---------------------------
    async def create_listen_key(self) -> str:
        """
        POST /api/v3/userDataStream
        """
        client = await self._client_get()
        url = f"{self.api_base}/v3/userDataStream"
        resp: httpx.Response = await self._with_retries(
            lambda: client.post(url, headers=self._headers_key())
        )
        data = resp.json()
        lk = data.get("listenKey")
        if not lk:
            raise SourceError(f"Binance REST: listenKey missing in response: {data}")
        self._log.info("create_listen_key OK")
        return lk

    async def keepalive_listen_key(self, listen_key: str) -> None:
        """
        PUT /api/v3/userDataStream
        """
        client = await self._client_get()
        url = f"{self.api_base}/v3/userDataStream"
        resp: httpx.Response = await self._with_retries(
            lambda: client.put(url, headers=self._headers_key(), params={"listenKey": listen_key})
        )
        # Успешный ответ: 200 с пустым JSON {}
        self._log.debug("keepalive_listen_key OK")

    async def close_listen_key(self, listen_key: str) -> None:
        """
        DELETE /api/v3/userDataStream
        """
        client = await self._client_get()
        url = f"{self.api_base}/v3/userDataStream"
        resp: httpx.Response = await self._with_retries(
            lambda: client.delete(url, headers=self._headers_key(), params={"listenKey": listen_key})
        )
        self._log.info("close_listen_key OK")

    # ---------------------------
    # (Опционально) асинхронные свечи
    # ---------------------------
    async def aget_klines(
        self,
        symbol: str,
        interval: str,
        *,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = _MAX_BINANCE_LIMIT,
    ) -> list[list]:
        """
        GET /api/v3/klines (async), единичный запрос (без пагинации).
        """
        client = await self._client_get()
        url = f"{self.api_base}/v3/klines"
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": max(1, min(limit, _MAX_BINANCE_LIMIT)),
        }
        if start_ms is not None:
            params["startTime"] = int(start_ms)
        if end_ms is not None:
            params["endTime"] = int(end_ms)

        resp: httpx.Response = await self._with_retries(
            lambda: client.get(url, params=params, timeout=self.timeout)
        )
        return resp.json()


# ------------------------------------------------------------------------------
# Вспомогательное: asyncio.sleep без прямого импорта asyncio в верхнем уровне
# ------------------------------------------------------------------------------
try:
    import asyncio

    async def asyncio_sleep(sec: float) -> None:  # pragma: no cover
        await asyncio.sleep(sec)
except Exception:  # pragma: no cover
    # fallback (не должен использоваться в проде)
    async def asyncio_sleep(sec: float) -> None:
        time.sleep(sec)
