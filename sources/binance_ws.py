from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, Optional

try:
    import websockets  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Library 'websockets' is required for BinanceWS. "
        "Add it to requirements.txt: websockets>=12.0"
    ) from e

# ──────────────────────────────────────────────────────────────────────────────
# REST helper (обёртка для listenKey), ожидаем, что у тебя есть sources/binance.py
# ──────────────────────────────────────────────────────────────────────────────
try:
    # ожидаем реализацию классов/методов в твоём проекте:
    #   BinanceREST(...).create_listen_key()
    #   BinanceREST(...).keepalive_listen_key(listen_key)
    #   BinanceREST(...).close_listen_key(listen_key)
    from .binance import BinanceREST  # type: ignore
except Exception:
    BinanceREST = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Константы
# ──────────────────────────────────────────────────────────────────────────────
BINANCE_WS_PUBLIC = "wss://stream.binance.com:9443"
BINANCE_WS_TESTNET = "wss://testnet.binance.vision"  # только часть публичных стримов доступна

# Согласно докам Binance listenKey живёт 60 минут, keepalive нужно слать чаще (≤30m)
LISTEN_KEY_KEEPALIVE_SEC = 25 * 60

# Для reconnect/backoff
BACKOFF_BASE = 1.0
BACKOFF_MAX = 30.0
HEARTBEAT_TIMEOUT = 60.0  # если не пришло ни одного сообщения дольше N сек — реконнект


# ──────────────────────────────────────────────────────────────────────────────
# DTO
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class WSConfig:
    symbols: list[str]
    intervals: list[str]
    depth: bool = False               # подписка depth@100ms
    testnet: bool = False
    user_stream: bool = False         # для user-data-stream
    recv_window: int = 5000           # на будущее
    # если есть прокси/подписки — можно расширить конфиг здесь


# ──────────────────────────────────────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────────────────────────────────────
def _utc_ms() -> int:
    return int(time.time() * 1000)


def _iso_from_ms(ms: int) -> str:
    try:
        # быстрый путь без зависимостей
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ms / 1000)) + f".{ms%1000:03d}Z"
    except Exception:
        return ""


def _build_public_streams(symbols: Iterable[str], intervals: Iterable[str], add_depth: bool) -> list[str]:
    """Собираем список каналов для combined stream."""
    streams: list[str] = []
    for s in symbols:
        s_lc = s.lower()
        for itv in intervals:
            streams.append(f"{s_lc}@kline_{itv}")
        if add_depth:
            # Частота 100ms — для аккуратности можно сделать параметром
            streams.append(f"{s_lc}@depth@100ms")
    return streams


def _combined_url(base: str, streams: list[str]) -> str:
    # /stream?streams=btcusdt@kline_1m/ethusdt@kline_1m...
    path = "/stream?streams=" + "/".join(streams)
    return base + path


def _random_jitter(a: float = 0.0, b: float = 0.5) -> float:
    return random.uniform(a, b)


# ──────────────────────────────────────────────────────────────────────────────
# Класс клиента
# ──────────────────────────────────────────────────────────────────────────────
class BinanceWS:
    """
    Унифицированный WebSocket-клиент для Binance:
      • start_market(symbols, intervals, depth) -> AsyncIterator[dict]
      • start_user() -> AsyncIterator[dict]

    События приводятся к унифицированному виду:
      {
        "type": "kline" | "depth" | "executionReport" | "outboundAccountPosition" | "balanceUpdate" | "accountUpdate",
        "symbol": "BTCUSDT",
        "ts": 1712345678901,           # UTC ms
        "ts_iso": "2025-09-03T12:34:56.789Z",
        "data": {...}                  # сырой полезный объект Binance
      }
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        testnet: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.testnet = testnet
        self.api_key = api_key
        self.api_secret = api_secret
        self._log = logger or logging.getLogger(__name__)
        self._stop_event = asyncio.Event()
        self._last_msg_ts = _utc_ms()

        # REST только для user-stream
        if BinanceREST and (api_key and api_secret):
            self._rest = BinanceREST(api_key=api_key, api_secret=api_secret, testnet=testnet)  # type: ignore
        else:
            self._rest = None

    # ────────────────────────────────────────────────────────────────────────
    # Публичные методы
    # ────────────────────────────────────────────────────────────────────────
    async def start_market(
        self,
        symbols: list[str],
        intervals: list[str],
        *,
        depth: bool = False,
    ) -> AsyncIterator[dict]:
        """
        Подписка на публичные стримы (kline и depth). Автовосстановление соединения, backoff.
        """
        cfg = WSConfig(symbols=symbols, intervals=intervals, depth=depth, testnet=self.testnet)

        base = BINANCE_WS_TESTNET if self.testnet else BINANCE_WS_PUBLIC
        streams = _build_public_streams(cfg.symbols, cfg.intervals, cfg.depth)
        url = _combined_url(base, streams)

        backoff = BACKOFF_BASE
        self._log.info("WS[market] connecting: %s", url)

        while not self._stop_event.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:  # type: ignore
                    self._log.info("WS[market] connected")
                    backoff = BACKOFF_BASE
                    self._last_msg_ts = _utc_ms()

                    # Читаем до остановки
                    async for raw in ws:
                        self._last_msg_ts = _utc_ms()
                        evt = self._parse_public_message(raw)
                        if evt is not None:
                            yield evt

                        # Простой heartbeat-контроль: если давно тишина — форсим reconnect
                        if (_utc_ms() - self._last_msg_ts) > (HEARTBEAT_TIMEOUT * 1000):
                            self._log.warning("WS[market] heartbeat timeout; reconnecting…")
                            break

            except asyncio.CancelledError:
                self._log.info("WS[market] cancelled")
                break
            except Exception as e:
                self._log.warning("WS[market] error (%s), reconnect in %.1fs", e, backoff)
                await asyncio.sleep(backoff + _random_jitter())
                backoff = min(BACKOFF_MAX, backoff * 2)

        self._log.info("WS[market] stopped")

    async def start_user(self) -> AsyncIterator[dict]:
        """
        Подписка на user data stream (executionReport, outboundAccountPosition, balanceUpdate, accountUpdate).
        Требует наличия REST-обёртки и API-ключей.
        """
        if not self._rest:
            raise RuntimeError(
                "BinanceWS.start_user() requires REST client and API keys. "
                "Make sure sources/binance.py provides BinanceREST and keys are passed."
            )

        base = BINANCE_WS_TESTNET if self.testnet else BINANCE_WS_PUBLIC
        backoff = BACKOFF_BASE

        while not self._stop_event.is_set():
            listen_key = None
            keepalive_task: Optional[asyncio.Task] = None
            try:
                # 1) Получаем listenKey через REST
                listen_key = await self._rest.create_listen_key()
                url = f"{base}/ws/{listen_key}"
                self._log.info("WS[user] connecting: %s", url)

                # 2) Запускаем keepalive-поддержку
                keepalive_task = asyncio.create_task(self._keepalive_loop(listen_key))

                async with websockets.connect(url, ping_interval=20, ping_timeout=20) as ws:  # type: ignore
                    self._log.info("WS[user] connected (listenKey=%s)", listen_key)
                    backoff = BACKOFF_BASE
                    self._last_msg_ts = _utc_ms()

                    async for raw in ws:
                        self._last_msg_ts = _utc_ms()
                        evt = self._parse_user_message(raw)
                        if evt is not None:
                            yield evt

                        if (_utc_ms() - self._last_msg_ts) > (HEARTBEAT_TIMEOUT * 1000):
                            self._log.warning("WS[user] heartbeat timeout; reconnecting…")
                            break

            except asyncio.CancelledError:
                self._log.info("WS[user] cancelled")
                break
            except Exception as e:
                self._log.warning("WS[user] error (%s), reconnect in %.1fs", e, backoff)
                await asyncio.sleep(backoff + _random_jitter())
                backoff = min(BACKOFF_MAX, backoff * 2)
            finally:
                # Закрываем keepalive
                if keepalive_task:
                    keepalive_task.cancel()
                    with contextlib_suppress(asyncio.CancelledError):
                        await keepalive_task
                # Пробуем закрыть listenKey
                if listen_key:
                    with contextlib_suppress(Exception):
                        await self._rest.close_listen_key(listen_key)

        self._log.info("WS[user] stopped")

    def stop(self) -> None:
        """Запрос на остановку из внешнего кода."""
        self._stop_event.set()

    # ────────────────────────────────────────────────────────────────────────
    # Внутреннее
    # ────────────────────────────────────────────────────────────────────────
    async def _keepalive_loop(self, listen_key: str) -> None:
        """
        Поддерживаем listenKey живым через REST. Binance требует keepalive ≤30 минут.
        """
        assert self._rest is not None
        interval = LISTEN_KEY_KEEPALIVE_SEC
        self._log.info("WS[user] keepalive loop started (every %ss)", interval)
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(interval)
                try:
                    await self._rest.keepalive_listen_key(listen_key)
                    self._log.debug("WS[user] keepalive OK")
                except Exception as e:
                    self._log.warning("WS[user] keepalive failed: %s", e)
                    # пусть внешний reconnect обработает
                    return
        except asyncio.CancelledError:
            self._log.debug("WS[user] keepalive cancelled")
            raise

    def _parse_public_message(self, raw: str | bytes) -> Optional[dict]:
        """
        Сообщения с публичного combined-потока приходят с обёрткой:
          {"stream":"btcusdt@kline_1m","data":{...}}
        """
        try:
            msg = json.loads(raw)
        except Exception:
            self._log.debug("WS[market] non-json message: %r", raw)
            return None

        data = msg.get("data")
        stream = msg.get("stream", "")
        if not isinstance(data, dict):
            return None

        # kline
        if "k" in data and data.get("e") == "kline":
            k = data["k"]
            symbol = data.get("s") or k.get("s")
            ts = int(data.get("E") or k.get("t") or _utc_ms())
            evt = {
                "type": "kline",
                "symbol": symbol,
                "ts": ts,
                "ts_iso": _iso_from_ms(ts),
                "data": {
                    "interval": k.get("i"),
                    "is_closed": bool(k.get("x")),
                    "open_time": int(k.get("t")),
                    "close_time": int(k.get("T")),
                    "open": k.get("o"),
                    "high": k.get("h"),
                    "low": k.get("l"),
                    "close": k.get("c"),
                    "volume": k.get("v"),
                    "trades": k.get("n"),
                },
                "raw_stream": stream,
            }
            return evt

        # depth
        if data.get("e") in ("depthUpdate",):
            symbol = data.get("s")
            ts = int(data.get("E") or _utc_ms())
            evt = {
                "type": "depth",
                "symbol": symbol,
                "ts": ts,
                "ts_iso": _iso_from_ms(ts),
                "data": {
                    "first_update_id": data.get("U"),
                    "final_update_id": data.get("u"),
                    "bids": data.get("b", []),
                    "asks": data.get("a", []),
                },
                "raw_stream": stream,
            }
            return evt

        # неизвестное событие — возвращаем «как есть»
        symbol = data.get("s")
        ts = int(data.get("E") or _utc_ms())
        return {
            "type": data.get("e") or "unknown",
            "symbol": symbol,
            "ts": ts,
            "ts_iso": _iso_from_ms(ts),
            "data": data,
            "raw_stream": stream,
        }

    def _parse_user_message(self, raw: str | bytes) -> Optional[dict]:
        """
        User data stream без обёртки 'stream', сразу объект event.
        Типичные e: executionReport, outboundAccountPosition, balanceUpdate, accountUpdate.
        """
        try:
            data = json.loads(raw)
        except Exception:
            self._log.debug("WS[user] non-json message: %r", raw)
            return None

        evt_type = data.get("e", "unknown")
        ts = int(data.get("E") or _utc_ms())
        symbol = data.get("s")

        # executionReport (по ордерам)
        if evt_type == "executionReport":
            # полезные поля: X (order status), x (execution type), i(orderId), c(clientOrderId)
            # L(lastExecutedPrice), l(lastExecutedQty), Z(cumQuote), z(cumQty)
            return {
                "type": "executionReport",
                "symbol": symbol,
                "ts": ts,
                "ts_iso": _iso_from_ms(ts),
                "data": {
                    "orderId": data.get("i"),
                    "clientOrderId": data.get("c"),
                    "orderStatus": data.get("X"),       # NEW, PARTIALLY_FILLED, FILLED, CANCELED, REJECTED, EXPIRED
                    "execType": data.get("x"),          # NEW, CANCELED, REPLACED, REJECTED, TRADE, EXPIRED, AMENDMENT
                    "side": data.get("S"),
                    "orderType": data.get("o"),
                    "timeInForce": data.get("f"),
                    "lastPrice": data.get("L"),
                    "lastQty": data.get("l"),
                    "cumQuote": data.get("Z"),
                    "cumQty": data.get("z"),
                    "commission": data.get("n"),
                    "commissionAsset": data.get("N"),
                    "orderPrice": data.get("p"),
                    "orderQty": data.get("q"),
                },
            }

        # изменения позиций/балансов
        if evt_type in ("outboundAccountPosition", "balanceUpdate", "accountUpdate"):
            return {
                "type": evt_type,
                "symbol": symbol,
                "ts": ts,
                "ts_iso": _iso_from_ms(ts),
                "data": data,
            }

        # неизвестное — не фильтруем
        return {
            "type": evt_type,
            "symbol": symbol,
            "ts": ts,
            "ts_iso": _iso_from_ms(ts),
            "data": data,
        }


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def market_stream(
    *,
    symbols: list[str],
    intervals: list[str],
    depth: bool = False,
    testnet: bool = False,
    logger: Optional[logging.Logger] = None,
) -> AsyncIterator[AsyncIterator[dict]]:
    """
    Пример контекст-менеджера, если удобно использовать в сервисах:
        async with market_stream(symbols=..., intervals=...) as stream:
            async for evt in stream:
                ...
    """
    ws = BinanceWS(testnet=testnet, logger=logger)
    agen = ws.start_market(symbols=symbols, intervals=intervals, depth=depth)
    try:
        yield agen
    finally:
        ws.stop()


@asynccontextmanager
async def user_stream(
    *,
    api_key: str,
    api_secret: str,
    testnet: bool = False,
    logger: Optional[logging.Logger] = None,
) -> AsyncIterator[AsyncIterator[dict]]:
    ws = BinanceWS(api_key=api_key, api_secret=api_secret, testnet=testnet, logger=logger)
    agen = ws.start_user()
    try:
        yield agen
    finally:
        ws.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательное подавление исключений (без импорта contextlib.suppress для 3.10-)
# ──────────────────────────────────────────────────────────────────────────────
class contextlib_suppress:
    def __init__(self, *exceptions):
        self._exceptions = exceptions

    def __enter__(self):
        return None

    def __exit__(self, exctype, exc, tb):
        return exctype is not None and issubclass(exctype, self._exceptions)
