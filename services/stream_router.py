from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, Iterable, List, Optional, Set, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Внутренние импорты — см. ранее выданные файлы
# ──────────────────────────────────────────────────────────────────────────────
from ai_trader.sources.binance_ws import BinanceWS
from ai_trader.sources.binance import fetch as fetch_klines_sync, BinanceREST

# ──────────────────────────────────────────────────────────────────────────────
# Конфигурация, типы
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class StreamConfig:
    symbols: List[str]
    intervals: List[str]
    backfill_bars: int = 1000           # сколько баров подтягивать при старте
    rest_poll_sec: float = 2.0          # период пулинга REST при фоллбэке
    testnet: bool = False
    # Если хочешь depth — делай отдельный роутер под стакан:
    with_depth: bool = False            # здесь фокус на kline (OHLCV)

@dataclass
class Bar:
    source: str
    asset: str
    tf: str
    ts: int            # unix seconds (UTC) — open time бара
    open: float
    high: float
    low: float
    close: float
    volume: float

# Унифицированный kline-ивент из WS (см. BinanceWS._parse_public_message)
# {
#   "type": "kline",
#   "symbol": "BTCUSDT",
#   "ts": <event_ts_ms>,
#   "ts_iso": "...",
#   "data": {
#       "interval": "1m",
#       "is_closed": bool,
#       "open_time": <ms>,
#       "close_time": <ms>,
#       "open": "....", "high": "...", "low": "...", "close": "...", "volume": "....",
#       ...
#   },
#   "raw_stream": "btcusdt@kline_1m"
# }

K = Tuple[str, str]  # (symbol, interval)

# ──────────────────────────────────────────────────────────────────────────────
# Вспомогательные
# ──────────────────────────────────────────────────────────────────────────────
def _bar_key(symbol: str, interval: str, ts_sec: int) -> Tuple[str, str, int]:
    return (symbol.upper(), interval, int(ts_sec))

def _normalize_ws_kline(evt: dict) -> Optional[Bar]:
    if evt.get("type") != "kline":
        return None
    d = evt.get("data") or {}
    symbol = (evt.get("symbol") or "").upper()
    tf = d.get("interval")
    if not symbol or not tf:
        return None
    try:
        open_time_ms = int(d.get("open_time"))
        ts_sec = open_time_ms // 1000
        return Bar(
            source="binance_ws",
            asset=symbol,
            tf=tf,
            ts=ts_sec,
            open=float(d.get("open")),
            high=float(d.get("high")),
            low=float(d.get("low")),
            close=float(d.get("close")),
            volume=float(d.get("volume")),
        )
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# StreamRouter — объединяем backfill + WS, делаем fallback на REST при сбоях WS
# ──────────────────────────────────────────────────────────────────────────────
class StreamRouter:
    """
    Сервис, который гарантирует непрерывную ленту OHLCV:
      • на старте: backfill по REST за N баров (per symbol/interval);
      • затем: подписка на WS kline, отдаём закрывающиеся бары;
      • при разрыве WS: переходим на REST-пулинг, без дырок;
      • при восстановлении WS: возвращаемся на WS, корректно дедуплицируя бары.

    async for bar in StreamRouter.run(): -> Bar
    """

    def __init__(
        self,
        cfg: StreamConfig,
        *,
        logger: Optional[logging.Logger] = None,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ) -> None:
        self.cfg = cfg
        self._log = logger or logging.getLogger("ai_trader.stream_router")
        self._stop = asyncio.Event()

        # Последний известный закрытый бар (по open_time, сек) на ключ (symbol, tf)
        self._last_closed_ts: Dict[K, int] = {}

        # Виденные ключи баров, для защиты от дублей
        self._seen: Set[Tuple[str, str, int]] = set()

        # Очередь исходящих Bar
        self._queue: asyncio.Queue[Bar] = asyncio.Queue(maxsize=2048)

        # WS клиент
        self._ws = BinanceWS(
            api_key=api_key,
            api_secret=api_secret,
            testnet=cfg.testnet,
            logger=logging.getLogger("ai_trader.binance.ws"),
        )

        # REST для listenKey (если нужен user stream в другой части приложения)
        self._rest = BinanceREST(
            api_key=api_key or "",
            api_secret=api_secret or None,
            testnet=cfg.testnet,
            logger=logging.getLogger("ai_trader.binance.rest"),
        )

        # фоновые задачи
        self._tasks: List[asyncio.Task] = []

    # ──────────────────────────────────────────────────────────────────────
    # API
    # ──────────────────────────────────────────────────────────────────────
    async def run(self) -> AsyncIterator[Bar]:
        """
        Главный асинхронный генератор баров.
        1) бэктилим историю,
        2) запускаем WS-консьюминг с автоматическим fallback на REST,
        3) возвращаем бары через внутреннюю очередь.
        """
        self._log.info("StreamRouter starting…")
        try:
            await self._initial_backfill()

            # Запускаем консьюмер очереди на выдачу
            consumer_task = asyncio.create_task(self._emit_loop())
            self._tasks.append(consumer_task)

            # Запускаем основной цикл WS→REST fallback
            main_task = asyncio.create_task(self._main_loop())
            self._tasks.append(main_task)

            # Читаем из выходного async-генератора
            while not self._stop.is_set():
                bar = await self._queue.get()
                yield bar
                self._queue.task_done()
        finally:
            await self.stop()

    async def stop(self) -> None:
        if self._stop.is_set():
            return
        self._stop.set()
        try:
            self._ws.stop()
        except Exception:
            pass

        for t in self._tasks:
            t.cancel()
        # подождём корректную отмену
        for t in self._tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self._log.debug("task finished with error: %r", e)
        self._tasks.clear()
        self._log.info("StreamRouter stopped")

    # ──────────────────────────────────────────────────────────────────────
    # Внутренние циклы
    # ──────────────────────────────────────────────────────────────────────
    async def _main_loop(self) -> None:
        """
        Основной управляющий цикл:
          — пробуем WS;
          — при ошибке переходим на REST-пулинг до восстановления WS.
        """
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._log.info("WS phase: connecting and consuming…")
                await self._consume_ws_forever()
                backoff = 1.0  # отработали штатно (выход по stop)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._log.warning("WS phase failed: %r → fallback to REST polling", e)

                # Fallback REST-пулинг до следующей попытки WS
                await self._rest_poll_until(lambda: self._stop.is_set(), poll_sec=self.cfg.rest_poll_sec)

                # Небольшой backoff перед новой WS-попыткой
                await asyncio.sleep(backoff)
                backoff = min(30.0, backoff * 2)

    async def _consume_ws_forever(self) -> None:
        """
        Подписка на WS (market kline) и обработка событий.
        Разрыв соединения выбрасывает исключение → ловит _main_loop().
        """
        # NOTE: Binance combined stream для всех symbol/interval
        async for evt in self._ws.start_market(self.cfg.symbols, self.cfg.intervals, depth=self.cfg.with_depth):
            if self._stop.is_set():
                break

            if evt.get("type") != "kline":
                # здесь игнорируем depth/прочие; делай отдельный роутер для стакана
                continue

            bar = _normalize_ws_kline(evt)
            if bar is None:
                continue

            # Пропускаем только ЗАКРЫТЫЕ бары — они финальны
            is_closed = bool(evt["data"].get("is_closed"))
            if not is_closed:
                # незакрытые свечи не публикуем, чтобы не «фликало»
                continue

            # Фильтруем дубликаты и строго соблюдаем порядок времени
            if self._dedupe_and_advance(bar):
                await self._queue.put(bar)

    async def _rest_poll_until(self, stop_predicate, *, poll_sec: float) -> None:
        """
        Пулинг REST, пока возвращается True из stop_predicate (или нас не остановили).
        Мы подтягиваем для каждого (symbol, interval) все новые закрывшиеся бары после
        последнего закрытого ts.
        """
        self._log.info("REST polling fallback started (every %.2fs)…", poll_sec)
        try:
            while not self._stop.is_set() and stop_predicate():
                await self._rest_poll_once()
                await asyncio.sleep(poll_sec)
        finally:
            self._log.info("REST polling fallback finished")

    async def _rest_poll_once(self) -> None:
        """
        Один REST-тик: по каждому (symbol, interval) забрать новые закрытые бары.
        """
        loop = asyncio.get_running_loop()
        tasks = []
        now_sec = int(time.time())

        for s in self.cfg.symbols:
            for itv in self.cfg.intervals:
                key = (s.upper(), itv)
                since = self._last_closed_ts.get(key, 0)

                # Запросим с запаса (−1 бар), чтобы прикрыть границу
                ts_from = max(0, since - 1)
                # Binance вернёт до текущего (возможно незакрытого) — мы ниже отфильтруем
                tasks.append(loop.run_in_executor(None, fetch_klines_sync, s, itv, 1000, ts_from, now_sec))

        results: List[Iterable[dict]] = await asyncio.gather(*tasks, return_exceptions=True)
        idx = 0
        for s in self.cfg.symbols:
            for itv in self.cfg.intervals:
                res = results[idx]
                idx += 1
                if isinstance(res, Exception):
                    self._log.warning("REST poll error for %s/%s: %r", s, itv, res)
                    continue

                # Преобразуем и отдаём только новые закрытые бары
                for d in res:
                    bar = Bar(
                        source=d.get("source", "binance"),
                        asset=d["asset"],
                        tf=d["tf"],
                        ts=int(d["ts"]),
                        open=float(d["open"]),
                        high=float(d["high"]),
                        low=float(d["low"]),
                        close=float(d["close"]),
                        volume=float(d["volume"]),
                    )

                    # Пропустим незакрытый последний бар: fetch возвращает открытое время;
                    # на REST /klines клауза закрытия определяется closeTime, но мы
                    # ориентируемся по тому, что «последний бар может ещё не закрыться».
                    # Так как мы отбираем только ts > last_closed_ts и используем −1 запас,
                    # «неполный» бар не пройдёт как новый закрытый при следующем вызове.
                    if self._dedupe_and_advance(bar):
                        await self._queue.put(bar)

    async def _initial_backfill(self) -> None:
        """
        Стартавая процедура: для каждого (symbol, interval) вытянуть N последних БАРОВ,
        заполнить карту _last_closed_ts и выдать их в очередь в строгом порядке.
        """
        self._log.info("Initial REST backfill: %s symbols × %s intervals × %s bars…",
                       len(self.cfg.symbols), len(self.cfg.intervals), self.cfg.backfill_bars)

        loop = asyncio.get_running_loop()
        tasks = []
        now_sec = int(time.time())

        # Тянем синхронный fetch в thread Pool для всех ключей
        for s in self.cfg.symbols:
            for itv in self.cfg.intervals:
                # ограничим верхнюю границу now — мы хотим только закрытые на момент старта
                tasks.append(loop.run_in_executor(None, fetch_klines_sync, s, itv, self.cfg.backfill_bars, None, now_sec))

        results: List[Iterable[dict]] = await asyncio.gather(*tasks, return_exceptions=True)

        # Соберём все бары в один список и отсортируем по времени, чтобы отдать строго возрастающе
        all_bars: List[Bar] = []
        for res in results:
            if isinstance(res, Exception):
                self._log.warning("Backfill error: %r", res)
                continue
            for d in res:
                try:
                    all_bars.append(
                        Bar(
                            source=d.get("source", "binance"),
                            asset=d["asset"],
                            tf=d["tf"],
                            ts=int(d["ts"]),
                            open=float(d["open"]),
                            high=float(d["high"]),
                            low=float(d["low"]),
                            close=float(d["close"]),
                            volume=float(d["volume"]),
                        )
                    )
                except Exception:
                    continue

        # Отсортировать: по (asset, tf, ts) и выдать без дублей
        all_bars.sort(key=lambda b: (b.asset, b.tf, b.ts))

        emitted = 0
        for bar in all_bars:
            if self._dedupe_and_advance(bar):
                await self._queue.put(bar)
                emitted += 1

        self._log.info("Backfill emitted %s bars", emitted)

    async def _emit_loop(self) -> None:
        """
        Заглушка: здесь можно вставить дополнительную агрегацию/метрики перед отдачей наружу.
        Сейчас просто «пасс» — отдачу делает .run().
        """
        try:
            while not self._stop.is_set():
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            return

    # ──────────────────────────────────────────────────────────────────────
    # Служебное: дедупликация и продвижение last_closed_ts
    # ──────────────────────────────────────────────────────────────────────
    def _dedupe_and_advance(self, bar: Bar) -> bool:
        """
        Возвращает True, если бар «новый и закрытый для своего ключа» и его надо эмитить.
        Правила:
          • отдаём только ts > last_closed_ts[asset, tf];
          • защищаемся от дублей (set of (asset, tf, ts));
          • по итогу продвигаем last_closed_ts для ключа.
        """
        key = (bar.asset.upper(), bar.tf)
        triple = _bar_key(bar.asset, bar.tf, bar.ts)

        # дубликат?
        if triple in self._seen:
            return False

        # порядок времени по ключу
        last_ts = self._last_closed_ts.get(key, -1)
        if bar.ts <= last_ts:
            return False

        # всё ок — «видели» и продвигаем
        self._seen.add(triple)
        self._last_closed_ts[key] = bar.ts
        return True


# ──────────────────────────────────────────────────────────────────────────────
# Пример использования
# ──────────────────────────────────────────────────────────────────────────────
# async def main():
#     cfg = StreamConfig(
#         symbols=["BTCUSDT", "ETHUSDT"],
#         intervals=["1m", "5m"],
#         backfill_bars=1000,
#         testnet=True,
#     )
#     router = StreamRouter(cfg, api_key=os.getenv("BINANCE_API_KEY"), api_secret=os.getenv("BINANCE_API_SECRET"))
#     async for bar in router.run():
#         print(bar)
#
# if __name__ == "__main__":
#     import os
#     import asyncio
#     logging.basicConfig(level=logging.INFO)
#     asyncio.run(main())
