import asyncio, os, time, logging, pathlib
from typing import Optional

HEARTBEAT_FILE = pathlib.Path("data/state/heartbeat.txt")
LOG = logging.getLogger("watchdog")

class EventLoopWatchdog:
    def __init__(self, interval: float = 5.0, max_consecutive_misses: int = 12):
        self.interval = interval
        self.max_misses = max_consecutive_misses
        self._stopped = False
        self._misses = 0

    async def run(self):
        HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
        while not self._stopped:
            t0 = time.perf_counter()
            try:
                await asyncio.sleep(self.interval)
                delay = time.perf_counter() - t0 - self.interval
                # heartbeat
                HEARTBEAT_FILE.write_text(str(int(time.time())))
                if delay > self.interval * 2:
                    self._misses += 1
                    LOG.warning("event loop lag=%.2fs, misses=%d", delay, self._misses)
                else:
                    self._misses = 0
                if self._misses >= self.max_misses:
                    LOG.error("Watchdog: too many misses -> exiting for supervisor restart")
                    os._exit(42)  # пусть супервизор перезапустит
            except asyncio.CancelledError:
                break
            except Exception as e:
                LOG.exception("Watchdog error: %s", e)

    def stop(self):
        self._stopped = True
