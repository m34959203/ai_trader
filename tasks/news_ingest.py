"""Background job for refreshing alternative data (RSS news feeds)."""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from news import rss_client

LOG = logging.getLogger("ai_trader.news_ingest")

_DEFAULT_INTERVAL_SEC = int(os.getenv("NEWS_REFRESH_INTERVAL_SEC", "900"))
_DEFAULT_LIMIT = int(os.getenv("NEWS_REFRESH_LIMIT", "50"))


@asynccontextmanager
def _log_context(action: str) -> AsyncIterator[None]:
    try:
        LOG.debug("News ingest step: %s", action)
        yield
    except Exception as exc:  # pragma: no cover - logging only
        LOG.warning("News ingest step %s failed: %r", action, exc)
        raise


async def refresh_once(*, limit: Optional[int] = None) -> int:
    """Fetch news feeds and persist them into the shared cache file."""

    limit = int(limit or _DEFAULT_LIMIT)
    async with _log_context(f"refresh(limit={limit})"):
        merged = await rss_client.refresh_news(limit=limit)
    LOG.info("Refreshed %s news items", len(merged))
    return len(merged)


async def background_loop(
    *,
    interval_seconds: Optional[int] = None,
    stop_event: Optional[asyncio.Event] = None,
    initial_delay: Optional[int] = None,
    limit: Optional[int] = None,
) -> None:
    """Continuously refresh news on a schedule."""

    sleep_interval = max(int(interval_seconds or _DEFAULT_INTERVAL_SEC), 60)
    news_limit = int(limit or _DEFAULT_LIMIT)

    if initial_delay:
        await asyncio.sleep(max(int(initial_delay), 0))

    while True:
        if stop_event and stop_event.is_set():
            LOG.info("News ingest loop stopped via stop_event")
            break
        try:
            await refresh_once(limit=news_limit)
        except Exception as exc:  # pragma: no cover - resilient loop
            LOG.warning("News ingest iteration failed: %r", exc)
        await asyncio.sleep(sleep_interval)


def run_sync() -> None:
    """CLI entry point for ad-hoc news refresh."""

    asyncio.run(refresh_once())


if __name__ == "__main__":  # pragma: no cover - manual execution helper
    run_sync()
