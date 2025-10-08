from __future__ import annotations

import asyncio
import logging
import os

from services.news_pipeline import NewsPipeline

LOG = logging.getLogger("ai_trader.news_ingest")


async def background_loop() -> None:
    interval = float(os.getenv("NEWS_REFRESH_INTERVAL_SEC", "600"))
    limit = int(os.getenv("NEWS_REFRESH_LIMIT", "60"))
    pipeline = NewsPipeline()
    while True:
        try:
            await pipeline.run(limit=limit)
        except Exception as exc:
            LOG.exception("news ingest loop failed: %r", exc)
        await asyncio.sleep(max(60.0, interval))
