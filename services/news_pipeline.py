from __future__ import annotations

"""Market intelligence ingestion pipeline."""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import httpx

from news import rss_client, nlp_gate

try:  # soft import for tests without DB
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    from db.session import AsyncSessionLocal
    from db.models_intel import NewsItem, SentimentSnapshot
    from db.crud_news import upsert_news_items, store_sentiment_snapshot
    from services.audit import record_audit_event
except Exception:  # pragma: no cover
    AsyncSession = None  # type: ignore
    async_sessionmaker = None  # type: ignore
    AsyncSessionLocal = None  # type: ignore
    NewsItem = None  # type: ignore
    SentimentSnapshot = None  # type: ignore
    upsert_news_items = None  # type: ignore
    store_sentiment_snapshot = None  # type: ignore
    record_audit_event = None  # type: ignore

LOG = logging.getLogger("ai_trader.news_pipeline")

_SOCIAL_CACHE = Path(os.getenv("SOCIAL_SENTIMENT_CACHE", "data/social_sentiment.json"))


@dataclass(slots=True)
class SentimentContext:
    news_score: float = 0.0
    social_score: float = 0.0
    fear_greed: float = 50.0
    composite_score: float = 0.0
    methodology: str = "v1"

    @property
    def bias(self) -> float:
        return max(-1.0, min(1.0, self.composite_score))


def _weight_from_impact(impact: str, importance: str) -> float:
    impact_map = {"high": 1.0, "medium": 0.65, "med": 0.65, "low": 0.35}
    importance_map = {"high": 1.0, "med": 0.75, "medium": 0.75, "low": 0.5}
    return impact_map.get(impact.lower(), 0.4) * importance_map.get(importance.lower(), 0.6)


async def _fetch_social_score() -> float:
    url = os.getenv("SOCIAL_SENTIMENT_URL")
    if url:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                data = resp.json()
                for key in ("score", "sentiment", "value"):
                    if key in data:
                        return max(-1.0, min(1.0, float(data[key])))
                if isinstance(data, list) and data:
                    return max(-1.0, min(1.0, float(data[0])))
        except Exception as exc:
            LOG.debug("social sentiment fetch failed: %r", exc)
    if _SOCIAL_CACHE.exists():
        try:
            cache = json.loads(_SOCIAL_CACHE.read_text(encoding="utf-8"))
            if isinstance(cache, dict) and "score" in cache:
                return max(-1.0, min(1.0, float(cache["score"])))
        except Exception:
            pass
    return 0.0


async def _fetch_fear_greed() -> float:
    url = os.getenv("FEAR_GREED_URL", "https://api.alternative.me/fng/?limit=1")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict):
                if "data" in data and isinstance(data["data"], list) and data["data"]:
                    value = data["data"][0].get("value")
                    if value is not None:
                        return float(value)
                if "value" in data:
                    return float(data["value"])
    except Exception as exc:
        LOG.debug("fear/greed fetch failed: %r", exc)
    return 50.0


def _composite(news_score: float, social_score: float, fear_greed: float) -> float:
    fg_norm = (float(fear_greed) - 50.0) / 50.0
    composite = 0.5 * news_score + 0.3 * social_score + 0.2 * fg_norm
    return max(-1.0, min(1.0, composite))


class NewsPipeline:
    """Coordinates RSS ingestion, NLP enrichment and storage."""

    def __init__(self, session_factory: Optional[async_sessionmaker[AsyncSession]] = None):
        self.session_factory = session_factory or AsyncSessionLocal
        if self.session_factory is None:
            raise RuntimeError("Database session factory is not available")

    async def _enrich_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        title = item.get("title", "")
        summary = item.get("summary", "")
        try:
            enriched = await nlp_gate.analyze_news(title, summary)
        except Exception as exc:
            LOG.warning("nlp gate failed, using heuristic fallback: %r", exc)
            enriched = {
                "summary": (summary or title)[:200],
                "sentiment": 0,
                "importance": item.get("impact", "low"),
            }
        item.update(enriched)
        return item

    @staticmethod
    def _to_model(item: Dict[str, Any]) -> NewsItem:
        published_ts = item.get("published_ts")
        if published_ts:
            published_at = datetime.fromtimestamp(int(published_ts), tz=timezone.utc)
        else:
            published_at = None
        fetched_at = datetime.now(timezone.utc)
        raw_fetched = item.get("fetched_at")
        if isinstance(raw_fetched, str) and raw_fetched:
            try:
                fetched_at = datetime.fromisoformat(raw_fetched.replace("Z", "+00:00"))
            except ValueError:
                pass
        return NewsItem(
            external_id=item.get("id") or item.get("external_id"),
            title=item.get("title", "")[:510],
            link=item.get("link"),
            summary=item.get("summary"),
            published_at=published_at,
            published_ts=int(published_ts) if published_ts else None,
            fetched_at=fetched_at,
            impact=item.get("impact", "low"),
            sentiment=int(item.get("sentiment", 0) or 0),
            importance=item.get("importance", "low"),
            source=item.get("source"),
            feed=item.get("feed"),
            raw_payload=item,
        )

    def _news_score(self, items: Iterable[Dict[str, Any]]) -> float:
        total = 0.0
        weight_sum = 0.0
        for item in items:
            sentiment = float(item.get("sentiment", 0))
            weight = _weight_from_impact(str(item.get("impact", "low")), str(item.get("importance", "low")))
            total += sentiment * weight
            weight_sum += weight
        return 0.0 if weight_sum == 0 else max(-1.0, min(1.0, total / weight_sum))

    async def run(self, limit: int = 40) -> SentimentContext:
        raw_items = await rss_client.refresh_news(limit=limit)
        if not raw_items:
            LOG.info("news pipeline: no items fetched")
            return SentimentContext()

        enriched: List[Dict[str, Any]] = []
        for item in raw_items:
            try:
                enriched_item = await self._enrich_item(item)
                enriched.append(enriched_item)
            except Exception as exc:
                LOG.warning("news pipeline enrichment failed: %r", exc)

        news_score = self._news_score(enriched)
        social_score = await _fetch_social_score()
        fear_greed = await _fetch_fear_greed()
        composite_score = _composite(news_score, social_score, fear_greed)
        ctx = SentimentContext(
            news_score=news_score,
            social_score=social_score,
            fear_greed=fear_greed,
            composite_score=composite_score,
        )

        async with self.session_factory() as session:  # type: ignore[call-arg]
            models = [self._to_model(item) for item in enriched if item.get("id")]
            await upsert_news_items(session, models)  # type: ignore[arg-type]
            snapshot = SentimentSnapshot(
                bucket_ts=int(datetime.now(timezone.utc).timestamp() // 300 * 300),
                news_score=news_score,
                social_score=social_score,
                fear_greed=fear_greed,
                composite_score=composite_score,
                methodology="v1",
                extra={
                    "items": len(models),
                    "source_feeds": sorted({item.get("feed") for item in enriched if item.get("feed")}),
                },
            )
            await store_sentiment_snapshot(session, snapshot)  # type: ignore[arg-type]
            if record_audit_event is not None:
                await record_audit_event(
                    session,
                    action="news_pipeline",
                    status="ok",
                    details=f"stored {len(models)} items",
                    payload={
                        "news_score": news_score,
                        "social_score": social_score,
                        "fear_greed": fear_greed,
                        "composite": composite_score,
                    },
                )
            await session.commit()

        LOG.info(
            "news pipeline run complete: items=%d news=%.2f social=%.2f fg=%.1f composite=%.2f",
            len(enriched),
            news_score,
            social_score,
            fear_greed,
            composite_score,
        )
        return ctx


async def refresh_and_store(limit: int = 40) -> SentimentContext:
    pipeline = NewsPipeline()
    return await pipeline.run(limit=limit)


async def load_latest_sentiment(session_factory: Optional[async_sessionmaker[AsyncSession]] = None) -> Optional[SentimentContext]:
    if session_factory is None:
        session_factory = AsyncSessionLocal
    if session_factory is None or SentimentSnapshot is None:
        return None
    async with session_factory() as session:  # type: ignore[call-arg]
        from db.crud_news import latest_sentiment

        snapshot = await latest_sentiment(session)
        if snapshot is None:
            return None
        return SentimentContext(
            news_score=float(snapshot.news_score),
            social_score=float(snapshot.social_score),
            fear_greed=float(snapshot.fear_greed),
            composite_score=float(snapshot.composite_score),
            methodology=snapshot.methodology or "v1",
        )
