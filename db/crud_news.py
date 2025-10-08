from __future__ import annotations

"""CRUD helpers for news intelligence tables."""

from typing import Iterable, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .models_intel import NewsItem, SentimentSnapshot


async def upsert_news_items(session: AsyncSession, items: Iterable[NewsItem]) -> int:
    """Insert or update news items by external_id."""
    count = 0
    for item in items:
        stmt = select(NewsItem).where(NewsItem.external_id == item.external_id).limit(1)
        existing = await session.scalar(stmt)
        if existing:
            existing.title = item.title
            existing.summary = item.summary
            existing.link = item.link
            existing.published_at = item.published_at
            existing.published_ts = item.published_ts
            existing.impact = item.impact
            existing.sentiment = item.sentiment
            existing.importance = item.importance
            existing.source = item.source
            existing.feed = item.feed
            existing.raw_payload = item.raw_payload
        else:
            session.add(item)
        count += 1
    return count


async def store_sentiment_snapshot(session: AsyncSession, snapshot: SentimentSnapshot) -> None:
    """Insert or replace snapshot for a time bucket."""
    stmt = select(SentimentSnapshot).where(SentimentSnapshot.bucket_ts == snapshot.bucket_ts).limit(1)
    existing = await session.scalar(stmt)
    if existing:
        existing.news_score = snapshot.news_score
        existing.social_score = snapshot.social_score
        existing.fear_greed = snapshot.fear_greed
        existing.composite_score = snapshot.composite_score
        existing.methodology = snapshot.methodology
        existing.extra = snapshot.extra
    else:
        session.add(snapshot)


async def latest_sentiment(session: AsyncSession) -> Optional[SentimentSnapshot]:
    stmt = select(SentimentSnapshot).order_by(SentimentSnapshot.bucket_ts.desc()).limit(1)
    return await session.scalar(stmt)


async def latest_news(session: AsyncSession, limit: int = 20) -> List[NewsItem]:
    stmt = select(NewsItem).order_by(NewsItem.published_ts.desc().nullslast(), NewsItem.id.desc()).limit(limit)
    rows = await session.scalars(stmt)
    return list(rows)
