from __future__ import annotations

"""Additional database models covering market intelligence, sentiment and audit."""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import String, Text, Integer, BigInteger, Float, Index, UniqueConstraint, JSON, func
from sqlalchemy.orm import Mapped, mapped_column

from .session import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class NewsItem(Base):
    """Normalized news item enriched with NLP sentiment."""

    __tablename__ = "news_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    external_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    link: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    published_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    published_ts: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(nullable=False, default=_utcnow, server_default=func.now())
    impact: Mapped[str] = mapped_column(String(16), nullable=False, default="low")
    sentiment: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    importance: Mapped[str] = mapped_column(String(16), nullable=False, default="low")
    source: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    feed: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    raw_payload: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, default=_utcnow, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
        server_default=func.now(),
    )

    __table_args__ = (
        Index("ix_news_items_published", "published_ts"),
    )


class SentimentSnapshot(Base):
    """Aggregated sentiment metrics for market overlays."""

    __tablename__ = "sentiment_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bucket_ts: Mapped[int] = mapped_column(BigInteger, nullable=False)
    news_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    social_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    fear_greed: Mapped[float] = mapped_column(Float, nullable=False, default=50.0)
    composite_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    methodology: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    extra: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, default=_utcnow, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("bucket_ts", name="uq_sentiment_bucket"),
        Index("ix_sentiment_bucket", "bucket_ts"),
    )


class AuditLog(Base):
    """Tracks operational actions for accountability."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(nullable=False, default=_utcnow, server_default=func.now())
    actor: Mapped[str] = mapped_column(String(64), nullable=False, default="system")
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    scope: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="ok")
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    payload: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("ix_audit_ts", "ts"),
        Index("ix_audit_action", "action"),
    )


__all__ = ["NewsItem", "SentimentSnapshot", "AuditLog"]
