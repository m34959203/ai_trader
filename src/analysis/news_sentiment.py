"""Utilities for aggregating news sentiment into the trading signal pipeline."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from news.nlp_gate import analyze_news_item

NEWS_FILE = Path(os.getenv("NEWS_FILE", "data/news.json"))

_IMPACT_WEIGHTS = {"high": 3.0, "medium": 2.0, "med": 2.0, "low": 1.0}


@dataclass(frozen=True)
class NewsHeadline:
    id: str
    title: str
    summary: str
    link: Optional[str]
    impact: str
    importance: str
    sentiment: int
    published_ts: int

    @property
    def weight(self) -> float:
        return _IMPACT_WEIGHTS.get(self.impact.lower(), 1.0)


@dataclass(frozen=True)
class AggregatedNews:
    score: float
    normalized: float
    sentiment_label: str
    total_weight: float
    items: List[NewsHeadline]
    counts: Dict[str, int]


def _load_news() -> List[Dict[str, object]]:
    if not NEWS_FILE.exists():
        return []
    try:
        data = json.loads(NEWS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(data, list):
        return [d for d in data if isinstance(d, dict)]
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return [d for d in data["items"] if isinstance(d, dict)]
    return []


def _aliases(symbol: str, extra: Optional[Dict[str, Sequence[str]]] = None) -> List[str]:
    symbol = symbol.upper()
    base = {symbol}
    if symbol.endswith("USDT"):
        base.add(symbol[:-4])
    if "/" in symbol:
        base.update(symbol.split("/"))
    defaults: Dict[str, Sequence[str]] = {
        "BTCUSDT": ("BTC", "BITCOIN", "BTC/USD"),
        "ETHUSDT": ("ETH", "ETHEREUM", "ETH/USD"),
    }
    if extra:
        defaults.update({k.upper(): tuple(v) for k, v in extra.items()})
    for key, aliases in defaults.items():
        if key == symbol:
            base.update(a.upper() for a in aliases)
    # общие слова для крипты, если символ не найден
    if len(base) == 1:
        base.update({symbol, symbol.replace("USDT", ""), symbol.replace("/", "")})
    return sorted(base)


def _match_symbol(text: str, tokens: Iterable[str]) -> bool:
    low = text.lower()
    for token in tokens:
        tok = token.lower().strip()
        if not tok:
            continue
        if tok in low:
            return True
    return False


def collect_recent_news(
    symbol: str,
    *,
    lookback_hours: int = 24,
    extra_aliases: Optional[Dict[str, Sequence[str]]] = None,
    allow_remote: bool = True,
) -> AggregatedNews:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=float(lookback_hours))
    rows = _load_news()
    if not rows:
        return AggregatedNews(0.0, 0.0, "neutral", 0.0, [], {"positive": 0, "negative": 0, "neutral": 0})

    tokens = _aliases(symbol, extra_aliases)
    selected: List[NewsHeadline] = []
    for row in rows:
        published_ts = int(row.get("published_ts", 0) or 0)
        published_dt = datetime.fromtimestamp(published_ts, tz=timezone.utc)
        if published_dt < cutoff:
            continue
        title = str(row.get("title", "")).strip()
        summary = str(row.get("summary", "")).strip()
        if symbol and tokens and not _match_symbol(f"{title} {summary}", tokens):
            # если нет прямого совпадения — допускаем high-impact новости
            if str(row.get("impact", "")).lower() != "high":
                continue
        analysis = analyze_news_item(title, summary, allow_remote=allow_remote)
        headline = NewsHeadline(
            id=str(row.get("id", "")),
            title=title,
            summary=analysis.get("summary", summary or title),
            link=row.get("link"),
            impact=str(row.get("impact", "low")),
            importance=str(analysis.get("importance", row.get("impact", "low"))),
            sentiment=int(analysis.get("sentiment", 0)),
            published_ts=published_ts,
        )
        selected.append(headline)

    if not selected:
        return AggregatedNews(0.0, 0.0, "neutral", 0.0, [], {"positive": 0, "negative": 0, "neutral": 0})

    score = 0.0
    weight_sum = 0.0
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    for item in selected:
        w = item.weight
        weight_sum += w
        score += w * float(item.sentiment)
        if item.sentiment > 0:
            counts["positive"] += 1
        elif item.sentiment < 0:
            counts["negative"] += 1
        else:
            counts["neutral"] += 1

    normalized = 0.0 if weight_sum == 0 else max(-1.0, min(1.0, score / (weight_sum * 1.0)))
    if normalized > 0.15:
        label = "bullish"
    elif normalized < -0.15:
        label = "bearish"
    else:
        label = "neutral"

    selected.sort(key=lambda item: item.published_ts, reverse=True)
    return AggregatedNews(score, normalized, label, weight_sum, selected, counts)


__all__ = ["AggregatedNews", "NewsHeadline", "collect_recent_news"]
