# news/rss_client.py
from __future__ import annotations

import os
import json
import time
import hmac
import hashlib
import pathlib
from typing import List, Dict, Any, Optional, Iterable, Tuple
from contextlib import asynccontextmanager

import httpx
import feedparser
from email.utils import parsedate_to_datetime

# ──────────────────────────────────────────────────────────────────────────────
# Конфиг
# ──────────────────────────────────────────────────────────────────────────────
NEWS_FILE = pathlib.Path(os.getenv("NEWS_FILE", "data/news.json"))
NEWS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Можно переопределить через ENV: NEWS_FEEDS="url1,url2,..."
DEFAULT_FEEDS = [
    "https://www.coindesk.com/arc/outboundfeeds/rss/",   # Крипто
    "https://www.investing.com/rss/news_25.rss",         # Рынки
    "https://www.forexfactory.com/ffcal_week_this.xml",  # Макро (XML, но feedparser умеет)
]
FEEDS: List[str] = [
    u for u in (os.getenv("NEWS_FEEDS", "") or "").split(",") if u.strip()
] or DEFAULT_FEEDS

# Ключевые слова (весовое ранжирование)
IMPACT_KEYWORDS: List[Tuple[str, int]] = [
    ("FOMC", 5), ("rate hike", 5), ("rate cut", 5), ("interest rate", 5),
    ("CPI", 5), ("NFP", 5), ("nonfarm payroll", 5), ("PPI", 4),
    ("ETF approval", 5), ("SEC", 4), ("lawsuit", 4),
    ("hack", 5), ("exploit", 4), ("downtime", 3),
    ("inflation", 4), ("recession", 4), ("liquidation", 3),
    ("blackrock", 3), ("spot etf", 4),
    ("binance", 2), ("coinbase", 2), ("okx", 2), ("kraken", 2),
    ("fed", 4), ("ecb", 3), ("boj", 3),
]
# Порог для уровней
IMPACT_THRESHOLDS = {
    "high": 5,
    "medium": 3,
}

USER_AGENT = os.getenv("NEWS_USER_AGENT", "AI-TraderRSS/1.0 (+https://example.local)")
HTTP_TIMEOUT = float(os.getenv("NEWS_HTTP_TIMEOUT", "12.0"))
HTTP_RETRIES = int(os.getenv("NEWS_HTTP_RETRIES", "2"))
MAX_ITEMS_TO_KEEP = int(os.getenv("NEWS_MAX_ITEMS", "800"))  # ограничим историю

# ──────────────────────────────────────────────────────────────────────────────
# Утилиты
# ──────────────────────────────────────────────────────────────────────────────
def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _to_iso(dt_str: Optional[str], fallback_struct: Optional[time.struct_time] = None) -> Tuple[str, int]:
    """
    Нормализуем published/updated в ISO-8601 (UTC) + epoch (сек).
    Пробуем:
      1) email.utils.parsedate_to_datetime
      2) entry.published_parsed (struct_time)
      3) fallback на текущее время
    """
    epoch = int(time.time())
    if dt_str:
        try:
            dt = parsedate_to_datetime(dt_str)
            if dt.tzinfo:
                epoch = int(dt.timestamp())
            else:
                # считаем строку UTC, если без TZ
                epoch = int(time.mktime(dt.timetuple()))
        except Exception:
            pass
    elif fallback_struct:
        try:
            epoch = int(time.mktime(fallback_struct))
        except Exception:
            pass

    iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))
    return iso, epoch

def _source_from_link(link: Optional[str]) -> Optional[str]:
    if not link:
        return None
    try:
        from urllib.parse import urlparse
        return urlparse(link).netloc
    except Exception:
        return None

def _impact_score(text: str) -> int:
    t = text.lower()
    score = 0
    for kw, w in IMPACT_KEYWORDS:
        if kw.lower() in t:
            score = max(score, w)  # берём максимум, а не сумму (чтобы не раздувать)
    # простая эвристика: "breaking" → высокий сигнал
    if "breaking" in t or "urgent" in t:
        score = max(score, 5)
    return score

def guess_impact(title: str, summary: str) -> str:
    score = _impact_score(f"{title} {summary}")
    if score >= IMPACT_THRESHOLDS["high"]:
        return "high"
    if score >= IMPACT_THRESHOLDS["medium"]:
        return "medium"
    return "low"

def _entry_id(link: Optional[str], title: str) -> str:
    """
    Устойчивый id для дедупликации: по ссылке, иначе по заголовку.
    """
    base = (link or "").strip() or title.strip()
    return _sha1(base.lower())

def _load_existing() -> List[Dict[str, Any]]:
    if not NEWS_FILE.exists():
        return []
    try:
        with NEWS_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            # возможен старый формат { items: [...] }
            if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
                return data["items"]
    except Exception:
        pass
    return []

# ──────────────────────────────────────────────────────────────────────────────
# HTTP клиент
# ──────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def _client() -> httpx.AsyncClient:
    headers = {"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8"}
    async with httpx.AsyncClient(headers=headers, timeout=HTTP_TIMEOUT, http2=True, follow_redirects=True) as c:
        yield c

async def _fetch_raw(c: httpx.AsyncClient, url: str) -> Optional[bytes]:
    last_err = None
    for attempt in range(1, HTTP_RETRIES + 2):  # 1 + retries
        try:
            resp = await c.get(url)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            last_err = e
            await _sleep(min(0.5 * attempt, 2.0))
    print(f"[rss] failed to fetch: {url} err={last_err!r}")
    return None

async def _sleep(seconds: float) -> None:
    import asyncio
    await asyncio.sleep(seconds)

# ──────────────────────────────────────────────────────────────────────────────
# Основная логика
# ──────────────────────────────────────────────────────────────────────────────
async def _parse_one(url: str, limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    async with _client() as c:
        raw = await _fetch_raw(c, url)
    if raw is None:
        return out

    # feedparser умеет читать bytes
    feed = feedparser.parse(raw)

    for entry in feed.entries[:limit]:
        title = entry.get("title", "").strip()
        link = (entry.get("link") or "").strip()
        summary = entry.get("summary", "").strip()
        published = entry.get("published")
        updated = entry.get("updated")
        # struct_time если есть
        published_struct = entry.get("published_parsed") or entry.get("updated_parsed")

        iso, epoch = _to_iso(published or updated, published_struct)
        item = {
            "id": _entry_id(link, title),
            "title": title,
            "link": link or None,
            "published": iso,
            "published_ts": epoch,
            "summary": summary,
            "impact": guess_impact(title, summary),
            "source": _source_from_link(link),
            "feed": url,
            "fetched_at": _now_iso(),
        }
        out.append(item)
    return out

async def parse_feeds(limit: int = 50, feeds: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    urls = list(feeds or FEEDS)
    all_items: List[Dict[str, Any]] = []
    for url in urls:
        try:
            items = await _parse_one(url, limit)
            all_items.extend(items)
        except Exception as e:
            print(f"[rss] parse failed {url}: {e!r}")
    return all_items

def _merge_dedup(old: List[Dict[str, Any]], new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {it.get("id"): it for it in old if it.get("id")}
    for it in new:
        iid = it.get("id")
        if not iid:
            continue
        # policy: новый поверх старого
        index[iid] = it
    merged = list(index.values())
    # сортировка по времени публикации (потом — по fetched_at)
    merged.sort(key=lambda x: (x.get("published_ts", 0), x.get("fetched_at", "")), reverse=True)
    # ограничение истории
    if len(merged) > MAX_ITEMS_TO_KEEP:
        merged = merged[:MAX_ITEMS_TO_KEEP]
    return merged

async def refresh_news(limit: int = 30) -> List[Dict[str, Any]]:
    old = _load_existing()
    new = await parse_feeds(limit=limit, feeds=FEEDS)
    merged = _merge_dedup(old, new)
    NEWS_FILE.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    return merged

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import asyncio
    merged = asyncio.run(refresh_news(limit=int(os.getenv("NEWS_LIMIT", "30"))))
    print(f"Saved {len(merged)} news items → {NEWS_FILE}")
