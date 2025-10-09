# news/nlp_gate.py
from __future__ import annotations

import os
import re
import json
import time
import hashlib
import pathlib
from typing import Dict, Any, Optional, Tuple

import httpx

# ──────────────────────────────────────────────────────────────────────────────
# Конфиг
# ──────────────────────────────────────────────────────────────────────────────
CACHE_FILE = pathlib.Path(os.getenv("NEWS_NLP_CACHE_FILE", "data/news_nlp_cache.json"))
CACHE_MAX_ITEMS = int(os.getenv("NEWS_NLP_CACHE_MAX", "2000"))
CACHE_TTL_SEC = int(os.getenv("NEWS_NLP_CACHE_TTL", str(7 * 24 * 60 * 60)))  # 7 дней по умолчанию

OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-3-12b-it:free")
OPENROUTER_BASE = os.getenv("OPENROUTER_API_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "https://localhost")  # рекомендуется OpenRouter

HTTP_TIMEOUT = float(os.getenv("NEWS_NLP_HTTP_TIMEOUT", "40.0"))
HTTP_RETRIES = int(os.getenv("NEWS_NLP_HTTP_RETRIES", "2"))
HTTP_BACKOFF_BASE = float(os.getenv("NEWS_NLP_BACKOFF_BASE", "0.7"))
LANG = os.getenv("NEWS_NLP_LANG", "ru")  # язык краткого резюме, по умолчанию RU

# ──────────────────────────────────────────────────────────────────────────────
# Кэш (в памяти + файл)
# ──────────────────────────────────────────────────────────────────────────────
def _load_cache() -> Dict[str, Any]:
    if not CACHE_FILE.exists():
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        return {}
    try:
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}

CACHE: Dict[str, Any] = _load_cache()


def _touch_cache(key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    now = int(time.time())
    payload = dict(payload)
    payload.setdefault("_ts", now)
    payload["_last_ts"] = now
    CACHE[key] = payload
    try:
        _persist_cache()
    except Exception:
        pass
    return payload

def _persist_cache() -> None:
    try:
        # лёгкая «уборка» просроченных и ограничение размера
        now = int(time.time())
        items = []
        for k, v in CACHE.items():
            ts = int(v.get("_ts", 0))
            if CACHE_TTL_SEC > 0 and (now - ts) > CACHE_TTL_SEC:
                continue
            items.append((k, v))
        # сортировка по используемости (last_ts ↓)
        items.sort(key=lambda kv: int(kv[1].get("_last_ts", kv[1].get("_ts", 0))), reverse=True)
        if CACHE_MAX_ITEMS > 0 and len(items) > CACHE_MAX_ITEMS:
            items = items[:CACHE_MAX_ITEMS]
        data = {k: v for k, v in items}
        CACHE_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # не мешаем основному потоку
        pass

def _cache_key(title: str, summary: str) -> str:
    key_src = f"{title}\n::\n{summary}"
    return hashlib.sha256(key_src.encode("utf-8")).hexdigest()

# ──────────────────────────────────────────────────────────────────────────────
# Валидация и нормализация результата
# ──────────────────────────────────────────────────────────────────────────────
_IMPORTANCE_MAP = {"low": "low", "med": "med", "medium": "med", "high": "high", "hi": "high"}

def _normalize_result(obj: Dict[str, Any], fallback_summary: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    # summary
    s = str(obj.get("summary") or obj.get("brief") or fallback_summary or "").strip()
    # ограничим длину на всякий
    out["summary"] = s[:280] if len(s) > 280 else s

    # sentiment → {-1,0,1}
    try:
        sent = int(obj.get("sentiment", 0))
    except Exception:
        sent = 0
    if sent < -1:
        sent = -1
    if sent > 1:
        sent = 1
    out["sentiment"] = sent

    # importance → {"low","med","high"}
    imp_raw = str(obj.get("importance", "low")).strip().lower()
    out["importance"] = _IMPORTANCE_MAP.get(imp_raw, "low")

    return out

def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    Пытаемся вытащить JSON даже если модель прислала лишний текст.
    """
    if not text:
        return None
    # быстрый путь: сразу json
    try:
        return json.loads(text)
    except Exception:
        pass
    # поищем первый JSON-объект в тексте
    m = re.search(r"\{(?:[^{}]|(?R))*\}", text, flags=re.DOTALL)  # рекурсивный шаблон поддерживается не везде
    if not m:
        # запасной простой поиск «самого большого» блока
        braces = []
        start = -1
        best: Optional[str] = None
        for i, ch in enumerate(text):
            if ch == "{":
                if not braces:
                    start = i
                braces.append("{")
            elif ch == "}":
                if braces:
                    braces.pop()
                    if not braces and start != -1:
                        candidate = text[start : i + 1]
                        if not best or len(candidate) > len(best):
                            best = candidate
        if best:
            try:
                return json.loads(best)
            except Exception:
                return None
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# Эвристический фолбэк (без API)
# ──────────────────────────────────────────────────────────────────────────────
_POS = ("rise", "rally", "bull", "bullish", "up", "gain", "beat", "approve", "approval", "ETF approval")
_NEG = ("fall", "drop", "bear", "bearish", "down", "loss", "miss", "lawsuit", "hack", "exploit", "ban", "halt", "crash")
_HI  = ("fomc", "cpi", "nfp", "interest rate", "sec", "etf", "blackrock", "ecb", "boj")

def _heuristic(title: str, summary: str) -> Dict[str, Any]:
    txt = f"{title} {summary}".lower()
    sentiment = 0
    if any(w in txt for w in _POS):
        sentiment = 1
    if any(w in txt for w in _NEG):
        sentiment = -1
    importance = "high" if any(w in txt for w in _HI) else ("med" if ("inflation" in txt or "recession" in txt) else "low")
    return {"summary": (summary or title)[:200], "sentiment": sentiment, "importance": importance}

# ──────────────────────────────────────────────────────────────────────────────
# HTTP клиент к OpenRouter
# ──────────────────────────────────────────────────────────────────────────────
def _headers() -> Dict[str, str]:
    h = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
    }
    # OpenRouter рекомендует передавать Referer/Origin (для аналитики)
    if OPENROUTER_REFERER:
        h["HTTP-Referer"] = OPENROUTER_REFERER
        h["X-Title"] = "AI-Trader"
    return h

async def _call_openrouter(prompt: str, model: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if not OPENROUTER_KEY:
        return None

    payload = {
        "model": model or OPENROUTER_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise financial news summarizer. "
                    "Return STRICT JSON only. No prose. "
                    "Schema: {\"summary\": string, \"sentiment\": -1|0|1, \"importance\": \"low|med|high\"}. "
                    "If unsure, choose sentiment=0, importance=low."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Language: {LANG}\n"
                    "Task: Summarize in ONE short sentence. "
                    "Then classify sentiment (-1,0,1) and importance (low/med/high).\n\n"
                    f"Title: {prompt[:2000]}\n"
                ),
            },
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},  # многие модели уважают это поле
    }

    last_err = None
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        for attempt in range(HTTP_RETRIES + 1):
            try:
                resp = await client.post(f"{OPENROUTER_BASE}/chat/completions", headers=_headers(), json=payload)
                if resp.status_code in (429, 503, 502, 500):
                    # бэкофф и повтор
                    await _sleep(min(HTTP_BACKOFF_BASE * (2 ** attempt), 5.0))
                    last_err = (resp.status_code, resp.text[:200])
                    continue
                resp.raise_for_status()
                data = resp.json()
                content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
                obj = _extract_json_block(content) or {}
                return obj
            except Exception as e:
                last_err = e
                await _sleep(min(HTTP_BACKOFF_BASE * (2 ** attempt), 5.0))
    # финальная ошибка в лог не бросаем (чтобы не шуметь) — вернём None
    return None

async def _sleep(seconds: float) -> None:
    import asyncio
    await asyncio.sleep(seconds)

# ──────────────────────────────────────────────────────────────────────────────
# Публичное API
# ──────────────────────────────────────────────────────────────────────────────
async def analyze_news(title: str, summary: str, *, model: Optional[str] = None) -> Dict[str, Any]:
    """
    Возвращает строго:
      {
        "summary": str,     # 1 короткое предложение на выбранном языке (ENV NEWS_NLP_LANG)
        "sentiment": -1|0|1,
        "importance": "low"|"med"|"high"
      }
    Кэширует результат (ключ = SHA256(title+summary)).
    """
    key = _cache_key(title, summary)

    # кэш-хит?
    if key in CACHE:
        item = CACHE[key]
        # TTL проверим по _ts
        now = int(time.time())
        if CACHE_TTL_SEC <= 0 or (now - int(item.get("_ts", 0))) <= CACHE_TTL_SEC:
            # отметим последнюю выдачу для LRU
            item["_last_ts"] = now
            return {k: item[k] for k in ("summary", "sentiment", "importance") if k in item}

    # без ключа → эвристика
    if not OPENROUTER_KEY:
        parsed = _heuristic(title, summary)
        now = int(time.time())
        CACHE[key] = {**parsed, "_ts": now, "_last_ts": now}
        _persist_cache()
        return parsed

    # запрос к OpenRouter
    prompt = f"{title}\n\n{summary}"
    obj = await _call_openrouter(prompt, model=model)

    if obj is None:
        # фолбэк на эвристику при сетевых проблемах/429
        parsed = _heuristic(title, summary)
    else:
        parsed = _normalize_result(obj, fallback_summary=(summary or title))

    # сохранить в кэш
    now = int(time.time())
    CACHE[key] = {**parsed, "_ts": now, "_last_ts": now}
    _persist_cache()
    return parsed


def analyze_news_item(
    title: str,
    summary: str = "",
    *,
    allow_remote: bool = True,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Синхронный хелпер вокруг :func:`analyze_news`."""

    key = _cache_key(title, summary)
    cached = CACHE.get(key)
    now = int(time.time())
    if isinstance(cached, dict):
        if CACHE_TTL_SEC <= 0 or (now - int(cached.get("_ts", 0))) <= CACHE_TTL_SEC:
            cached["_last_ts"] = now
            return {
                "summary": str(cached.get("summary", summary or title)),
                "sentiment": int(cached.get("sentiment", 0)),
                "importance": str(cached.get("importance", "low")),
            }

    result: Optional[Dict[str, Any]] = None
    if allow_remote and OPENROUTER_KEY:
        try:
            import asyncio

            result = asyncio.run(analyze_news(title, summary, model=model))
        except RuntimeError:
            # если уже есть запущенный цикл — фолбэк к эвристике (чтобы не блокировать)
            result = None
        except Exception:
            result = None

    if not result:
        result = _heuristic(title, summary)

    stored = _touch_cache(key, result)
    return {
        "summary": str(stored.get("summary", summary or title)),
        "sentiment": int(stored.get("sentiment", 0)),
        "importance": str(stored.get("importance", "low")),
    }

# удобный синхронный запуск на случай CLI-отладки
if __name__ == "__main__":
    import asyncio
    demo_title = "FOMC keeps interest rate unchanged, signals possible cut later this year"
    demo_summary = "Federal Reserve held the benchmark rate; comments suggest easing if inflation cools."
    res = asyncio.run(analyze_news(demo_title, demo_summary))
    print(json.dumps(res, ensure_ascii=False, indent=2))
