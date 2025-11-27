# ⚡ ACTION PLAN: Критические исправления

## 🔥 ДЕНЬ 1: SECURITY (4-6 часов)

### Задача 1.1: Удалить API ключи из git (30 мин)
```bash
# Немедленно выполнить:
cd /home/user/ai_trader

# 1. Удалить из индекса
git rm --cached configs/.env

# 2. Добавить в .gitignore
cat >> .gitignore << 'EOF'
# Secrets
configs/.env*
.env*
*.pem
*.key
secrets/
EOF

# 3. Коммит
git add .gitignore
git commit -m "security: Remove .env from git and add to .gitignore"

# 4. ВАЖНО: Очистить историю
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch configs/.env' \
  --prune-empty -- --all

# 5. Force push (ОСТОРОЖНО!)
# git push origin --force --all
```

### Задача 1.2: Сменить API ключи (15 мин)
```
⚠️ РУЧНОЕ ДЕЙСТВИЕ:
1. Зайти на https://www.binance.com/en/my/settings/api-management
2. Удалить старый API key
3. Создать новый с ограничениями:
   - ✅ Enable Reading
   - ✅ Enable Spot & Margin Trading
   - ❌ Disable Withdrawals
   - ❌ Disable Internal Transfer
4. Сохранить новый ключ в безопасном месте (1Password/LastPass)
```

### Задача 1.3: Удалить LocalHSM (1 час)
```python
# Файл: services/security/vault.py

# УДАЛИТЬ строки 25-41:
# class LocalHSM: ...

# ЗАМЕНИТЬ НА:
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import secrets

class SecureHSM:
    """AES-GCM based HSM (cryptographically secure)."""

    def __init__(self, key_path: Path):
        self._key = self._load_or_generate_key(key_path)

    def _load_or_generate_key(self, path: Path) -> bytes:
        if path.exists():
            key = path.read_bytes()
            if len(key) != 32:
                raise ValueError("Invalid key length")
            return key

        # Generate new 256-bit key
        key = AESGCM.generate_key(bit_length=256)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(key)
        path.chmod(0o600)  # read-only for owner
        return key

    def encrypt(self, data: bytes) -> bytes:
        aesgcm = AESGCM(self._key)
        nonce = secrets.token_bytes(12)
        ciphertext = aesgcm.encrypt(nonce, data, None)
        return nonce + ciphertext

    def decrypt(self, data: bytes) -> bytes:
        aesgcm = AESGCM(self._key)
        nonce, ciphertext = data[:12], data[12:]
        return aesgcm.decrypt(nonce, ciphertext, None)

# Установить зависимость:
pip install cryptography>=41.0.0
```

### Задача 1.4: Добавить API authentication (2 часа)
```python
# Файл: routers/auth.py (создать новый)

from fastapi import Header, HTTPException, Depends
import secrets
import os

async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    """Verify API key from X-API-Key header."""
    expected = os.getenv("API_KEY_SECRET")

    if not expected:
        raise HTTPException(500, "API_KEY_SECRET not configured")

    if not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(403, "Invalid API key")

    return x_api_key

# Применить ко всем критичным endpoints:
# routers/trading.py
from routers.auth import verify_api_key

@router.post("/trade")
async def place_trade(
    request: OrderRequest,
    _: str = Depends(verify_api_key),  # ← добавить
):
    ...

# routers/live_trading.py
@router.post("/live/trade")
async def live_trade(
    request: OrderRequest,
    _: str = Depends(verify_api_key),  # ← добавить
):
    ...

# Создать .env с новым ключом:
# API_KEY_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
```

### Задача 1.5: Rate limiting (1 час)
```bash
# Установить
pip install slowapi

# Файл: src/main.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# В каждом router:
from fastapi import Request

@router.post("/trade")
@app.state.limiter.limit("10/minute")
async def place_trade(request: Request, ...):
    ...
```

### Задача 1.6: Ограничить CORS (15 мин)
```python
# src/main.py
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://yourdomain.com"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # ← вместо ["*"]
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # ← вместо ["*"]
    allow_headers=["Content-Type", "X-API-Key"],
)
```

**✅ Checklist День 1:**
- [ ] Удалить .env из git
- [ ] Сменить API ключи на Binance
- [ ] Удалить LocalHSM, добавить SecureHSM
- [ ] Добавить API key verification
- [ ] Добавить rate limiting
- [ ] Ограничить CORS

---

## 🛠️ ДЕНЬ 2: ERROR HANDLING (4 часа)

### Задача 2.1: Исправить broad exceptions (2 часа)
```bash
# Найти все проблемные места:
grep -rn "except Exception:" src/ services/ > broad_exceptions.txt

# Исправить каждый случай:
# ❌ БЫЛО:
try:
    ...
except Exception:
    pass

# ✅ СТАЛО:
try:
    ...
except (SpecificError1, SpecificError2) as e:
    LOG.warning("Expected error: %s", e)
except Exception as e:
    LOG.error("Unexpected error: %s", e, exc_info=True)
    raise
```

### Задача 2.2: Добавить timeouts (1 час)
```python
# Файл: services/broker_gateway.py

import httpx

class BinanceBrokerGateway:
    def __init__(self):
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0, connect=10.0),  # ← добавить
            limits=httpx.Limits(max_connections=10),
        )

    async def submit_order(self, request: OrderRequest):
        try:
            response = await self._client.post(
                url,
                json=data,
                timeout=30.0,  # ← explicit timeout
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            LOG.error("Broker timeout for %s", request.symbol)
            raise BrokerGatewayError("Timeout")
        except httpx.HTTPStatusError as e:
            LOG.error("HTTP %d: %s", e.response.status_code, e)
            raise BrokerGatewayError(f"HTTP {e.response.status_code}")
```

### Задача 2.3: Добавить retry с jitter (1 час)
```bash
pip install tenacity

# Использовать:
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(httpx.TimeoutException),
)
async def fetch_with_retry(url: str):
    return await client.get(url)
```

**✅ Checklist День 2:**
- [ ] Исправить все broad exceptions
- [ ] Добавить explicit timeouts
- [ ] Добавить retry с jitter
- [ ] Улучшить логирование ошибок

---

## 🚀 ДЕНЬ 3: TESTING & PERFORMANCE (6 часов)

### Задача 3.1: Настроить pytest coverage (30 мин)
```ini
# pytest.ini
[tool:pytest]
testpaths = tests
addopts =
    -v
    --cov=src
    --cov=services
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=70
    --asyncio-mode=auto

[coverage:run]
branch = True
omit =
    */tests/*
    */conftest.py
    */__init__.py
```

### Задача 3.2: Добавить Redis caching (2 часа)
```python
# services/cache.py (создать новый)
import redis.asyncio as redis
import json
from functools import wraps

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    decode_responses=True,
)

def cache_result(ttl: int = 300):
    """Cache function result in Redis."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{func.__name__}:{args}:{kwargs}"

            # Try cache
            cached = await redis_client.get(cache_key)
            if cached:
                return json.loads(cached)

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            await redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result, default=str),
            )

            return result
        return wrapper
    return decorator

# Применить:
@cache_result(ttl=300)
async def get_ohlcv(symbol: str, timeframe: str):
    return await crud.get_ohlcv(symbol, timeframe)
```

### Задача 3.3: Streaming для больших датасетов (2 часа)
```python
# routers/trading.py

from typing import Iterator

def _rows_to_df_chunked(
    rows: Iterable[Any],
    chunk_size: int = 10_000
) -> Iterator[pd.DataFrame]:
    """Stream data in chunks."""
    chunk = []
    for r in rows:
        chunk.append({...})
        if len(chunk) >= chunk_size:
            yield pd.DataFrame(chunk)
            chunk = []
    if chunk:
        yield pd.DataFrame(chunk)

@router.get("/ohlcv/stream")
async def stream_ohlcv(symbol: str, tf: str):
    """Stream OHLCV in chunks (NDJSON)."""
    rows = await crud.get_ohlcv(symbol, tf, limit=1_000_000)

    async def generate():
        for chunk in _rows_to_df_chunked(rows):
            yield chunk.to_json(orient='records') + '\n'

    return StreamingResponse(
        generate(),
        media_type='application/x-ndjson',
    )
```

### Задача 3.4: Добавить integration tests (1.5 часа)
```python
# tests/test_trading_integration.py
import pytest

@pytest.mark.asyncio
async def test_full_trading_flow():
    """Test complete trading flow end-to-end."""
    # 1. Fetch OHLCV
    df = await get_ohlcv("BTCUSDT", "1h")
    assert len(df) > 0

    # 2. Generate signal
    signal = await generate_signal(df)
    assert signal.signal in ["buy", "sell", "hold"]

    # 3. Calculate position size
    size = calculate_position_size(signal, balance=10000)
    assert 0 < size < 10000

    # 4. Execute trade (simulated)
    result = await execute_trade(signal, size)
    assert result.status == "success"
```

**✅ Checklist День 3:**
- [ ] Настроить pytest coverage
- [ ] Добавить Redis caching
- [ ] Реализовать streaming
- [ ] Добавить 5+ integration tests
- [ ] Запустить tests и достичь 70%+ coverage

---

## 📊 ПРОВЕРКА ПРОГРЕССА

После каждого дня запускать:

```bash
#!/bin/bash
# scripts/check_progress.sh

echo "🔍 Проверка прогресса..."

# Security scan
echo "1. Security scan..."
pip install bandit
bandit -r src/ services/ -f json -o reports/security.json
echo "   Report: reports/security.json"

# Test coverage
echo "2. Test coverage..."
pytest tests/ \
    --cov=src \
    --cov=services \
    --cov-report=html \
    --cov-report=term
echo "   Report: htmlcov/index.html"

# Code quality
echo "3. Code quality..."
pip install pylint
pylint src/ services/ --output-format=json > reports/pylint.json
echo "   Report: reports/pylint.json"

# Type checking
echo "4. Type checking..."
pip install mypy
mypy src/ services/ --ignore-missing-imports > reports/mypy.txt
echo "   Report: reports/mypy.txt"

echo "✅ All checks complete!"
echo "Next: Review reports/ directory"
```

---

## 🎯 КРИТЕРИИ УСПЕХА

После 3 дней работы:

| Метрика | Было | Цель | Проверка |
|---------|------|------|----------|
| Security Score | 3/10 | 8/10 | `bandit -r src/` |
| Test Coverage | ~30% | 70%+ | `pytest --cov` |
| API Auth | ❌ | ✅ | Попробовать запрос без API key |
| Broad Exceptions | 37 | 0 | `grep -r "except Exception:" src/` |
| Rate Limiting | ❌ | ✅ | Отправить 20 запросов за минуту |
| CORS Security | allow_origins=["*"] | Whitelist | Check `main.py:250` |

---

## 📞 ПОМОЩЬ

Если возникли вопросы:

1. **Security**: См. `doc/expert_audit_2025.md` секция "SECURITY УЯЗВИМОСТИ"
2. **ML Models**: См. `doc/implementation_roadmap.md` секция "ЭТАП 1"
3. **Testing**: См. примеры в `tests/test_*.py`
4. **Performance**: См. audit секция "PERFORMANCE BOTTLENECKS"

---

**Создано**: 2025-11-27
**Приоритет**: 🔴 КРИТИЧЕСКИЙ
**Deadline**: 3 дня
