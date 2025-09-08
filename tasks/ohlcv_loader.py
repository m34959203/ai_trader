# ai_trader/tasks/ohlcv_loader.py
import asyncio
import logging
from httpx import AsyncClient

LOG = logging.getLogger("ai_trader.ohlcv_loader")

# Какие пары и таймфреймы хранить
PAIRS = ["BTCUSDT", "ETHUSDT"]
INTERVALS = ["1h", "15m"]

async def load_ohlcv_once():
    async with AsyncClient(base_url="http://127.0.0.1:8001") as client:
        for symbol in PAIRS:
            for interval in INTERVALS:
                try:
                    resp = await client.post(
                        "/ohlcv/prices/store",
                        json={
                            "source": "binance",
                            "symbol": symbol,
                            "timeframe": interval,
                            "limit": 500,
                            "testnet": True,
                        },
                        timeout=60.0,
                    )
                    LOG.info("Stored OHLCV %s/%s: %s", symbol, interval, resp.status_code)
                except Exception as e:
                    LOG.error("Failed to store OHLCV %s/%s: %r", symbol, interval, e)

async def background_loop():
    while True:
        await load_ohlcv_once()
        await asyncio.sleep(60 * 60)  # каждые 60 минут
