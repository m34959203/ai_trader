import os
import json
from fastapi.testclient import TestClient

from src.main import app  # путь оставь как у тебя

client = TestClient(app)


def test_sim_open_and_positions():
    r = client.post("/exec/open?mode=sim&testnet=true&symbol=BTCUSDT&side=buy&type=market&qty=0.002")
    assert r.status_code == 200, r.text
    data = r.json()
    assert "order_id" in data

    r2 = client.get("/exec/positions?mode=sim&testnet=true")
    assert r2.status_code == 200
    positions = r2.json()
    assert isinstance(positions, list)


def test_sim_balance():
    r = client.get("/exec/balance?mode=sim&testnet=true")
    assert r.status_code == 200
    data = r.json()
    assert data["exchange"] == "sim"
    assert "free" in data


def test_open_requires_params():
    r = client.post("/exec/open?mode=binance&testnet=true")  # без обязательных
    assert r.status_code in (400, 422)


def test_sim_stop_limit_order():
    r = client.post(
        "/exec/open",
        params={
            "mode": "sim",
            "testnet": "true",
            "type": "stop_limit",
            "symbol": "BTCUSDT",
            "side": "buy",
            "qty": 0.001,
            "price": 61000,
            "stop_price": 60500,
        },
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["type"] == "stop_limit"
    assert data["stop_price"] == 60500
