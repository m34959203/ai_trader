from __future__ import annotations

import asyncio

import pytest


@pytest.mark.asyncio
async def test_autopilot_start_stop_cycle(client, monkeypatch):
    from routers import autopilot

    if not getattr(autopilot, "_HAS_AUTO_TRADER_RUNTIME", False):  # pragma: no cover - зависимость не установлена
        pytest.skip("auto trader runtime unavailable")

    runtime = {
        "running": False,
        "config": autopilot.get_config().to_public_dict(),
    }

    started = asyncio.Event()

    async def fake_background_loop(*, config, state):
        runtime["running"] = True
        runtime["config"] = config.to_public_dict()
        state.iteration += 1
        started.set()
        try:
            while True:
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            runtime["running"] = False
            raise

    monkeypatch.setattr(autopilot, "background_loop", fake_background_loop)
    monkeypatch.setattr(autopilot, "get_runtime_status", lambda: dict(runtime))

    original_cfg = autopilot.get_config()
    try:
        resp = await client.post(
            "/autopilot/start",
            json={"symbols": ["ETHUSDT"], "loop_sec": 5.0, "dry_run": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "started"
        assert "ETHUSDT" in data["config"]["symbols"]

        await started.wait()
        status = await client.get("/autopilot/status")
        status_json = status.json()
        assert status_json["running"] is True
        assert "ETHUSDT" in status_json["auto_trader"].get("config", {}).get("symbols", [])

        stop = await client.post("/autopilot/stop")
        assert stop.status_code == 200
        assert stop.json()["status"] == "stopped"

        status_after = await client.get("/autopilot/status")
        assert status_after.json()["running"] is False
    finally:
        await client.post("/autopilot/stop")
        autopilot.set_config(original_cfg)


@pytest.mark.asyncio
async def test_autopilot_config_restart(client, monkeypatch):
    from routers import autopilot

    if not getattr(autopilot, "_HAS_AUTO_TRADER_RUNTIME", False):  # pragma: no cover
        pytest.skip("auto trader runtime unavailable")

    runtime = {"running": False, "config": autopilot.get_config().to_public_dict(), "runs": 0}

    started = asyncio.Event()

    async def fake_background_loop(*, config, state):
        runtime["running"] = True
        runtime["config"] = config.to_public_dict()
        runtime["runs"] += 1
        started.set()
        try:
            while True:
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            runtime["running"] = False
            raise

    monkeypatch.setattr(autopilot, "background_loop", fake_background_loop)
    monkeypatch.setattr(autopilot, "get_runtime_status", lambda: dict(runtime))

    original_cfg = autopilot.get_config()
    try:
        start = await client.post("/autopilot/start", json={"dry_run": True})
        assert start.status_code == 200
        await started.wait()
        assert runtime["runs"] == 1

        started.clear()
        update = await client.put(
            "/autopilot/config?restart=true",
            json={"quote_usdt": 25.0},
        )
        assert update.status_code == 200
        await started.wait()
        assert pytest.approx(runtime["config"]["quote_usdt"]) == 25.0
        assert runtime["runs"] == 2

        # Запрос без restart не создаёт новый цикл
        update_no_restart = await client.put(
            "/autopilot/config",
            json={"quote_usdt": 30.0},
        )
        assert update_no_restart.status_code == 200
        assert runtime["runs"] == 2
    finally:
        await client.post("/autopilot/stop")
        autopilot.set_config(original_cfg)


@pytest.mark.asyncio
async def test_autopilot_start_conflict(client, monkeypatch):
    from routers import autopilot

    if not getattr(autopilot, "_HAS_AUTO_TRADER_RUNTIME", False):  # pragma: no cover
        pytest.skip("auto trader runtime unavailable")

    async def fake_background_loop(*, config, state):
        try:
            while True:
                await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            raise

    monkeypatch.setattr(autopilot, "background_loop", fake_background_loop)
    monkeypatch.setattr(autopilot, "get_runtime_status", lambda: {"running": True})

    original_cfg = autopilot.get_config()
    try:
        start = await client.post("/autopilot/start", json={"dry_run": True})
        assert start.status_code == 200

        conflict = await client.post("/autopilot/start", json={"dry_run": True})
        assert conflict.status_code == 409

        restart = await client.post("/autopilot/start?restart=1", json={"dry_run": True})
        assert restart.status_code == 200
    finally:
        await client.post("/autopilot/stop")
        autopilot.set_config(original_cfg)
