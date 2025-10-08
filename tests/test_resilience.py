from pathlib import Path

import pytest

from services.auto_heal import AutoHealingOrchestrator, StateSnapshot
from monitoring.observability import ObservabilityHub
from monitoring.slo import SLO, SLOTracker


@pytest.mark.asyncio
async def test_auto_heal_snapshot(tmp_path: Path):
    orchestrator = AutoHealingOrchestrator(state_dir=tmp_path)
    called = {}

    async def restore_cb(payload):
        called.update(payload)

    orchestrator.register_restore("exec", restore_cb)
    await orchestrator.write_snapshot(StateSnapshot(name="exec", payload={"mode": "binance"}))
    ok = await orchestrator.restore("exec")
    assert ok
    assert called["mode"] == "binance"

    called.clear()
    count = await orchestrator.replay()
    assert count >= 1
    assert called["mode"] == "binance"


def test_slo_tracker():
    tracker = SLOTracker([SLO(name="latency", target=0.95, window=5)])
    tracker.record("latency", True)
    tracker.record("latency", False)
    tracker.record("latency", True)
    report = tracker.report()["latency"]
    assert report["target"] == 0.95
    assert isinstance(report["actual"], float)


def test_observability_hub():
    hub = ObservabilityHub()
    hub.record_order(symbol="BTCUSDT", success=True, latency=0.2)
    hub.record_order(symbol="BTCUSDT", success=False, latency=3.0)
    hub.record_failover(context="open_market:BTCUSDT", recovered=False)
    report = hub.slo_report()
    assert "order_success" in report
    metrics = hub.export_prometheus()
    assert b"ai_trader_orders_total" in metrics


