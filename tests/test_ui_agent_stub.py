import pytest

from executors.ui_agent import UIExecutorAgent


@pytest.mark.asyncio
async def test_ui_agent_dom_and_fallback():
    agent = UIExecutorAgent(testnet=True, config={"latency_ms": 1})
    res_dom = await agent.open_market(symbol="BTCUSDT", side="buy", amount=0.01)
    assert res_dom.raw["source"] == "dom"

    agent._backend.dom_available = False  # type: ignore[attr-defined]
    res_ocr = await agent.open_market(symbol="BTCUSDT", side="sell", amount=0.01)
    assert res_ocr.raw["source"] == "ocr"
    assert res_ocr.raw.get("failover_reason")

    closed = await agent.close_all(symbol="BTCUSDT")
    assert closed and closed[0].symbol == "BTCUSDT"
    await agent.close()
