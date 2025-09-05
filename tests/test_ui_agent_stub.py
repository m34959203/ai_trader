import asyncio
from executors.ui_agent import UIExecutorStub


async def test_ui_stub():
    ex = UIExecutorStub(testnet=True)
    res = await ex.open_market(symbol="BTCUSDT", side="buy", amount=0.01)
    assert res.id == "ui-stub"
    await ex.close()
