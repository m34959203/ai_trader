import pyotp
import pytest

from services.trading_service import TradingService


class FailingExecutor:
    name = "failing"

    async def open_market(self, **kwargs):  # noqa: D401 - test stub
        raise RuntimeError("network down")

    async def fetch_balance(self):
        return {"total": 1000.0}

    async def get_positions(self, *, symbols=None):  # pragma: no cover - deterministic empty
        return []

    async def close(self):  # pragma: no cover - simple stub
        pass


@pytest.mark.asyncio
async def test_trading_service_failover(monkeypatch):
    service = TradingService(mode="sim")
    failing = FailingExecutor()
    service._executor = failing  # type: ignore[attr-defined]
    res = await service.open_market(symbol="BTCUSDT", side="buy", amount=0.01)
    assert res.raw["source"] in {"dom", "ocr"}
    assert service.failover_count == 1
    await service.close()


@pytest.mark.asyncio
async def test_trading_service_security_enforcement():
    service = TradingService(
        mode="sim",
        config={
            "security": {
                "require_2fa": True,
                "roles": {"trader": ["trade:open"]},
                "assignments": {"alice": "trader"},
            }
        },
    )
    uri = service.enroll_twofactor("alice")
    assert uri.startswith("otpauth")
    secret = service._vault.decrypt(service._twofactor._secrets["alice"])  # type: ignore[attr-defined]
    otp = pyotp.TOTP(secret).now()
    with pytest.raises(PermissionError):
        await service.open_market(symbol="ETHUSDT", side="buy", amount=0.01)
    res = await service.open_market(
        symbol="ETHUSDT",
        side="buy",
        amount=0.01,
        security_ctx={"user": "alice", "otp": otp},
    )
    symbol = res.symbol if hasattr(res, "symbol") else res.get("symbol")
    assert symbol == "ETHUSDT"
    with pytest.raises(PermissionError):
        await service.open_market(
            symbol="ETHUSDT",
            side="buy",
            amount=0.01,
            security_ctx={"user": "bob", "otp": otp},
        )
    await service.close()


