import pyotp

from services.security import AccessController, Role, SecretVault, TwoFactorGuard


def test_secret_vault_roundtrip():
    vault = SecretVault()
    token = vault.store("api", "super-secret")
    assert token != "super-secret"
    assert vault.fetch("api") == "super-secret"


def test_twofactor_guard():
    vault = SecretVault()
    guard = TwoFactorGuard(vault)
    uri = guard.enroll("alice")
    assert "otpauth" in uri
    secret = vault.decrypt(guard._secrets["alice"])  # type: ignore[attr-defined]
    otp = pyotp.TOTP(secret).now()
    assert guard.verify("alice", otp)


def test_rbac_permissions():
    roles = {
        "trader": Role.from_permissions("trader", ["trade:open", "trade:close"]),
        "viewer": Role.from_permissions("viewer", ["trade:view"]),
    }
    ac = AccessController(roles=roles)
    ac.assign("bob", "trader")
    assert ac.can("bob", "trade:open")
    assert not ac.can("bob", "trade:view")
    ac.assign("eve", "viewer")
    ac.require("eve", "trade:view")


