"""Two-factor authentication helper built on TOTP."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pyotp


@dataclass(slots=True)
class TwoFactorGuard:
    vault: "SecretVault"
    issuer: str = "AI-Trader"
    drift_window: int = 1
    _secrets: Dict[str, str] = field(default_factory=dict)

    def enroll(self, user_id: str) -> str:
        secret = pyotp.random_base32()
        self._secrets[user_id] = self.vault.store(f"2fa:{user_id}", secret)
        totp = pyotp.TOTP(secret, issuer=self.issuer, name=user_id)
        return totp.provisioning_uri()

    def _resolve_secret(self, user_id: str) -> Optional[str]:
        token = self._secrets.get(user_id)
        if token is None:
            return None
        return self.vault.decrypt(token)

    def verify(self, user_id: str, otp: str, *, at_time: Optional[int] = None) -> bool:
        secret = self._resolve_secret(user_id)
        if secret is None:
            return False
        totp = pyotp.TOTP(secret)
        return bool(totp.verify(otp, valid_window=self.drift_window, for_time=at_time))

    def require(self, user_id: str, otp: str) -> None:
        if not self.verify(user_id, otp):
            raise PermissionError("Two-factor authentication failed")


# Late import to avoid circular dependency type check issues
from services.security.vault import SecretVault  # noqa: E402  pylint: disable=wrong-import-position

