"""Encrypted secret storage backed by AES-GCM and optional HSM."""
from __future__ import annotations

import base64
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Protocol

from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class HardwareSecurityModule(Protocol):
    """Abstraction over a device that protects the master key."""

    def encrypt(self, data: bytes) -> bytes:
        ...

    def decrypt(self, data: bytes) -> bytes:
        ...


@dataclass(slots=True)
class SecureHSM:
    """AES-GCM based HSM for cryptographically secure key protection.

    This replaces the previous XOR-based LocalHSM which was cryptographically insecure.
    Uses AES-256-GCM for authenticated encryption of the master key.
    """

    storage_dir: Path

    def __post_init__(self) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._key_path = self.storage_dir / "hsm.key"
        self._key = self._load_or_generate_key()

    def _load_or_generate_key(self) -> bytes:
        """Load existing key or generate new 256-bit key."""
        if self._key_path.exists():
            key = self._key_path.read_bytes()
            if len(key) != 32:
                raise ValueError(f"Invalid HSM key length: {len(key)} bytes, expected 32")
            return key

        # Generate new 256-bit key
        key = AESGCM.generate_key(bit_length=256)
        self._key_path.write_bytes(key)
        # Set read-only for owner only (0o600)
        self._key_path.chmod(0o600)
        return key

    def encrypt(self, data: bytes) -> bytes:  # type: ignore[override]
        """Encrypt data using AES-GCM with random nonce."""
        aesgcm = AESGCM(self._key)
        nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM
        ciphertext = aesgcm.encrypt(nonce, data, None)
        # Return nonce + ciphertext
        return nonce + ciphertext

    def decrypt(self, data: bytes) -> bytes:  # type: ignore[override]
        """Decrypt data using AES-GCM, verifying authentication tag."""
        if len(data) < 12:
            raise ValueError("Invalid encrypted data: too short")
        aesgcm = AESGCM(self._key)
        nonce, ciphertext = data[:12], data[12:]
        return aesgcm.decrypt(nonce, ciphertext, None)


# Backwards compatibility alias (deprecated)
LocalHSM = SecureHSM


class SecretVault:
    """Simple encrypted key/value store used for API credentials."""

    def __init__(
        self,
        *,
        master_key: Optional[bytes] = None,
        hsm: Optional[HardwareSecurityModule] = None,
        nonce_size: int = 12,
    ):
        self._hsm = hsm
        if master_key is None:
            master_key = os.environ.get("AI_TRADER_MASTER_KEY")
            if master_key is not None:
                master_key = base64.b64decode(master_key)
        if master_key is None:
            master_key = secrets.token_bytes(32)
        if self._hsm is not None:
            master_key = self._hsm.encrypt(master_key)
        self._master = master_key
        self._nonce_size = int(nonce_size)
        self._cache: Dict[str, str] = {}

    def _aes(self) -> AESGCM:
        key = self._master
        if self._hsm is not None:
            key = self._hsm.decrypt(key)
        if len(key) not in {16, 24, 32}:
            raise ValueError("Master key must be 128, 192 or 256 bits")
        return AESGCM(key)

    def encrypt(self, name: str, secret: str) -> str:
        aes = self._aes()
        nonce = secrets.token_bytes(self._nonce_size)
        ct = aes.encrypt(nonce, secret.encode("utf-8"), None)
        token = base64.b64encode(nonce + ct).decode("ascii")
        self._cache[name] = token
        return token

    def decrypt(self, token: str) -> str:
        raw = base64.b64decode(token)
        nonce, ct = raw[: self._nonce_size], raw[self._nonce_size :]
        plain = self._aes().decrypt(nonce, ct, None)
        return plain.decode("utf-8")

    def store(self, name: str, secret: str) -> str:
        return self.encrypt(name, secret)

    def fetch(self, name: str) -> Optional[str]:
        token = self._cache.get(name)
        return self.decrypt(token) if token else None

    def rotate_master_key(self, new_key: Optional[bytes] = None) -> bytes:
        new_key = new_key or secrets.token_bytes(32)
        stored = self._hsm.encrypt(new_key) if self._hsm is not None else new_key
        self._master = stored
        return new_key

