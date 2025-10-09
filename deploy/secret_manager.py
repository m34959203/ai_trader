"""Deployment helpers for bootstrapping live-trading secrets with 2FA gating."""
from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

from services.security import SecretVault, TwoFactorGuard

LOG = logging.getLogger("ai_trader.deploy.secret_manager")


@dataclass(frozen=True)
class SecretSpec:
    """Description of a secret required for live trading."""

    env: str
    required: bool = True
    description: str = ""


LIVE_SECRET_SPECS: Mapping[str, SecretSpec] = {
    "binance_api_key": SecretSpec(
        env="BINANCE_API_KEY",
        description="Binance Spot production API key",
    ),
    "binance_api_secret": SecretSpec(
        env="BINANCE_API_SECRET",
        description="Binance Spot production API secret",
    ),
    "binance_testnet_api_key": SecretSpec(
        env="BINANCE_TESTNET_API_KEY",
        description="Binance Testnet API key",
    ),
    "binance_testnet_api_secret": SecretSpec(
        env="BINANCE_TESTNET_API_SECRET",
        description="Binance Testnet API secret",
    ),
    "hf_token": SecretSpec(
        env="HF_TOKEN",
        required=False,
        description="Optional HuggingFace token for private models",
    ),
    "prometheus_basic_auth": SecretSpec(
        env="PROMETHEUS_BASIC_AUTH",
        required=False,
        description="Optional Prometheus basic auth secret",
    ),
    "slack_webhook": SecretSpec(
        env="SLACK_WEBHOOK_URL",
        required=False,
        description="Optional Slack alerting webhook",
    ),
}


def _load_payload_from_env() -> Dict[str, str]:
    payload_raw = os.getenv("SECRET_MANAGER_PAYLOAD")
    if not payload_raw:
        return {}
    try:
        parsed = json.loads(payload_raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive guard
        raise RuntimeError("SECRET_MANAGER_PAYLOAD must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("SECRET_MANAGER_PAYLOAD must decode into an object")
    return {str(k): str(v) for k, v in parsed.items()}


def _resolve_secret(spec: SecretSpec, payload: Mapping[str, str]) -> Optional[str]:
    if spec.env in payload:
        return payload[spec.env]
    env_value = os.getenv(spec.env)
    if env_value:
        return env_value
    file_path = os.getenv(f"{spec.env}_FILE")
    if file_path:
        path = Path(file_path)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    return None


def materialise_live_trading_secrets(
    vault: SecretVault,
    *,
    specs: Mapping[str, SecretSpec] = LIVE_SECRET_SPECS,
) -> Dict[str, str]:
    """Load secrets from the configured manager and cache them in :class:`SecretVault`."""

    payload = _load_payload_from_env()
    resolved: Dict[str, str] = {}
    missing: Dict[str, SecretSpec] = {}
    for name, spec in specs.items():
        value = _resolve_secret(spec, payload)
        if value is None:
            if spec.required:
                missing[spec.env] = spec
            continue
        vault.store(spec.env, value)
        resolved[spec.env] = value

    if missing:
        details = ", ".join(f"{env}: {spec.description}" for env, spec in missing.items())
        raise RuntimeError(f"Missing required secrets: {details}")

    LOG.info("Loaded %d secrets into SecretVault", len(resolved))
    return resolved


def ensure_deployment_2fa(guard: TwoFactorGuard, *, user_id: str = "deploy") -> None:
    """Enforce 2FA verification prior to secret materialisation."""

    seed = os.getenv("DEPLOY_2FA_SECRET")
    if not seed:
        raise RuntimeError("DEPLOY_2FA_SECRET must be provided for 2FA enforcement")

    token = guard.vault.store(f"2fa:{user_id}", seed)
    guard._secrets[user_id] = token  # pylint: disable=protected-access

    otp = os.getenv("DEPLOY_2FA_CODE")
    if not otp:
        raise RuntimeError("DEPLOY_2FA_CODE must contain a valid OTP for deployment")

    if not guard.verify(user_id, otp):
        raise PermissionError("Invalid deployment OTP supplied")

    LOG.info("2FA verification succeeded for deployment user %s", user_id)


def _write_env_file(path: Path, secrets: Mapping[str, str], *, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        path.write_text(json.dumps(secrets, indent=2, sort_keys=True), encoding="utf-8")
        return
    lines = [f"{key}={value}" for key, value in secrets.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bootstrap live trading secrets")
    parser.add_argument("--write-env", type=Path, default=None, help="Path to write resolved secrets")
    parser.add_argument(
        "--format",
        choices=("env", "json"),
        default="env",
        help="Output format when writing secrets to file or stdout",
    )
    parser.add_argument("--skip-2fa", action="store_true", help="Skip OTP verification (testing only)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    vault = SecretVault()
    guard = TwoFactorGuard(vault)

    if not args.skip_2fa:
        ensure_deployment_2fa(guard)
    else:
        LOG.warning("2FA enforcement skipped — use only in non-production environments")

    secrets = materialise_live_trading_secrets(vault)

    if args.write_env:
        _write_env_file(args.write_env, secrets, fmt=args.format)
    else:
        if args.format == "json":
            print(json.dumps(secrets, indent=2, sort_keys=True))
        else:
            for key, value in secrets.items():
                print(f"export {key}='{value}'")

    return 0


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
