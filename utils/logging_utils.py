"""Utilities for secure logging and sensitive data masking."""
from __future__ import annotations

import logging
import re
from typing import Any, Iterable

SENSITIVE_KEYS = {"api_key", "api_secret", "secret", "token", "password"}
MASK = "***"

_SECRET_RE = re.compile(r"([A-Za-z0-9]{6})[A-Za-z0-9]+([A-Za-z0-9]{4})")


def _mask_value(value: Any) -> Any:
    if isinstance(value, str):
        if len(value) <= 8:
            return MASK
        return _SECRET_RE.sub(r"\1***\2", value)
    return value


def mask_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {k: (MASK if k.lower() in SENSITIVE_KEYS else mask_payload(v)) for k, v in payload.items()}
    if isinstance(payload, (list, tuple, set)):
        return type(payload)(mask_payload(v) for v in payload)
    if isinstance(payload, str):
        return _mask_value(payload)
    return payload


class SensitiveDataFilter(logging.Filter):
    """Logging filter that masks common secrets in structured payloads."""

    def __init__(self, *, fields: Iterable[str] = ()):
        super().__init__("sensitive")
        self._fields = {*(f.lower() for f in fields), *SENSITIVE_KEYS}

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - logging API
        if isinstance(record.args, dict):
            record.args = mask_payload(record.args)
        elif isinstance(record.args, tuple):
            record.args = tuple(mask_payload(arg) for arg in record.args)
        if isinstance(record.msg, dict):
            record.msg = mask_payload(record.msg)
        elif isinstance(record.msg, str):
            for field in self._fields:
                record.msg = re.sub(fr"{field}=([^\s]+)", f"{field}={MASK}", record.msg, flags=re.IGNORECASE)
        return True


def install_sensitive_filter(logger: logging.Logger, *, fields: Iterable[str] = ()) -> None:
    if any(isinstance(f, SensitiveDataFilter) for f in logger.filters):
        return
    logger.addFilter(SensitiveDataFilter(fields=fields))

