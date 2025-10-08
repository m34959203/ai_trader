from __future__ import annotations

"""Audit trail helpers."""

import logging
from typing import Any, Dict, Optional

try:
    from sqlalchemy.ext.asyncio import AsyncSession
    from db.models_intel import AuditLog
except Exception:  # pragma: no cover
    AsyncSession = None  # type: ignore
    AuditLog = None  # type: ignore

LOG = logging.getLogger("ai_trader.audit")


async def record_audit_event(
    session: AsyncSession,
    *,
    action: str,
    actor: str = "system",
    scope: Optional[str] = None,
    status: str = "ok",
    details: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    if AuditLog is None:
        LOG.debug("audit disabled (no model)")
        return
    entry = AuditLog(
        actor=actor,
        action=action,
        scope=scope,
        status=status,
        details=details,
        payload=payload,
    )
    session.add(entry)
