"""Auto-healing orchestration for executors and services."""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

LOG_STATE_DIR = Path("state/auto_heal")


@dataclass(slots=True)
class StateSnapshot:
    name: str
    payload: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "payload": self.payload}, sort_keys=True)

    @staticmethod
    def from_json(data: str) -> "StateSnapshot":
        obj = json.loads(data)
        return StateSnapshot(name=obj["name"], payload=obj.get("payload", {}))


@dataclass(slots=True)
class AutoHealingOrchestrator:
    state_dir: Path = LOG_STATE_DIR
    restore_callbacks: Dict[str, Callable[[Dict[str, Any]], Awaitable[None]]] = field(default_factory=dict)
    _lock: asyncio.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def write_snapshot(self, snapshot: StateSnapshot) -> Path:
        async with self._lock:
            path = self.state_dir / f"{snapshot.name}.json"
            path.write_text(snapshot.to_json(), encoding="utf-8")
            return path

    async def load_snapshot(self, name: str) -> Optional[StateSnapshot]:
        path = self.state_dir / f"{name}.json"
        if not path.exists():
            return None
        data = path.read_text(encoding="utf-8")
        return StateSnapshot.from_json(data)

    def register_restore(
        self,
        name: str,
        callback: Callable[[Dict[str, Any]], Awaitable[None]],
    ) -> None:
        self.restore_callbacks[name] = callback

    async def trigger(self, name: str, payload: Optional[Dict[str, Any]] = None) -> bool:
        cb = self.restore_callbacks.get(name)
        if cb is None:
            return False
        await cb(payload or {})
        return True

    async def restore(self, name: str) -> bool:
        snap = await self.load_snapshot(name)
        if snap is None:
            return False
        return await self.trigger(name, snap.payload)

    async def replay(self, name: Optional[str] = None) -> int:
        count = 0
        if not self.state_dir.exists():
            return 0
        for path in sorted(self.state_dir.glob("*.json")):
            try:
                snap = StateSnapshot.from_json(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if name and snap.name != name:
                continue
            if await self.trigger(snap.name, snap.payload):
                count += 1
        return count

    async def topology_restart(self, components: Dict[str, Callable[[], Awaitable[None]]]) -> None:
        for name, action in components.items():
            try:
                await action()
            except Exception as exc:  # pragma: no cover - best effort logging
                import logging

                logging.getLogger("ai_trader.auto_heal").warning(
                    "Failed to restart component %s: %r", name, exc
                )

