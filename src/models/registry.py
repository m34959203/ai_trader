"""Simple dependency injection registry for model factories."""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

RegistryFactory = Callable[[Optional[Dict[str, Any]]], Any]


class ModelRegistry:
    def __init__(self) -> None:
        self._factories: Dict[str, RegistryFactory] = {}

    def register(self, name: str, factory: RegistryFactory) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("Model name must be non-empty")
        self._factories[key] = factory

    def create(self, name: str, params: Optional[Dict[str, Any]] = None) -> Any:
        key = name.strip().lower()
        try:
            factory = self._factories[key]
        except KeyError as exc:  # pragma: no cover - guard rail
            raise KeyError(f"Model '{name}' is not registered") from exc
        return factory(params)

    def has(self, name: str) -> bool:
        return name.strip().lower() in self._factories


_registry = ModelRegistry()


def register_model(name: str, factory: RegistryFactory) -> None:
    _registry.register(name, factory)


def create_model(name: str, params: Optional[Dict[str, Any]] = None) -> Any:
    return _registry.create(name, params)


def registry_has(name: str) -> bool:
    return _registry.has(name)
