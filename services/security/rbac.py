"""Simple role based access control (RBAC) implementation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Set


@dataclass(slots=True, frozen=True)
class Role:
    name: str
    permissions: frozenset[str]

    @staticmethod
    def from_permissions(name: str, permissions: Iterable[str]) -> "Role":
        return Role(name=name, permissions=frozenset(str(p) for p in permissions))


@dataclass(slots=True)
class AccessController:
    roles: Dict[str, Role]
    assignments: Dict[str, str] = field(default_factory=dict)

    def assign(self, user_id: str, role_name: str) -> None:
        if role_name not in self.roles:
            raise KeyError(f"Unknown role: {role_name}")
        self.assignments[user_id] = role_name

    def permissions_for(self, user_id: str) -> Set[str]:
        role_name = self.assignments.get(user_id)
        if role_name is None:
            return set()
        role = self.roles.get(role_name)
        return set(role.permissions) if role else set()

    def can(self, user_id: str, permission: str) -> bool:
        return str(permission) in self.permissions_for(user_id)

    def require(self, user_id: str, permission: str) -> None:
        if not self.can(user_id, permission):
            raise PermissionError(f"User {user_id} lacks permission {permission}")


