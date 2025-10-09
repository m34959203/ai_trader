"""Placeholder DRL consultant agent built around stable-baselines3."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

LOG = logging.getLogger("ai_trader.models.drl.consultant")


class ConsultantAgent:
    def __init__(self, algo: str = "ppo", params: Optional[Dict[str, Any]] = None) -> None:
        self._algo = algo.lower()
        self._params = params or {}
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from stable_baselines3 import PPO, DQN  # type: ignore

            LOG.info("Initialising stable-baselines3 placeholder agent (%s)", self._algo)
            if self._algo == "dqn":
                self._model = DQN("MlpPolicy", env=None, verbose=0)  # type: ignore[arg-type]
            else:
                self._model = PPO("MlpPolicy", env=None, verbose=0)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover
            LOG.warning("stable-baselines3 not available (%s); agent will operate in noop mode", exc)
            self._model = None

    def advise(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_model()
        if self._model is None:
            return {"action": "hold", "confidence": 0.0, "provider": "placeholder"}
        LOG.info("DRL consultant placeholder invoked; integrate environment for production use")
        return {"action": "hold", "confidence": 0.0, "provider": self._algo}
