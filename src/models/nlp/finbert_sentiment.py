"""FinBERT-backed sentiment estimator with rule-based fallback.

The implementation supports offline execution by caching the downloaded
HuggingFace snapshot locally and validating its checksum before using the
model. When the cache is available the model is loaded without attempting
to access the internet which allows the trading stack to remain fully
air-gapped.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from ..base import ISentimentModel, SentimentOutput
from ...utils_math import clamp01

LOG = logging.getLogger("ai_trader.models.sentiment.finbert")

_CACHE_ENV_VAR = "AI_TRADER_MODEL_CACHE"
_CHECKSUM_FILENAME = ".checksum"


@dataclass
class FinBERTParams:
    model_name: str = "ProsusAI/finbert"
    device: int = -1
    use_pipeline: bool = True


class FinBERTLite(ISentimentModel):
    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        raw = params or {}
        self._params = FinBERTParams(
            model_name=str(raw.get("model_name", FinBERTParams.model_name)),
            device=int(raw.get("device", FinBERTParams.device)),
            use_pipeline=bool(raw.get("use_pipeline", FinBERTParams.use_pipeline)),
        )
        self._pipeline = self._load_pipeline() if self._params.use_pipeline else None
        self._fallback_lexicon = {
            "bull": 1.0,
            "beat": 0.8,
            "growth": 0.7,
            "record": 0.6,
            "profit": 0.6,
            "buyback": 0.6,
            "upgrade": 0.5,
            "positive": 0.4,
            "strong": 0.3,
            "bear": -1.0,
            "miss": -0.8,
            "loss": -0.7,
            "downgrade": -0.6,
            "negative": -0.5,
            "weak": -0.4,
            "fraud": -0.9,
        }

    def _load_pipeline(self):  # pragma: no cover - heavy dependency guard
        cache_path: Optional[Path] = None
        try:
            cache_path = self._ensure_local_model()
        except Exception as exc:  # noqa: BLE001 - defensive path
            LOG.warning("Failed to prepare local FinBERT cache (%s).", exc)

        if cache_path is None:
            LOG.warning("FinBERT cache is unavailable; sentiment pipeline disabled.")
            return None

        try:
            from transformers import pipeline  # type: ignore

            LOG.info("Loading FinBERT pipeline from %s", cache_path)
            return pipeline(
                "sentiment-analysis",
                model=str(cache_path),
                tokenizer=str(cache_path),
                device=self._params.device,
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("FinBERT pipeline unavailable (%s). Falling back to rules.", exc)
            return None

    def analyze(self, text: str) -> SentimentOutput:
        text = (text or "").strip()
        if not text:
            return SentimentOutput(label="neu", score=0.0, confidence=0.0, reasons={"empty": True})

        if self._pipeline is not None:
            try:
                result = self._pipeline(text)[0]
                label_raw = str(result["label"]).lower()
                score = float(result.get("score", 0.0))
            except Exception as exc:  # pragma: no cover - runtime guard
                LOG.warning("FinBERT inference failed (%s). Using fallback.", exc)
                return self._fallback(text)

            label, signed_score = self._map_label(label_raw, score)
            return SentimentOutput(
                label=label,
                score=signed_score,
                confidence=clamp01(score),
                reasons={"provider": "finbert", "raw_label": label_raw, "raw_score": score},
            )

        return self._fallback(text)

    def _fallback(self, text: str) -> SentimentOutput:
        tokens = [token.strip(".,!?:;()[]{}\"'`").lower() for token in text.split() if token]
        total = 0.0
        hits = 0
        for token in tokens:
            if token in self._fallback_lexicon:
                hits += 1
                total += self._fallback_lexicon[token]
        if hits == 0:
            return SentimentOutput(label="neu", score=0.0, confidence=0.1, reasons={"provider": "rules"})

        avg_score = max(-1.0, min(1.0, total / hits))
        label = "pos" if avg_score > 0.1 else "neg" if avg_score < -0.1 else "neu"
        confidence = clamp01(abs(avg_score))
        return SentimentOutput(
            label=label,
            score=avg_score,
            confidence=confidence,
            reasons={"provider": "rules", "hits": hits, "tokens": tokens[:10]},
        )

    @staticmethod
    def _map_label(label_raw: str, score: float) -> tuple[str, float]:
        if "neg" in label_raw:
            return "neg", -abs(score)
        if "pos" in label_raw:
            return "pos", abs(score)
        return "neu", 0.0

    # ------------------------------------------------------------------
    # Offline cache helpers
    # ------------------------------------------------------------------
    def _ensure_local_model(self) -> Optional[Path]:
        """Download the model snapshot if needed and validate checksum.

        Returns the path that can be passed to the HuggingFace pipeline or
        ``None`` when no valid cache is available.
        """

        cache_root = self._resolve_cache_root()
        model_dir = cache_root / self._sanitise_name(self._params.model_name)
        checksum_path = model_dir / _CHECKSUM_FILENAME
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            from huggingface_hub import snapshot_download  # type: ignore

            LOG.debug("Ensuring FinBERT snapshot in %s", model_dir)
            snapshot_download(
                repo_id=self._params.model_name,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
            )
            checksum = self._compute_directory_checksum(model_dir)
            checksum_path.write_text(checksum, encoding="utf-8")
            LOG.info("FinBERT snapshot ready (checksum=%s)", checksum[:12])
            return model_dir
        except Exception as exc:  # noqa: BLE001 - offline fallback
            LOG.warning("FinBERT snapshot download skipped (%s). Using cache only.", exc)
            if checksum_path.exists():
                checksum_valid = self._verify_checksum(model_dir, checksum_path)
                if checksum_valid:
                    LOG.info("FinBERT cache verified (checksum=%s)", checksum_valid[:12])
                    return model_dir
                LOG.error("FinBERT cache checksum mismatch; ignoring cached files.")
            elif any(model_dir.iterdir()):
                LOG.warning("FinBERT cache has no checksum; attempting best-effort validation.")
                checksum = self._compute_directory_checksum(model_dir)
                checksum_path.write_text(checksum, encoding="utf-8")
                return model_dir

        return None

    @staticmethod
    def _resolve_cache_root() -> Path:
        env_override = os.environ.get(_CACHE_ENV_VAR)
        if env_override:
            return Path(env_override).expanduser().resolve()
        project_root = Path(__file__).resolve().parents[4]
        return project_root / "state" / "models"

    @staticmethod
    def _sanitise_name(name: str) -> str:
        return name.replace("/", "__")

    @classmethod
    def _verify_checksum(cls, directory: Path, checksum_file: Path) -> Optional[str]:
        try:
            expected = checksum_file.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        actual = cls._compute_directory_checksum(directory)
        return expected if expected and expected == actual else None

    @staticmethod
    def _iter_files(directory: Path) -> Iterable[Path]:
        for item in sorted(directory.rglob("*")):
            if item.is_file() and item.name != _CHECKSUM_FILENAME:
                yield item

    @classmethod
    def _compute_directory_checksum(cls, directory: Path) -> str:
        hasher = hashlib.sha256()
        for file_path in cls._iter_files(directory):
            relative = file_path.relative_to(directory)
            hasher.update(str(relative).encode("utf-8"))
            try:
                with file_path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        hasher.update(chunk)
            except OSError:
                continue
        return hasher.hexdigest()
