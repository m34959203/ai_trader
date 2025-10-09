"""Skeleton CLI for offline meta-label training with purged CV."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

LOG = logging.getLogger("ai_trader.meta_label_cli")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Meta-label training pipeline")
    parser.add_argument("dataset", type=Path, help="Path to the labelled events dataset (CSV/Parquet)")
    parser.add_argument("--model", default="rf", choices=["rf", "xgb"], help="Base learner type")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of Purged CV folds")
    parser.add_argument("--output", type=Path, default=Path("artifacts/meta_model.pkl"))
    return parser.parse_args()


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args()
    LOG.info(
        "Meta-label training placeholder: dataset=%s model=%s folds=%s output=%s",
        args.dataset,
        args.model,
        args.cv_folds,
        args.output,
    )
    LOG.info("This CLI is a skeleton awaiting integration with offline training stack.")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
