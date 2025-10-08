"""Asset universe helpers."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

ASSET_FILE = Path("configs/assets_extended.json")


@lru_cache(maxsize=1)
def load_assets() -> Dict[str, List[str]]:
    if not ASSET_FILE.exists():
        return {"spot": [], "futures": []}
    data = json.loads(ASSET_FILE.read_text(encoding="utf-8"))
    return {k: sorted(set(v)) for k, v in data.items()}


