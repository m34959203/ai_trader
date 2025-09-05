# scripts/reset_db.py
from __future__ import annotations

import os
import sys
from pathlib import Path
from sqlalchemy import text

# --- Добавляем корень проекта в sys.path ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from db.session import engine, Base  # type: ignore

DB_FILE = ROOT / "data" / "ai_trader.db"
DB_FILE.parent.mkdir(parents=True, exist_ok=True)

def reset():
    # Для SQLite проще и чище — удалить файл БД:
    if DB_FILE.exists():
        DB_FILE.unlink()

    # Создаём новую схему
    Base.metadata.create_all(bind=engine)

    # Проверочный запрос
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))

if __name__ == "__main__":
    reset()
    print(f"✅ DB reset OK → {DB_FILE}")
