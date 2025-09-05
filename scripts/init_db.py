# scripts/init_db.py
from __future__ import annotations
import asyncio
import os
import sys
from pathlib import Path

# --- убедимся, что корень проекта в sys.path ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Если .env лежит в configs/.env — это уже обрабатывает db/session.py,
# но на всякий случай можно подсказать PYTHONPATH через окружение.
os.environ.setdefault("PYTHONPATH", f"{ROOT};" + os.environ.get("PYTHONPATH", ""))

async def main() -> None:
    # ВАЖНО: импортируем модели до create_all, чтобы таблицы были зарегистрированы в Base.metadata
    # Пример: from db import models_orders, models_ohlcv, models_positions
    # ЗАМЕНЙ НА ТВОИ РЕАЛЬНЫЕ МОДУЛИ МОДЕЛЕЙ:
    try:
        # если все модели собраны в одном модуле:
        import db.models  # noqa: F401
    except Exception:
        # или импортируй по отдельности:
        # import db.models_orders  # noqa: F401
        # import db.models_ohlcv  # noqa: F401
        pass

    from db.session import apply_startup_pragmas_and_schema

    await apply_startup_pragmas_and_schema()
    print("✅ DB schema created / ensured successfully.")

if __name__ == "__main__":
    asyncio.run(main())
