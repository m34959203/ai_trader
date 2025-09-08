# ai_trader/db/session.py
from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import AsyncIterator, Optional, Callable, Awaitable, Tuple

from dotenv import load_dotenv, dotenv_values
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

# ──────────────────────────────────────────────────────────────────────────────
# Пути/окружение
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../ai_trader


def _load_env_robust() -> Optional[Path]:
    """
    Ищем .env сначала в configs/.env, затем в корне проекта.
    Чиним BOM и лишние пробелы в ключах.
    """
    for p in (PROJECT_ROOT / "configs" / ".env", PROJECT_ROOT / ".env"):
        if p.exists():
            load_dotenv(p.as_posix(), override=True, encoding="utf-8")
            for k, v in (dotenv_values(p, encoding="utf-8") or {}).items():
                if k is None or v is None:
                    continue
                os.environ[k.strip().lstrip("\ufeff")] = v
            return p
    return None


ENV_PATH = _load_env_robust()

# ──────────────────────────────────────────────────────────────────────────────
# ENV helpers
# ──────────────────────────────────────────────────────────────────────────────
def env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "on", "y"}


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None else str(v)


# ──────────────────────────────────────────────────────────────────────────────
# Конфигурация БД
#  - SQLite (aiosqlite) по умолчанию в data/ai_trader.db (АБСОЛЮТНЫЙ путь)
#  - PostgreSQL/TimescaleDB (asyncpg)
# ──────────────────────────────────────────────────────────────────────────────
def _default_db_url() -> str:
    """Абсолютный путь к SQLite внутри проекта; гарантируем существование каталога."""
    db_file = (PROJECT_ROOT / "data" / "ai_trader.db")
    db_file.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{db_file.as_posix()}"


def _normalize_db_url(raw: Optional[str]) -> str:
    """
    Нормализация популярных вариантов:
      - sqlite:// → sqlite+aiosqlite://
      - для SQLite — абсолютный путь; поддержка :memory:
      - грубая валидация «dialect+driver»; при мусоре → дефолтная SQLite
    """
    if not raw:
        return _default_db_url()

    url = raw.strip()

    # Приведение SQLite к async-драйверу
    if url.startswith("sqlite:///") and not url.startswith("sqlite+aiosqlite:///"):
        url = url.replace("sqlite:///", "sqlite+aiosqlite:///")
    elif url.startswith("sqlite://") and not url.startswith("sqlite+aiosqlite://"):
        url = url.replace("sqlite://", "sqlite+aiosqlite://")

    # Грубая валидация головы URL
    head = url.split("://", 1)[0]  # например sqlite+aiosqlite
    if head.replace("+", ".").count(".") > 1:
        return _default_db_url()

    # SQLite: абсолютный путь (кроме :memory:)
    if head.startswith("sqlite+aiosqlite"):
        if url.startswith("sqlite+aiosqlite://:memory:"):
            return url.replace("sqlite+aiosqlite://:memory:", "sqlite+aiosqlite:///:memory:")

        if url.startswith("sqlite+aiosqlite://") and not url.startswith("sqlite+aiosqlite:///"):
            url = url.replace("sqlite+aiosqlite://", "sqlite+aiosqlite:///")

        if ":memory:" in url:
            return url

        prefix = "sqlite+aiosqlite:///"
        if url.startswith(prefix):
            path_str = url[len(prefix):]
            p = Path(path_str)
            if not p.is_absolute():
                p = (PROJECT_ROOT / p).resolve()
            p.parent.mkdir(parents=True, exist_ok=True)
            return f"{prefix}{p.as_posix()}"

    return url


DB_URL = _normalize_db_url(os.getenv("DB_URL"))
DB_ECHO = env_bool("DB_ECHO", False)

# Диалектные флаги (после нормализации)
IS_SQLITE = DB_URL.startswith("sqlite+aiosqlite://")
IS_POSTGRES = DB_URL.startswith("postgresql+asyncpg://") or DB_URL.startswith("postgres+asyncpg://")

# PostgreSQL тюнинг
PG_STATEMENT_TIMEOUT_MS = env_int("PG_STATEMENT_TIMEOUT_MS", 0)  # 0 = off
PG_LOCK_TIMEOUT_MS = env_int("PG_LOCK_TIMEOUT_MS", 0)
PG_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_MS = env_int("PG_IDLE_IN_XACT_TIMEOUT_MS", 0)
PG_TIMEZONE = env_str("PG_TIMEZONE", "UTC")
PG_ENABLE_TIMESCALE = env_bool("PG_ENABLE_TIMESCALE", False)

# SQLite PRAGMA/кэш
SQLITE_WAL = env_bool("SQLITE_WAL", True)
SQLITE_SYNC_NORMAL = env_bool("SQLITE_SYNC_NORMAL", True)
SQLITE_FOREIGN_KEYS = env_bool("SQLITE_FOREIGN_KEYS", True)
SQLITE_TEMP_STORE = os.getenv("SQLITE_TEMP_STORE", "memory").strip().lower()  # memory|file
SQLITE_CACHE_SIZE = os.getenv("SQLITE_CACHE_SIZE", "-20000")  # отрицательное => КБ

# Периодические снапшоты SQLite (для файловой БД)
SQLITE_SNAPSHOT_ENABLED = env_bool("SQLITE_SNAPSHOT_ENABLED", False)
SQLITE_SNAPSHOT_DIR = Path(env_str("SQLITE_SNAPSHOT_DIR", (PROJECT_ROOT / "data" / "sqlite_snaps").as_posix()))
SQLITE_SNAPSHOT_INTERVAL_SEC = env_int("SQLITE_SNAPSHOT_INTERVAL_SEC", 3600)
SQLITE_SNAPSHOT_RETENTION = env_int("SQLITE_SNAPSHOT_RETENTION", 14)

# ──────────────────────────────────────────────────────────────────────────────
# Declarative Base
# ──────────────────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    """Базовый класс моделей. Наследуйтесь от него в db.models*."""
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Создание движка
# ──────────────────────────────────────────────────────────────────────────────
def _build_engine() -> AsyncEngine:
    """
    Создаёт AsyncEngine с безопасным фолбэком на локальную SQLite при ошибке URL.
    Вешает диалект-специфичные слушатели connect() (PRAGMA/SET ...).
    """
    global DB_URL, IS_SQLITE, IS_POSTGRES  # переопределяем при фолбэке

    kwargs = dict(
        future=True,
        echo=DB_ECHO,
        pool_pre_ping=True,
    )

    try:
        engine_ = create_async_engine(DB_URL, **kwargs)
    except Exception:
        # страховка от ошибок парсинга/драйвера
        safe_url = _default_db_url()
        DB_URL = safe_url
        os.environ["DB_URL"] = safe_url
        IS_SQLITE = True
        IS_POSTGRES = False
        engine_ = create_async_engine(safe_url, **kwargs)

    backend = engine_.url.get_backend_name()

    if backend.startswith("sqlite"):
        @event.listens_for(engine_.sync_engine, "connect")
        def _sqlite_pragmas(dbapi_connection, connection_record):  # type: ignore
            try:
                c = dbapi_connection.cursor()
                c.execute("PRAGMA foreign_keys=ON;")
                if SQLITE_SYNC_NORMAL:
                    c.execute("PRAGMA synchronous=NORMAL;")
                if SQLITE_TEMP_STORE in ("memory", "file"):
                    c.execute(f"PRAGMA temp_store={(0 if SQLITE_TEMP_STORE == 'file' else 2)};")
                try:
                    int(SQLITE_CACHE_SIZE)
                    c.execute(f"PRAGMA cache_size={SQLITE_CACHE_SIZE};")
                except Exception:
                    pass
                c.close()
            except Exception:
                pass

    if backend.startswith("postgresql") or backend.startswith("postgres"):
        @event.listens_for(engine_.sync_engine, "connect")
        def _pg_session_setup(dbapi_connection, connection_record):  # type: ignore
            try:
                cur = dbapi_connection.cursor()
                if PG_TIMEZONE:
                    cur.execute(f"SET TIME ZONE '{PG_TIMEZONE}';")
                if PG_STATEMENT_TIMEOUT_MS > 0:
                    cur.execute(f"SET statement_timeout = {PG_STATEMENT_TIMEOUT_MS};")
                if PG_LOCK_TIMEOUT_MS > 0:
                    cur.execute(f"SET lock_timeout = {PG_LOCK_TIMEOUT_MS};")
                if PG_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_MS > 0:
                    cur.execute(f"SET idle_in_transaction_session_timeout = {PG_IDLE_IN_TRANSACTION_SESSION_TIMEOUT_MS};")
                cur.close()
            except Exception:
                pass

    return engine_


engine: AsyncEngine = _build_engine()

# Основная фабрика сессий
AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    expire_on_commit=False,
    class_=AsyncSession,
)

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI DI
# ──────────────────────────────────────────────────────────────────────────────
async def get_session() -> AsyncIterator[AsyncSession]:
    """
    FastAPI-совместимая фабрика сессии: dependency injection через Depends(get_session).
    """
    async with AsyncSessionLocal() as session:
        yield session


# ──────────────────────────────────────────────────────────────────────────────
# Стартап-инициализация: PRAGMA/расширения + лёгкая миграция ohlcv + create_all
# ──────────────────────────────────────────────────────────────────────────────
def _is_memory_sqlite(url: str) -> bool:
    return url.endswith(":memory:") or url.endswith(":memory:?cache=shared")


async def apply_startup_pragmas_and_schema() -> None:
    """
    Вызывайте один раз на старте приложения (например, в lifespan FastAPI).
      • SQLite: PRAGMA journal_mode=WAL (если не :memory:), synchronous, FK, temp_store, cache_size.
      • PostgreSQL: SET timezone/statement_timeout/lock_timeout; опц. TimescaleDB.
      • Лёгкая миграция SQLite-таблицы ohlcv: добавление source/asset/tf + уникальный индекс (source,asset,tf,ts).
      • Создание таблиц: Base.metadata.create_all (если не используете Alembic).
    """
    # Гарантируем каталог для файловой SQLite
    if IS_SQLITE and DB_URL.startswith("sqlite+aiosqlite:///"):
        path_str = DB_URL.replace("sqlite+aiosqlite:///", "")
        if path_str and not path_str.startswith(":memory:"):
            Path(path_str).parent.mkdir(parents=True, exist_ok=True)

    async with engine.begin() as conn:
        if IS_SQLITE:
            # PRAGMA
            if not _is_memory_sqlite(DB_URL) and SQLITE_WAL:
                await conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
            if SQLITE_SYNC_NORMAL:
                await conn.exec_driver_sql("PRAGMA synchronous=NORMAL;")
            if SQLITE_FOREIGN_KEYS:
                await conn.exec_driver_sql("PRAGMA foreign_keys=ON;")
            if SQLITE_TEMP_STORE in ("memory", "file"):
                await conn.exec_driver_sql(f"PRAGMA temp_store={(0 if SQLITE_TEMP_STORE == 'file' else 2)};")
            try:
                int(SQLITE_CACHE_SIZE)
                await conn.exec_driver_sql(f"PRAGMA cache_size={SQLITE_CACHE_SIZE};")
            except Exception:
                pass

            # Лёгкая миграция ohlcv (если таблица уже есть)
            try:
                res = await conn.exec_driver_sql(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='ohlcv';"
                )
                row = await res.first()
                if row:
                    cols_res = await conn.exec_driver_sql("PRAGMA table_info('ohlcv');")
                    cols = [r[1] for r in await cols_res.fetchall()]  # r[1] = name
                    alter_stmts = []
                    if "source" not in cols:
                        alter_stmts.append("ALTER TABLE ohlcv ADD COLUMN source TEXT DEFAULT 'binance'")
                    if "asset" not in cols:
                        alter_stmts.append("ALTER TABLE ohlcv ADD COLUMN asset TEXT DEFAULT ''")
                    if "tf" not in cols:
                        alter_stmts.append("ALTER TABLE ohlcv ADD COLUMN tf TEXT DEFAULT '1h'")
                    for stmt in alter_stmts:
                        try:
                            await conn.exec_driver_sql(stmt + ";")
                        except Exception:
                            pass  # гонка/повтор — игнор

                    await conn.exec_driver_sql(
                        "CREATE UNIQUE INDEX IF NOT EXISTS idx_ohlcv_unique ON ohlcv (source, asset, tf, ts);"
                    )
            except Exception:
                pass  # best-effort миграция

        if IS_POSTGRES:
            if PG_ENABLE_TIMESCALE:
                try:
                    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb;"))
                except Exception:
                    pass
            if PG_TIMEZONE:
                try:
                    await conn.execute(text(f"SET TIME ZONE '{PG_TIMEZONE}';"))
                except Exception:
                    pass

        # Создание таблиц (если без Alembic)
        await conn.run_sync(Base.metadata.create_all)


# Синоним
async def init_db_schema() -> None:
    await apply_startup_pragmas_and_schema()


# ──────────────────────────────────────────────────────────────────────────────
# Грейсфул выключение
# ──────────────────────────────────────────────────────────────────────────────
async def shutdown_engine() -> None:
    try:
        await engine.dispose()
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Health-check
# ──────────────────────────────────────────────────────────────────────────────
async def db_healthcheck() -> bool:
    try:
        async with engine.begin() as conn:
            await conn.exec_driver_sql("SELECT 1;")
        return True
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Утилиты для unit-тестов (in-memory SQLite)
# ──────────────────────────────────────────────────────────────────────────────
def make_test_engine_and_session(
    url: str = "sqlite+aiosqlite:///:memory:",
) -> Tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    test_engine = create_async_engine(url, future=True, echo=False)

    @event.listens_for(test_engine.sync_engine, "connect")
    def _test_pragmas(dbapi_connection, connection_record):  # type: ignore
        try:
            c = dbapi_connection.cursor()
            c.execute("PRAGMA foreign_keys=ON;")
            c.execute("PRAGMA synchronous=OFF;")
            c.close()
        except Exception:
            pass

    test_session = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)
    return test_engine, test_session


# ──────────────────────────────────────────────────────────────────────────────
# Периодические снапшоты SQLite (VACUUM INTO) + checkpoint WAL
# ──────────────────────────────────────────────────────────────────────────────
_snapshot_task: Optional[asyncio.Task] = None


async def _sqlite_take_snapshot_once(
    *,
    dest_dir: Path,
    retention_days: int,
) -> Optional[Path]:
    """
    «Горячий» снапшот файловой SQLite через VACUUM INTO (SQLite ≥ 3.27).
    Делает wal_checkpoint(TRUNCATE) перед VACUUM, чистит старые снапшоты.
    """
    if not IS_SQLITE or _is_memory_sqlite(DB_URL):
        return None

    db_path = Path(DB_URL.replace("sqlite+aiosqlite:///", ""))
    dest_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_path = dest_dir / f"{db_path.stem}.{ts}.db"

    async with engine.begin() as conn:
        try:
            await conn.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass

        await conn.exec_driver_sql(f"VACUUM INTO '{out_path.as_posix()}';")

    cutoff = datetime.now(timezone.utc) - timedelta(days=max(0, retention_days))
    for f in dest_dir.glob(f"{db_path.stem}.*.db"):
        try:
            mtime = datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc)
            if mtime < cutoff:
                f.unlink(missing_ok=True)
        except Exception:
            pass

    return out_path


async def _sqlite_snapshot_loop(
    *,
    interval_sec: int,
    dest_dir: Path,
    retention_days: int,
    stop_event: asyncio.Event,
) -> None:
    try:
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=max(5, interval_sec))
            except asyncio.TimeoutError:
                try:
                    await _sqlite_take_snapshot_once(dest_dir=dest_dir, retention_days=retention_days)
                except Exception:
                    pass
                continue
            break
    finally:
        # На выходе ещё раз чекпоинт
        try:
            async with engine.begin() as conn:
                await conn.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE);")
        except Exception:
            pass


def start_sqlite_snapshot_task(
    *,
    interval_sec: Optional[int] = None,
    dest_dir: Optional[Path] = None,
    retention_days: Optional[int] = None,
) -> Optional[Callable[[], Awaitable[None]]]:
    """
    Запускает фоновую задачу периодических снапшотов SQLite.
    Возвращает асинхронный stop()-колбэк для корректной остановки.
    Ничего не делает для PostgreSQL или in-memory SQLite.
    """
    global _snapshot_task

    if not IS_SQLITE or _is_memory_sqlite(DB_URL):
        return None

    interval = int(interval_sec or SQLITE_SNAPSHOT_INTERVAL_SEC)
    if interval <= 0:
        return None

    directory = dest_dir or SQLITE_SNAPSHOT_DIR
    retention = int(retention_days or SQLITE_SNAPSHOT_RETENTION)

    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    async def _stop() -> None:
        if _snapshot_task and not _snapshot_task.done():
            stop_event.set()
            try:
                await _snapshot_task
            except asyncio.CancelledError:
                pass

    if _snapshot_task and not _snapshot_task.done():
        return _stop

    _snapshot_task = loop.create_task(
        _sqlite_snapshot_loop(
            interval_sec=interval,
            dest_dir=directory,
            retention_days=retention,
            stop_event=stop_event,
        )
    )
    return _stop


# ──────────────────────────────────────────────────────────────────────────────
# Экспорт API модуля
# ──────────────────────────────────────────────────────────────────────────────
__all__ = [
    "Base",
    "engine",
    "AsyncSessionLocal",
    "get_session",
    "apply_startup_pragmas_and_schema",
    "init_db_schema",
    "shutdown_engine",
    "db_healthcheck",
    "make_test_engine_and_session",
    "start_sqlite_snapshot_task",
    # Дополнительно экспортируем диалектные флаги — удобно для логов/диагностики
    "IS_SQLITE",
    "IS_POSTGRES",
    "DB_URL",
]
