# db/models.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict

from sqlalchemy import BigInteger, Float, String, Index
from sqlalchemy.orm import Mapped, mapped_column

from .session import Base


class OHLCV(Base):
    """
    Унифицированная таблица свечей.

    Обязательные поля (ожидаются CRUD/тестами):
      - source : str   (например, "binance")
      - asset  : str   (например, "BTCUSDT")
      - tf     : str   (например, "1h")
      - ts     : int   (unix seconds, UTC)
      - open, high, low, close, volume : float

    Первичный ключ составной: (source, asset, tf, ts)
    Это даёт ON CONFLICT (source, asset, tf, ts) ... для SQLite.
    """

    __tablename__ = "ohlcv"

    # Составной PK (ровно те же поля, что и в ON CONFLICT)
    source: Mapped[str] = mapped_column(String(32), primary_key=True, nullable=False)
    asset:  Mapped[str] = mapped_column(String(64), primary_key=True, nullable=False)
    tf:     Mapped[str] = mapped_column(String(16), primary_key=True, nullable=False)
    ts:     Mapped[int] = mapped_column(BigInteger, primary_key=True, nullable=False)  # unix seconds (UTC)

    # Значения бара
    open:   Mapped[float] = mapped_column(Float, nullable=False)
    high:   Mapped[float] = mapped_column(Float, nullable=False)
    low:    Mapped[float] = mapped_column(Float, nullable=False)
    close:  Mapped[float] = mapped_column(Float, nullable=False)
    volume: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Полезные индексы для типичных выборок
    __table_args__ = (
        Index("ix_ohlcv_source_asset_tf_ts", "source", "asset", "tf", "ts"),
        Index("ix_ohlcv_asset_tf_ts", "asset", "tf", "ts"),
        Index("ix_ohlcv_ts", "ts"),
    )

    # ── Удобства для отладки/сериализации ─────────────────────────────────────
    @property
    def timestamp_dt(self) -> datetime:
        """ts → datetime (UTC)."""
        return datetime.fromtimestamp(int(self.ts), tz=timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "asset": self.asset,
            "tf": self.tf,
            "ts": self.ts,
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": float(self.volume),
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            "OHLCV("
            f"source={self.source!r}, asset={self.asset!r}, tf={self.tf!r}, ts={self.ts}"
            f" {self.timestamp_dt.isoformat()}, "
            f"o={self.open}, h={self.high}, l={self.low}, c={self.close}, v={self.volume})"
        )
