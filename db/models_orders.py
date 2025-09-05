# db/models_orders.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    String,
    Float,
    Integer,
    BigInteger,
    Index,
    text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

# В проекте уже есть единая декларативная база
from .session import Base


def _utcnow() -> datetime:
    # Явный UTC, чтобы не плодить локальные таймзоны
    return datetime.now(timezone.utc)


class OrderLog(Base):
    """
    Журнал биржевых и симулированных ордеров.
    Назначение:
      - Трассировка всех попыток торгового исполнения (успешных и неуспешных)
      - Сверка с биржей (reconcile)
      - Поддержка риск-ограничений и пост-фактум аналитики (PnL, частота, отмены)

    Важные примечания:
      - 'testnet' хранится как 0/1 для переносимости между SQLite/Postgres.
      - Денежные поля как float: для SQLIte приемлемо. Если перейдем на Postgres — можно заменить на DECIMAL.
      - Поля *_ts_ms хранятся в миллисекундах от биржи (event time, transact time и т.д.), если они приходят.
    """

    __tablename__ = "order_log"

    # --- PK ---
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    # --- источник исполнения ---
    exchange: Mapped[str] = mapped_column(String(16), nullable=False, server_default="binance")
    testnet: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))  # 1/0

    # --- символ/направление/вид ---
    symbol: Mapped[str] = mapped_column(String(24), nullable=False, index=True)  # например, BTCUSDT
    side: Mapped[str] = mapped_column(String(4), nullable=False)                 # BUY | SELL
    type: Mapped[str] = mapped_column(String(16), nullable=False)                # MARKET | LIMIT | ...
    time_in_force: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)  # GTC | IOC | FOK

    status: Mapped[str] = mapped_column(String(24), nullable=False, server_default="NEW")
    # Возможные статусы (ориентир Binance): NEW, PARTIALLY_FILLED, FILLED, CANCELED,
    # PENDING_CANCEL, REJECTED, EXPIRED, EXPIRED_IN_MATCH

    # --- идентификаторы ордера ---
    order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)            # биржевой ордерId
    client_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)     # newClientOrderId
    orig_client_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)  # для замен/отмен

    # --- ценовые/количественные параметры ---
    price: Mapped[float] = mapped_column(Float, nullable=False, server_default=text("0"))         # лимитная цена
    stop_price: Mapped[float] = mapped_column(Float, nullable=False, server_default=text("0"))    # стоп-триггер
    qty: Mapped[float] = mapped_column(Float, nullable=False, server_default=text("0"))           # базовый объем
    quote_qty: Mapped[float] = mapped_column(Float, nullable=False, server_default=text("0"))     # котируемый объем (для MARKET по quoteOrderQty)
    filled_qty: Mapped[float] = mapped_column(Float, nullable=False, server_default=text("0"))
    cummulative_quote_qty: Mapped[float] = mapped_column(Float, nullable=False, server_default=text("0"))  # quote, из ответа Binance

    commission: Mapped[float] = mapped_column(Float, nullable=False, server_default=text("0"))
    commission_asset: Mapped[Optional[str]] = mapped_column(String(12), nullable=True)

    # --- метки времени от биржи (мс) ---
    event_ts_ms: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)      # например, E из stream
    transact_ts_ms: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)   # transactTime из ответа
    order_ts_ms: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)      # время создания/оригинала

    # --- технические таймштампы записи ---
    created_at: Mapped[datetime] = mapped_column(nullable=False, default=_utcnow, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
        server_default=func.now(),
    )

    # --- полезные представления ---
    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "exchange": self.exchange,
            "testnet": bool(self.testnet),
            "symbol": self.symbol,
            "side": self.side,
            "type": self.type,
            "time_in_force": self.time_in_force,
            "status": self.status,
            "order_id": self.order_id,
            "client_order_id": self.client_order_id,
            "orig_client_order_id": self.orig_client_order_id,
            "price": float(self.price),
            "stop_price": float(self.stop_price),
            "qty": float(self.qty),
            "quote_qty": float(self.quote_qty),
            "filled_qty": float(self.filled_qty),
            "cummulative_quote_qty": float(self.cummulative_quote_qty),
            "commission": float(self.commission),
            "commission_asset": self.commission_asset,
            "event_ts_ms": self.event_ts_ms,
            "transact_ts_ms": self.transact_ts_ms,
            "order_ts_ms": self.order_ts_ms,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"<OrderLog id={self.id} {self.exchange} "
            f"{'TESTNET' if self.testnet else 'LIVE'} {self.symbol} "
            f"{self.side}/{self.type} status={self.status} qty={self.qty} price={self.price}>"
        )


# Индексы для частых выборок и уникальности на клиентских id
Index("ix_orderlog_symbol_created", OrderLog.symbol, OrderLog.created_at.desc())
Index("ix_orderlog_client_id", OrderLog.client_order_id)
Index("ix_orderlog_exchange_testnet", OrderLog.exchange, OrderLog.testnet)
