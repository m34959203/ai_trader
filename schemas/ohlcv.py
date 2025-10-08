from __future__ import annotations

from typing import Optional, List, Literal, Dict, Any
from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, Field, validator, ConfigDict


class StoreRequest(BaseModel):
    """Схема для запроса сохранения OHLCV данных."""
    model_config = ConfigDict(extra='forbid')  # Запрещаем лишние поля
    
    source: str = Field(..., description="Источник данных (binance, alpha_vantage, etc.)")
    symbol: str = Field(..., description="Торговая пара (BTCUSDT, ETHUSDT, etc.)", examples=["BTCUSDT"])
    timeframe: Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d"] = Field(..., description="Таймфрейм")
    limit: Optional[int] = Field(default=1000, ge=1, le=10000, description="Лимит записей для загрузки")
    ts_from: Optional[int] = Field(default=None, ge=0, description="Начальная временная метка (timestamp)")
    ts_to: Optional[int] = Field(default=None, ge=0, description="Конечная временная метка (timestamp)")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        """Валидация символа - приводим к верхнему регистру."""
        if not v or len(v.strip()) == 0:
            raise ValueError('Symbol cannot be empty')
        return v.strip().upper()
    
    @validator('ts_to')
    def validate_timestamps(cls, v, values):
        """Валидация временного диапазона."""
        if v is not None and 'ts_from' in values and values['ts_from'] is not None:
            if v <= values['ts_from']:
                raise ValueError('ts_to must be greater than ts_from')
        return v


class StoreResponse(BaseModel):
    """Схема ответа на запрос сохранения данных."""
    stored: int = Field(..., ge=0, description="Количество сохраненных записей")
    source: str = Field(..., description="Источник данных")
    symbol: str = Field(..., description="Торговая пара")
    timeframe: str = Field(..., description="Таймфрейм")
    duplicates_skipped: Optional[int] = Field(default=0, ge=0, description="Количество пропущенных дубликатов")
    errors: Optional[int] = Field(default=0, ge=0, description="Количество ошибок при обработке")
    
    model_config = ConfigDict(extra='forbid')


class Candle(BaseModel):
    """Схема представления одной свечи."""
    source: str = Field(..., description="Источник данных")
    asset: str = Field(..., description="Торговая пара")
    tf: str = Field(..., description="Таймфрейм")
    ts: int = Field(..., description="Временная метка Unix")
    open: float = Field(..., ge=0, description="Цена открытия")
    high: float = Field(..., ge=0, description="Максимальная цена")
    low: float = Field(..., ge=0, description="Минимальная цена")
    close: float = Field(..., ge=0, description="Цена закрытия")
    volume: float = Field(..., ge=0, description="Объем торгов")
    
    model_config = ConfigDict(
        json_encoders={
            float: lambda v: float(Decimal(str(v)).quantize(Decimal('0.00000001'))),
        },
        extra='forbid'
    )
    
    @validator('high')
    def validate_high(cls, v, values):
        """Проверяем, что high >= open и high >= low."""
        if 'open' in values and v < values['open']:
            raise ValueError('High must be >= open')
        if 'low' in values and v < values['low']:
            raise ValueError('High must be >= low')
        return v
    
    @validator('low')
    def validate_low(cls, v, values):
        """Проверяем, что low <= open и low <= high."""
        if 'open' in values and v > values['open']:
            raise ValueError('Low must be <= open')
        if 'high' in values and v > values['high']:
            raise ValueError('Low must be <= high')
        return v
    
    @property
    def datetime(self) -> datetime:
        """Возвращает datetime объект для временной метки."""
        return datetime.fromtimestamp(self.ts)
    
    @property
    def is_bullish(self) -> bool:
        """Возвращает True если свеча бычья (close > open)."""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """Возвращает True если свеча медвежья (close < open)."""
        return self.close < self.open
    
    @property
    def body_size(self) -> float:
        """Возвращает размер тела свечи."""
        return abs(self.close - self.open)
    
    @property
    def total_range(self) -> float:
        """Возвращает полный диапазон свечи."""
        return self.high - self.low


class QueryRequest(BaseModel):
    """Схема запроса для получения OHLCV данных."""
    source: str = Field(..., description="Источник данных")
    ticker: str = Field(..., description="Торговая пара", examples=["BTCUSDT"])
    timeframe: str = Field(..., description="Таймфрейм", examples=["1h", "4h", "1d"])
    limit: int = Field(default=100, ge=1, le=5000, description="Лимит записей")
    offset: int = Field(default=0, ge=0, description="Смещение для пагинации")
    order: Literal["asc", "desc"] = Field(default="desc", description="Порядок сортировки")
    ts_from: Optional[int] = Field(default=None, description="Начальная временная метка")
    ts_to: Optional[int] = Field(default=None, description="Конечная временная метка")
    
    model_config = ConfigDict(extra='forbid')
    
    @validator('ticker')
    def validate_ticker(cls, v):
        """Валидация тикера."""
        if not v or len(v.strip()) == 0:
            raise ValueError('Ticker cannot be empty')
        return v.strip().upper()


class QueryResponse(BaseModel):
    """Базовый ответ с данными свечей."""
    candles: List[Candle] = Field(default_factory=list, description="Список свечей")
    
    model_config = ConfigDict(extra='forbid')
    
    @property
    def count(self) -> int:
        """Количество свечей в ответе."""
        return len(self.candles)
    
    @property
    def first_timestamp(self) -> Optional[int]:
        """Первая временная метка в данных."""
        return self.candles[0].ts if self.candles else None
    
    @property
    def last_timestamp(self) -> Optional[int]:
        """Последняя временная метка в данных."""
        return self.candles[-1].ts if self.candles else None


class QueryResponseWithPage(QueryResponse):
    """Ответ с пагинацией."""
    next_offset: Optional[int] = Field(
        default=None,
        description="Смещение для следующей страницы; null, если дальше данных нет",
        ge=0
    )
    total: int = Field(..., ge=0, description="Общее количество записей")
    limit: int = Field(..., ge=1, le=5000, description="Использованный лимит")
    offset: int = Field(..., ge=0, description="Использованное смещение")
    has_next_page: bool = Field(..., description="Есть ли следующая страница")
    
    model_config = ConfigDict(extra='forbid')


class OHLCVStats(BaseModel):
    """Статистика по OHLCV данным."""
    source: str = Field(..., description="Источник данных")
    symbol: str = Field(..., description="Торговая пара")
    timeframe: str = Field(..., description="Таймфрейм")
    total_candles: int = Field(..., ge=0, description="Общее количество свечей")
    first_timestamp: Optional[int] = Field(None, description="Первая временная метка")
    last_timestamp: Optional[int] = Field(None, description="Последняя временная метка")
    time_range_days: Optional[float] = Field(None, description="Диапазон времени в днях")
    
    model_config = ConfigDict(extra='forbid')


class BulkStoreResponse(BaseModel):
    """Ответ на массовое сохранение данных."""
    operations: List[StoreResponse] = Field(..., description="Результаты операций")
    total_stored: int = Field(..., ge=0, description="Всего сохранено записей")
    total_errors: int = Field(..., ge=0, description="Всего ошибок")
    processing_time: float = Field(..., ge=0, description="Время обработки в секундах")
    
    model_config = ConfigDict(extra='forbid')


# Утилитарные классы для внутреннего использования
class OHLCVCreate(BaseModel):
    """Схема для создания OHLCV записи (внутреннее использование)."""
    source: str
    asset: str
    tf: str
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    model_config = ConfigDict(extra='forbid')


class OHLCVUpdate(BaseModel):
    """Схема для обновления OHLCV записи (внутреннее использование)."""
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[float] = None
    
    model_config = ConfigDict(extra='forbid')


# Response модели для API эндпоинтов
class SuccessResponse(BaseModel):
    """Успешный ответ API."""
    success: bool = Field(True, description="Статус операции")
    message: str = Field(..., description="Сообщение")
    data: Optional[Dict[str, Any]] = Field(None, description="Дополнительные данные")
    
    model_config = ConfigDict(extra='forbid')


class ErrorResponse(BaseModel):
    """Ответ с ошибкой API."""
    success: bool = Field(False, description="Статус операции")
    error: str = Field(..., description="Текст ошибки")
    code: str = Field(..., description="Код ошибки")
    details: Optional[Dict[str, Any]] = Field(None, description="Детали ошибки")
    
    model_config = ConfigDict(extra='forbid')


# Дополнительные схемы для аналитики
class PriceMovement(BaseModel):
    """Информация о движении цены."""
    absolute: float = Field(..., description="Абсолютное изменение")
    percentage: float = Field(..., description="Процентное изменение")
    direction: Literal["up", "down", "unchanged"] = Field(..., description="Направление")


class CandleAnalysis(BaseModel):
    """Анализ отдельной свечи."""
    candle: Candle = Field(..., description="Анализируемая свеча")
    is_doji: bool = Field(..., description="Является ли доджи")
    body_ratio: float = Field(..., description="Отношение тела к общему диапазону")
    volume_profile: Literal["low", "medium", "high"] = Field(..., description="Профиль объема")


# Экспортируем основные схемы
__all__ = [
    "StoreRequest",
    "StoreResponse", 
    "Candle",
    "QueryRequest",
    "QueryResponse",
    "QueryResponseWithPage",
    "OHLCVStats",
    "BulkStoreResponse",
    "SuccessResponse",
    "ErrorResponse",
    "PriceMovement",
    "CandleAnalysis",
]