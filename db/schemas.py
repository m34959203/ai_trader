from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class OHLCVBase(BaseModel):
    """Базовая схема OHLCV данных."""
    model_config = ConfigDict(from_attributes=True)
    
    source: str = Field(..., description="Источник данных")
    asset: str = Field(..., description="Торговая пара") 
    tf: str = Field(..., description="Таймфрейм")
    ts: int = Field(..., description="Временная метка")
    open: float = Field(..., description="Цена открытия")
    high: float = Field(..., description="Максимальная цена")
    low: float = Field(..., description="Минимальная цена")
    close: float = Field(..., description="Цена закрытия")
    volume: float = Field(..., description="Объем")

class OHLCVCreate(OHLCVBase):
    """Схема для создания новой OHLCV записи."""
    pass

class OHLCVResponse(OHLCVBase):
    """Схема ответа с OHLCV данными."""
    id: int

class PaginationInfo(BaseModel):
    """Информация о пагинации."""
    total: int
    limit: int
    offset: int
    next_offset: Optional[int]
    has_next_page: bool
    total_pages: int

class OHLCVListResponse(BaseModel):
    """Ответ со списком OHLCV данных и пагинацией."""
    candles: List[OHLCVResponse]
    pagination: PaginationInfo
    metadata: Dict[str, Any]