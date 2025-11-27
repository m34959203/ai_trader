# План реализации недостающих компонентов AI-Трейдера

**Дата создания:** 2025-11-27
**Базовая готовность:** 65-70%
**Целевая готовность:** 95%+

---

## 🎯 ЭТАП 1: AI Модели MVP (2-3 недели)

### 1.1 Triple-Barrier Labeling
**Срок:** 3 дня
**Приоритет:** 🔴 Критический
**Файл:** `src/models/labeling/triple_barrier.py`

#### Спецификация:
```python
def triple_barrier_labels(
    prices: pd.Series,
    *,
    profit_target: float,  # PT barrier (в ATR множителях)
    stop_loss: float,      # SL barrier (в ATR множителях)
    max_holding: int,      # Временной барьер (в барах)
    atr: pd.Series,        # ATR для динамических барьеров
) -> pd.DataFrame:
    """
    Генерирует метки по методу Triple Barrier.

    Returns:
        DataFrame с колонками:
        - label: {-1, 0, 1}
        - barrier_hit: {'profit', 'stop', 'time'}
        - holding_period: int
        - return_pct: float
    """
```

#### Алгоритм:
1. Для каждой свечи установить 3 барьера:
   - Upper: entry_price + (profit_target * ATR)
   - Lower: entry_price - (stop_loss * ATR)
   - Time: max_holding bars вперед
2. Определить, какой барьер коснулся первым
3. Присвоить метку:
   - +1: коснулся Upper (profit)
   - -1: коснулся Lower (stop loss)
   - 0: коснулся Time (timeout)

#### Тесты:
- `tests/test_triple_barrier.py`:
  - Проверка корректности меток
  - Edge cases (все 3 барьера на одной свече)
  - Производительность на больших датасетах

#### Зависимости:
```python
# requirements.txt
pandas>=1.5.0
numpy>=1.24.0
```

---

### 1.2 LSTM для временных рядов
**Срок:** 5 дней
**Приоритет:** 🔴 Критический
**Файл:** `src/models/forecast/lstm_model.py`

#### Архитектура:
```python
class LSTMForecaster:
    """
    LSTM модель для прогнозирования следующих N свечей.

    Architecture:
        Input: [batch, sequence_length, features]
        LSTM1: 128 units, return_sequences=True, dropout=0.2
        LSTM2: 64 units, return_sequences=False, dropout=0.2
        Dense1: 32 units, ReLU
        Dense2: n_forecast units (close prices)

    Features (по умолчанию):
        - close, high, low, open
        - volume
        - RSI, MACD, ATR
        - Returns (log)
    """

    def __init__(
        self,
        sequence_length: int = 60,    # 60 баров для обучения
        n_forecast: int = 3,           # Предсказать 3 свечи
        features: list[str] = None,
        lstm_units: tuple[int, int] = (128, 64),
        dropout: float = 0.2,
    ):
        ...

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        validation_split: float = 0.2,
        epochs: int = 100,
        batch_size: int = 32,
        early_stopping_patience: int = 10,
    ) -> dict:
        """Обучение модели с early stopping."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказание следующих n_forecast свечей."""
        ...

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Метрики:
        - MSE, RMSE, MAE
        - Directional accuracy (правильное предсказание направления)
        - R² score
        """
        ...
```

#### Препроцессинг:
```python
class LSTMDataPreprocessor:
    """Подготовка данных для LSTM."""

    def __init__(self, scaler_type: str = "minmax"):
        """
        scaler_type: 'minmax', 'standard', 'robust'
        """
        ...

    def create_sequences(
        self,
        df: pd.DataFrame,
        sequence_length: int,
        n_forecast: int,
        features: list[str],
        target: str = "close",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Создание последовательностей для обучения.

        Returns:
            X: [n_samples, sequence_length, n_features]
            y: [n_samples, n_forecast]
        """
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit scaler и трансформация."""
        ...

    def inverse_transform(self, predictions: np.ndarray) -> np.ndarray:
        """Обратная трансформация предсказаний."""
        ...
```

#### Интеграция с ModelRouter:
```python
# services/model_router.py

from src.models.forecast.lstm_model import LSTMForecaster

class ModelRouter:
    def __init__(self, config: ExecutionConfig):
        ...
        if config.forecast.name == "lstm":
            self.forecaster = LSTMForecaster.load(
                config.forecast.params.get("model_path")
            )
```

#### Конфигурация:
```yaml
# configs/exec.yaml

models:
  signal: "signal:rf_rule"
  sentiment: "sentiment:finbert"
  regime: "regime:kmeans"
  forecast: "forecast:lstm"  # НОВОЕ

forecast:
  lstm:
    model_path: "models/lstm_v1.h5"
    sequence_length: 60
    n_forecast: 3
    features:
      - close
      - high
      - low
      - open
      - volume
      - rsi
      - macd
      - atr
```

#### Тесты:
- `tests/test_lstm_model.py`:
  - Обучение на синтетических данных
  - Предсказание формы выхода
  - Directional accuracy > 55%
  - Интеграция с ModelRouter

#### Зависимости:
```python
# requirements.txt (обновить)
tensorflow>=2.13.0  # или pytorch>=2.0.0
scikit-learn>=1.3.0
```

---

### 1.3 Purged Walk-Forward Cross-Validation
**Срок:** 3 дня
**Приоритет:** 🔴 Критический
**Файл:** `src/models/validation/purged_cv.py`

#### Спецификация:
```python
class PurgedWalkForwardCV:
    """
    Walk-forward валидация с purging и embargo.

    Предотвращает leakage между train/test:
    - Purging: удаляет overlapping samples
    - Embargo: gap между train и test
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_period: int = 252,  # ~1 год
        test_period: int = 63,    # ~3 месяца
        embargo_period: int = 21, # ~1 месяц
        purge_pct: float = 0.01,  # 1% перекрытия удалить
    ):
        ...

    def split(self, X: pd.DataFrame, y: pd.Series) -> Iterator[tuple]:
        """
        Генератор train/test индексов.

        Yields:
            (train_idx, test_idx) для каждого fold
        """
        ...

    def cross_val_score(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        scoring: str = "accuracy",
    ) -> dict:
        """
        Кросс-валидация с метриками.

        Returns:
            {
                'scores': [score1, score2, ...],
                'mean': float,
                'std': float,
                'sharpe': float,  # если scoring == 'returns'
                'max_dd': float,
            }
        """
        ...
```

#### Визуализация:
```python
def plot_cv_splits(cv: PurgedWalkForwardCV, X: pd.DataFrame):
    """
    График train/test splits на временной шкале.

    Показывает:
    - Train periods (синий)
    - Test periods (зеленый)
    - Embargo gaps (красный)
    """
    ...
```

#### Пример использования:
```python
from src.models.validation.purged_cv import PurgedWalkForwardCV
from src.models.signal.random_forest_rule import RandomForestSignal

# Подготовка данных
X = df[['rsi', 'macd', 'atr', 'volume']]
y = triple_barrier_labels(df['close'])

# Кросс-валидация
cv = PurgedWalkForwardCV(
    n_splits=5,
    train_period=252,
    test_period=63,
    embargo_period=21,
)

model = RandomForestSignal()
results = cv.cross_val_score(model, X, y, scoring='accuracy')

print(f"Mean accuracy: {results['mean']:.3f} ± {results['std']:.3f}")
print(f"Sharpe ratio: {results['sharpe']:.2f}")
```

#### Тесты:
- `tests/test_purged_cv.py`:
  - Проверка отсутствия overlaps
  - Embargo period соблюдается
  - Количество splits корректно

---

### 1.4 Автоматическое переобучение моделей
**Срок:** 4 дня
**Приоритет:** 🔴 Критический
**Файл:** `tasks/model_retraining.py`

#### Спецификация:
```python
class ModelRetrainingPipeline:
    """
    Автоматическое переобучение моделей на новых данных.

    Workflow:
    1. Загрузить свежие данные из БД
    2. Генерировать метки (triple-barrier)
    3. Walk-forward валидация
    4. Обучить новую модель
    5. Сравнить с текущей (A/B test)
    6. Deploy если метрики лучше
    7. Rollback если хуже
    """

    def __init__(
        self,
        model_type: str,  # 'lstm', 'cnn', 'transformer'
        config_path: str,
        db_connection: str,
    ):
        ...

    async def run_retraining_cycle(self) -> dict:
        """
        Полный цикл переобучения.

        Returns:
            {
                'status': 'success' | 'failed' | 'rolled_back',
                'new_model_path': str,
                'metrics': {
                    'accuracy': float,
                    'sharpe': float,
                    'max_dd': float,
                },
                'comparison': {
                    'old_model': {...},
                    'new_model': {...},
                    'improvement_pct': float,
                },
            }
        """
        # 1. Загрузить данные
        df = await self._load_recent_data(days=365)

        # 2. Генерировать метки
        labels = triple_barrier_labels(df['close'], ...)

        # 3. Walk-forward CV
        cv_results = self._validate_model(df, labels)

        # 4. Обучить
        new_model = self._train_model(df, labels)

        # 5. A/B тест
        if self._is_better_than_current(new_model, cv_results):
            self._deploy_model(new_model)
            return {'status': 'success', ...}
        else:
            self._rollback()
            return {'status': 'rolled_back', ...}

    def _is_better_than_current(
        self,
        new_model,
        cv_results: dict,
        *,
        min_improvement_pct: float = 5.0,
    ) -> bool:
        """
        Проверка улучшения метрик.

        Критерии:
        - Sharpe ratio >= +5%
        - Max drawdown <= -5%
        - Directional accuracy >= +2%
        """
        ...

    def _deploy_model(self, model, version: str):
        """
        Deploy модели:
        1. Сохранить в models/{type}_v{version}.h5
        2. Обновить configs/exec.yaml
        3. Создать git commit
        4. Отправить уведомление в Telegram
        """
        ...

    def _rollback(self):
        """Откат к предыдущей версии."""
        ...
```

#### Scheduler (Celery/APScheduler):
```python
# tasks/scheduler.py

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from tasks.model_retraining import ModelRetrainingPipeline

scheduler = AsyncIOScheduler()

@scheduler.scheduled_job('cron', day_of_week='sun', hour=2)
async def weekly_lstm_retraining():
    """Переобучение LSTM каждое воскресенье в 2:00."""
    pipeline = ModelRetrainingPipeline(
        model_type='lstm',
        config_path='configs/exec.yaml',
        db_connection=settings.DATABASE_URL,
    )

    result = await pipeline.run_retraining_cycle()

    # Отправить отчет в Telegram
    await telegram_bot.send_message(
        f"LSTM Retraining: {result['status']}\n"
        f"New Sharpe: {result['metrics']['sharpe']:.2f}\n"
        f"Improvement: {result['comparison']['improvement_pct']:.1f}%"
    )

scheduler.start()
```

#### Интеграция в main.py:
```python
# src/main.py

@app.on_event("startup")
async def startup_tasks():
    ...

    # Запустить scheduler переобучения
    if settings.MODEL_RETRAINING_ENABLED:
        from tasks.scheduler import scheduler
        scheduler.start()
        logger.info("Model retraining scheduler started")
```

#### Конфигурация:
```yaml
# configs/exec.yaml

model_retraining:
  enabled: true
  schedule:
    lstm: "weekly"      # каждое воскресенье
    cnn: "biweekly"     # каждые 2 недели
    transformer: "monthly"
  min_improvement_pct: 5.0
  max_models_to_keep: 5  # история версий
  notification:
    telegram: true
    email: false
```

#### Версионирование моделей:
```
models/
├── lstm_v1.h5        (2025-01-15, sharpe=1.2)
├── lstm_v2.h5        (2025-01-22, sharpe=1.35)
├── lstm_v3.h5        (2025-01-29, sharpe=1.28) ← rollback
├── lstm_current.h5 -> lstm_v2.h5  (symlink к best)
└── metadata.json
```

#### Тесты:
- `tests/test_model_retraining.py`:
  - Mock полного цикла
  - A/B comparison logic
  - Rollback механизм
  - Версионирование

#### Зависимости:
```python
# requirements.txt
apscheduler>=3.10.0  # или celery>=5.3.0
mlflow>=2.8.0        # опционально, для tracking
```

---

## 🎯 ЭТАП 2: Уведомления и стабильность (1 неделя)

### 2.1 Telegram-бот
**Срок:** 5 дней
**Приоритет:** 🟡 Высокий
**Файл:** `services/telegram_bot.py`

#### Спецификация:
```python
class TradingTelegramBot:
    """
    Telegram бот для мониторинга и управления трейдером.

    Команды:
    - /start - приветствие и помощь
    - /status - статус системы
    - /pnl - текущий PnL
    - /positions - открытые позиции
    - /trades - последние сделки
    - /stop - экстренная остановка
    - /resume - возобновить торговлю
    - /limits - дневные лимиты
    - /config - текущая конфигурация

    Уведомления:
    - Новые сделки (открытие/закрытие)
    - Ошибки и алерты
    - Достижение дневных лимитов
    - Результаты переобучения моделей
    - Дневные отчеты
    """

    def __init__(
        self,
        bot_token: str,
        allowed_users: list[int],  # Telegram user IDs
        trading_service: TradingService,
    ):
        ...

    async def start(self):
        """Запуск бота."""
        self.app = Application.builder().token(self.bot_token).build()

        # Регистрация handlers
        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("pnl", self.cmd_pnl))
        self.app.add_handler(CommandHandler("positions", self.cmd_positions))
        self.app.add_handler(CommandHandler("stop", self.cmd_emergency_stop))

        # Inline кнопки для emergency stop
        self.app.add_handler(CallbackQueryHandler(self.btn_confirm_stop))

        await self.app.run_polling()

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Команда /status

        Показывает:
        - Состояние системы (running/stopped)
        - Соединение с биржей
        - Количество активных позиций
        - Текущий equity
        - Дневной PnL
        """
        status = await self.trading_service.get_status()

        message = f"""
🤖 **AI Trader Status**

System: {'🟢 Running' if status.running else '🔴 Stopped'}
Broker: {'🟢 Connected' if status.broker_connected else '🔴 Disconnected'}

💼 Positions: {status.open_positions}
💰 Equity: ${status.equity:,.2f}
📊 Day PnL: ${status.day_pnl:+,.2f} ({status.day_pnl_pct:+.2f}%)

⚠️ Day Trades: {status.day_trades}/{status.max_day_trades}
🛑 Day Loss Limit: ${status.day_loss_limit:,.2f}
        """

        await update.message.reply_text(message, parse_mode='Markdown')

    async def cmd_emergency_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Команда /stop с подтверждением

        Показывает inline кнопки:
        - ✅ Confirm Stop
        - ❌ Cancel
        """
        keyboard = [
            [
                InlineKeyboardButton("✅ Confirm STOP", callback_data="stop_confirmed"),
                InlineKeyboardButton("❌ Cancel", callback_data="stop_cancelled"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text(
            "⚠️ **EMERGENCY STOP**\n\n"
            "This will:\n"
            "- Close all positions at market price\n"
            "- Cancel all pending orders\n"
            "- Pause trading\n\n"
            "Confirm?",
            reply_markup=reply_markup,
            parse_mode='Markdown',
        )

    async def btn_confirm_stop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка нажатия кнопки подтверждения."""
        query = update.callback_query
        await query.answer()

        if query.data == "stop_confirmed":
            result = await self.trading_service.emergency_stop()

            await query.edit_message_text(
                f"🛑 **EMERGENCY STOP EXECUTED**\n\n"
                f"Closed positions: {result.closed_positions}\n"
                f"Cancelled orders: {result.cancelled_orders}\n"
                f"Final PnL: ${result.final_pnl:+,.2f}"
            )
        else:
            await query.edit_message_text("❌ Emergency stop cancelled")

    async def send_trade_notification(self, trade: Trade):
        """
        Уведомление о новой сделке.

        Формат:
        📈 LONG BTC/USDT OPENED
        Entry: $45,123.45
        Size: 0.05 BTC ($2,256)
        SL: $44,000 | TP: $47,000
        Confidence: 78%
        """
        direction = "📈 LONG" if trade.side == "buy" else "📉 SHORT"
        action = "OPENED" if trade.action == "open" else "CLOSED"

        message = f"""
{direction} {trade.symbol} {action}

{'Entry' if trade.action == 'open' else 'Exit'}: ${trade.price:,.2f}
Size: {trade.quantity:.4f} ({trade.notional:,.0f} USDT)
"""

        if trade.action == "open":
            message += f"""
SL: ${trade.stop_loss:,.2f} | TP: ${trade.take_profit:,.2f}
Confidence: {trade.confidence:.0f}%
Reason: {trade.reason}
"""
        else:
            message += f"""
PnL: ${trade.pnl:+,.2f} ({trade.pnl_pct:+.2f}%)
Duration: {trade.duration}
Reason: {trade.close_reason}
"""

        await self.send_to_all_users(message)

    async def send_daily_report(self):
        """
        Ежедневный отчет (отправляется в 00:00 UTC).

        Формат:
        📊 Daily Report - 2025-01-27

        Trades: 12
        Win rate: 58.3% (7W/5L)
        PnL: +$1,234.56 (+1.23%)
        Best trade: +$456.78 (BTC/USDT LONG)
        Worst trade: -$123.45 (ETH/USDT SHORT)

        Sharpe: 1.85
        Max DD: -2.3%
        """
        report = await self.trading_service.get_daily_report()

        message = f"""
📊 **Daily Report** - {report.date}

📈 Trades: {report.total_trades}
✅ Win rate: {report.win_rate:.1f}% ({report.wins}W/{report.losses}L)
💰 PnL: ${report.pnl:+,.2f} ({report.pnl_pct:+.2f}%)

🏆 Best: +${report.best_trade:,.2f} ({report.best_symbol})
💔 Worst: ${report.worst_trade:+,.2f} ({report.worst_symbol})

📊 Sharpe: {report.sharpe:.2f}
📉 Max DD: {report.max_dd:.2f}%
        """

        await self.send_to_all_users(message, parse_mode='Markdown')

    async def send_to_all_users(self, message: str, **kwargs):
        """Отправка сообщения всем разрешенным пользователям."""
        for user_id in self.allowed_users:
            try:
                await self.app.bot.send_message(user_id, message, **kwargs)
            except Exception as e:
                logger.error(f"Failed to send message to {user_id}: {e}")
```

#### Интеграция в main.py:
```python
# src/main.py

telegram_bot: Optional[TradingTelegramBot] = None

@app.on_event("startup")
async def startup_telegram():
    global telegram_bot

    if settings.TELEGRAM_BOT_ENABLED:
        telegram_bot = TradingTelegramBot(
            bot_token=settings.TELEGRAM_BOT_TOKEN,
            allowed_users=settings.TELEGRAM_ALLOWED_USERS,
            trading_service=trading_service,
        )

        asyncio.create_task(telegram_bot.start())
        logger.info("Telegram bot started")
```

#### Конфигурация:
```yaml
# configs/exec.yaml

telegram:
  enabled: true
  bot_token_env: TELEGRAM_BOT_TOKEN
  allowed_users:
    - 123456789  # Ваш Telegram user ID
  notifications:
    trades: true
    errors: true
    daily_report: true
    daily_report_time: "00:00"  # UTC
    retraining_results: true
  rate_limit:
    max_messages_per_minute: 10
```

#### Environment variables:
```bash
# .env
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_ALLOWED_USERS=123456789,987654321
```

#### Тесты:
- `tests/test_telegram_bot.py`:
  - Mock команды
  - Emergency stop workflow
  - Уведомления
  - Rate limiting

#### Зависимости:
```python
# requirements.txt
python-telegram-bot>=20.7
```

---

### 2.2 Auto-restart инфраструктура
**Срок:** 2 дня
**Приоритет:** 🟡 Высокий

#### 2.2.1 Systemd service
**Файл:** `deploy/systemd/ai-trader.service`

```ini
[Unit]
Description=AI Trading Bot
After=network.target postgresql.service
Wants=postgresql.service

[Service]
Type=simple
User=aitrader
Group=aitrader
WorkingDirectory=/opt/ai_trader

# Environment
EnvironmentFile=/opt/ai_trader/.env

# Start command
ExecStart=/opt/ai_trader/venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

# Restart policy
Restart=always
RestartSec=10
StartLimitIntervalSec=0

# Health check (требует systemd >=248)
# Если /health возвращает не 200, restart
ExecStartPost=/bin/bash -c 'sleep 5 && curl -f http://localhost:8000/health || exit 1'

# Graceful shutdown
TimeoutStopSec=30
KillMode=mixed
KillSignal=SIGTERM

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=ai-trader

# Security
NoNewPrivileges=true
PrivateTmp=true

# Resource limits
LimitNOFILE=65536
LimitNPROC=4096

[Install]
WantedBy=multi-user.target
```

#### Установка:
```bash
# Скопировать service file
sudo cp deploy/systemd/ai-trader.service /etc/systemd/system/

# Reload systemd
sudo systemctl daemon-reload

# Включить auto-start
sudo systemctl enable ai-trader

# Запустить
sudo systemctl start ai-trader

# Проверить статус
sudo systemctl status ai-trader

# Логи
sudo journalctl -u ai-trader -f
```

---

#### 2.2.2 Docker restart policies
**Файл:** `docker-compose.yml` (обновить)

```yaml
version: '3.8'

services:
  app:
    build: .
    container_name: ai_trader_app
    restart: unless-stopped  # НОВОЕ: auto-restart
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/aitrader
      - BINANCE_API_KEY=${BINANCE_API_KEY}
      - BINANCE_API_SECRET=${BINANCE_API_SECRET}
    depends_on:
      db:
        condition: service_healthy
    healthcheck:  # НОВОЕ: health check
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:  # НОВОЕ: resource limits
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G

  db:
    image: postgres:15-alpine
    container_name: ai_trader_db
    restart: unless-stopped  # НОВОЕ
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: aitrader
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:  # НОВОЕ
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 10s
      timeout: 5s
      retries: 5

  prometheus:  # НОВОЕ: мониторинг
    image: prom/prometheus:latest
    container_name: ai_trader_prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.retention.time=30d'

volumes:
  postgres_data:
  prometheus_data:
```

#### Healthcheck endpoint enhancement:
```python
# routers/health.py (обновить)

@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """
    Enhanced health check.

    Проверяет:
    - Database connection
    - Broker connection
    - Disk space
    - Memory usage
    """
    checks = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {},
    }

    # DB check
    try:
        await db.execute(text("SELECT 1"))
        checks["checks"]["database"] = "ok"
    except Exception as e:
        checks["status"] = "unhealthy"
        checks["checks"]["database"] = f"error: {e}"

    # Broker check
    try:
        broker_status = await broker_gateway.ping()
        checks["checks"]["broker"] = "ok" if broker_status else "error"
    except Exception as e:
        checks["status"] = "unhealthy"
        checks["checks"]["broker"] = f"error: {e}"

    # Disk space check (>10% free)
    disk_usage = shutil.disk_usage("/")
    free_pct = (disk_usage.free / disk_usage.total) * 100
    if free_pct < 10:
        checks["status"] = "degraded"
        checks["checks"]["disk"] = f"low: {free_pct:.1f}% free"
    else:
        checks["checks"]["disk"] = f"ok: {free_pct:.1f}% free"

    # Memory check (>20% free)
    mem = psutil.virtual_memory()
    if mem.available / mem.total < 0.2:
        checks["status"] = "degraded"
        checks["checks"]["memory"] = f"low: {mem.percent}% used"
    else:
        checks["checks"]["memory"] = f"ok: {mem.percent}% used"

    status_code = 200 if checks["status"] == "healthy" else 503
    return JSONResponse(content=checks, status_code=status_code)
```

---

#### Тесты auto-restart:
```bash
# Тест 1: Kill процесс
sudo systemctl stop ai-trader
# Должен перезапуститься автоматически через 10 сек
sleep 15
sudo systemctl status ai-trader  # Должен быть active

# Тест 2: Simulate crash
docker exec ai_trader_app kill -9 1
# Docker должен перезапустить контейнер
sleep 10
docker ps  # Должен быть running

# Тест 3: Health check failure
# Временно сломать /health endpoint
# Docker должен перезапустить через 3 failed checks (90 сек)
```

---

## 🎯 ЭТАП 3: Продвинутые AI модели (4-6 недель)

### 3.1 CNN для паттернов
**Срок:** 1-2 недели
**Файл:** `src/models/signal/cnn_pattern.py`

#### Архитектура:
```python
class CNNPatternDetector:
    """
    CNN для детекции графических паттернов.

    Architecture:
        Input: [batch, height=50, width=50, channels=4]  # OHLC as image

        Conv2D1: 32 filters, 3x3, ReLU
        MaxPooling2D: 2x2
        Conv2D2: 64 filters, 3x3, ReLU
        MaxPooling2D: 2x2
        Conv2D3: 128 filters, 3x3, ReLU
        GlobalAveragePooling2D

        Dense1: 256 units, ReLU, Dropout(0.5)
        Dense2: 128 units, ReLU
        Output: 3 units, Softmax (BUY/SELL/HOLD)

    Patterns to detect:
        - Head and Shoulders
        - Double Top/Bottom
        - Triangle (ascending/descending/symmetric)
        - Wedge (rising/falling)
        - Flag, Pennant
        - Cup and Handle
    """

    def ohlc_to_image(
        self,
        df: pd.DataFrame,
        window: int = 50,
        img_size: tuple[int, int] = (50, 50),
    ) -> np.ndarray:
        """
        Конвертация OHLC в изображение.

        Метод:
        1. Нормализация цен в [0, 1]
        2. Создание candlestick визуализации
        3. Resize до img_size

        Returns:
            [height, width, 4]  # RGBA
        """
        ...
```

*(Детали опущены для краткости - см. отдельное ТЗ)*

---

### 3.2 Transformer
**Срок:** 2-3 недели
**Файл:** `src/models/signal/transformer_signal.py`

*(Архитектура будет описана в отдельном ТЗ)*

---

### 3.3 DRL Agent (PPO)
**Срок:** 2-3 недели
**Файл:** `src/models/drl/ppo_agent.py`

*(Архитектура будет описана в отдельном ТЗ)*

---

## 🎯 ЭТАП 4: Дополнительные источники данных (2-3 недели)

### 4.1 Twitter/X Integration
**Срок:** 1 неделя
**Файл:** `news/twitter_client.py`

*(Спецификация будет описана в отдельном ТЗ)*

---

### 4.2 Fear & Greed Index
**Срок:** 2 дня
**Файл:** `sources/fear_greed.py`

```python
class FearGreedIndexClient:
    """
    Crypto Fear & Greed Index from alternative.me

    API: https://api.alternative.me/fng/

    Values:
    - 0-24: Extreme Fear
    - 25-49: Fear
    - 50-74: Greed
    - 75-100: Extreme Greed
    """

    async def get_current(self) -> dict:
        """
        Текущее значение индекса.

        Returns:
            {
                'value': 45,
                'classification': 'Fear',
                'timestamp': '2025-01-27T12:00:00Z',
            }
        """
        ...

    async def get_historical(self, days: int = 30) -> pd.DataFrame:
        """История за N дней."""
        ...
```

---

### 4.3 Экономический календарь
**Срок:** 1 неделя
**Файл:** `sources/economic_calendar.py`

*(Спецификация будет описана в отдельном ТЗ)*

---

## 📊 Сводная таблица трудозатрат

| Этап | Компонент | Срок | Приоритет |
|------|-----------|------|-----------|
| **1** | Triple-Barrier Labeling | 3 дня | 🔴 Критический |
| **1** | LSTM модель | 5 дней | 🔴 Критический |
| **1** | Purged Walk-Forward CV | 3 дня | 🔴 Критический |
| **1** | Автопереобучение | 4 дней | 🔴 Критический |
| | **ИТОГО ЭТАП 1** | **2-3 недели** | |
| **2** | Telegram-бот | 5 дней | 🟡 Высокий |
| **2** | Auto-restart | 2 дня | 🟡 Высокий |
| | **ИТОГО ЭТАП 2** | **1 неделя** | |
| **3** | CNN для паттернов | 1-2 недели | 🟢 Средний |
| **3** | Transformer | 2-3 недели | 🟢 Средний |
| **3** | DRL Agent | 2-3 недели | 🟢 Средний |
| | **ИТОГО ЭТАП 3** | **4-6 недель** | |
| **4** | Twitter/X | 1 неделя | 🟢 Средний |
| **4** | Fear & Greed | 2 дня | 🟢 Низкий |
| **4** | Эконом. календарь | 1 неделя | 🟢 Средний |
| | **ИТОГО ЭТАП 4** | **2-3 недели** | |
| | **ОБЩИЙ СРОК** | **8-12 недель** | |

---

## 💡 Рекомендуемая стратегия

### Вариант 1: MVP (80% готовности за 4 недели)
Реализовать **ЭТАП 1 + ЭТАП 2**

**Результат:**
- ✅ LSTM модель с автопереобучением
- ✅ Triple-barrier labels
- ✅ Purged CV валидация
- ✅ Telegram уведомления
- ✅ Auto-restart
- **Готовность: ~80%**
- **Можно запускать production 24/7**

---

### Вариант 2: Полное ТЗ (95% готовности за 12 недель)
Реализовать **ВСЕ 4 ЭТАПА**

**Результат:**
- ✅ Все AI модели (LSTM, CNN, Transformer, DRL)
- ✅ Все источники данных
- ✅ 100% автономность
- **Готовность: ~95%**

---

## 📝 Следующие действия

**Выберите вариант:**

1. **Начать ЭТАП 1** (Triple-Barrier + LSTM + CV + переобучение)
2. **Начать ЭТАП 2** (Telegram + Auto-restart)
3. **Запустить текущую версию** для тестирования
4. **Детализировать ТЗ** для конкретного компонента

---

**Документ подготовлен:** Claude Code Agent
**Дата:** 2025-11-27
**Версия:** 1.0
