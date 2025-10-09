# AI Trader — Functional Overview and File Map

## Delivery Status
- **Stage 2 (Advanced Market Intelligence)** is fully delivered: the indicator library, news-aware analysis pipeline, background ingestion worker, strategy validation, and runtime dependencies are complete and regression-tested.【F:doc/stage2_status.md†L1-L17】
- **Stage 3 (Paper Trading & Resilience)** is also 100 % ready: enhanced backtesting exports, automated risk controls, auto-heal orchestration, observability endpoints, and operational runbooks are in place with dependencies declared.【F:doc/stage3_status.md†L1-L21】

## Core Capabilities
### Market Intelligence & Alternative Data
- `src/indicators.py` provides momentum, volatility, volume, candlestick, and resampling utilities used throughout the feature stack, keeping inputs immutable and handling short histories gracefully.【F:src/indicators.py†L1-L55】
- `src/analysis/analyze_market.py` combines the indicator suite, configurable scoring, and multi-timeframe logic into a single `analyze_market` entry point with news-aware adjustments controlled by `AnalysisConfig`.【F:src/analysis/analyze_market.py†L1-L132】
- `src/analysis/news_sentiment.py` loads cached headlines, filters them by symbol aliases, re-scores items with NLP hints, and returns aggregated sentiment labels for the analysis layer.【F:src/analysis/news_sentiment.py†L1-L120】
- `tasks/news_ingest.py` schedules asynchronous RSS refreshes, while `news/rss_client.py` normalizes multi-source feeds, deduplicates entries, and enriches impact metadata; both are wired into the API startup when background tasks are enabled.【F:tasks/news_ingest.py†L1-L71】【F:news/rss_client.py†L1-L120】【F:src/main.py†L490-L522】

### Strategy Orchestration
- `src/strategy.py` defines typed strategy/ensemble schemas with validation rules covering supported strategy kinds, weight positivity, frequency filters, and threshold resolution.【F:src/strategy.py†L1-L120】
- The default ensemble configuration in `configs/strategy.yaml` blends EMA crossovers, RSI reversion, and Bollinger breakout signals with a frequency governor, mirroring production-ready ensembles.【F:configs/strategy.yaml†L1-L43】
- Tests in `tests/test_strategy.py` assert signal generation correctness and catch invalid ensemble configs, ensuring schema validation remains enforced.【F:tests/test_strategy.py†L1-L44】

### Trading, Simulation, and Risk Controls
- `routers/trading.py` exposes paper-trading endpoints (e.g., `/paper/backtest`) that return win-rate, PnL, Sharpe metrics, and optional trade/equity exports for journaling.【F:routers/trading.py†L520-L656】
- `src/paper.py` implements a one-position paper trader with SL/TP, trailing stops, fee accounting, and detailed trade logs compatible with the API schemas.【F:src/paper.py†L1-L168】
- `services/trading_service.py` encapsulates execution with per-trade and daily risk limits, portfolio guardrails, executor failover hooks, and auto-heal registration for live trading modes.【F:services/trading_service.py†L1-L200】
- `services/auto_heal.py` persists executor snapshots and replays restore callbacks, providing recovery support leveraged by the trading service and tested in resilience suites.【F:services/auto_heal.py†L1-L64】

### Observability & Operations
- `monitoring/observability.py` registers Prometheus histograms, counters, and gauges while tracking SLO compliance through an `ObservabilityHub`.【F:monitoring/observability.py†L1-L95】
- `routers/metrics.py` surfaces `/metrics` for Prometheus scrapes and `/observability/slo` for JSON SLO snapshots, completing the monitoring contract.【F:routers/metrics.py†L1-L19】
- Operational runbooks describe standard procedures for news ingestion, auto-healing, Prometheus onboarding, and paper-trading validation, supporting day-two readiness.【F:doc/runbooks.md†L1-L53】
- Core dependencies, including `feedparser` and `prometheus-client`, are now part of `setup.py`, guaranteeing that ingestion and observability modules run in clean deployments.【F:setup.py†L1-L40】

### Quality Assurance
- Indicator regression tests verify EMA/SMA/RSI math, volume indicators, candlestick detection, and resampling helpers.【F:tests/test_indicators.py†L1-L44】
- Strategy tests cover ensemble validation and EMA cross signal integrity, while resilience tests validate auto-heal snapshotting, SLO trackers, and Prometheus export paths.【F:tests/test_strategy.py†L1-L44】【F:tests/test_resilience.py†L1-L34】

## API & Background Services
- FastAPI application wiring in `src/main.py` conditionally mounts OHLCV, trading, execution, autopilot, UI, and monitoring routers, ensuring modular deployment toggles.【F:src/main.py†L772-L793】
- The same startup lifecycle kicks off background OHLCV loaders, auto-trading loops, and the news ingestion worker when allowed, and cleans them up gracefully on shutdown.【F:src/main.py†L490-L558】

## File Structure Cheat Sheet
| Path | Purpose |
| --- | --- |
| `src/analysis/` | Market analysis engine with sentiment integration and helper utilities.【F:src/analysis/analyze_market.py†L1-L200】 |
| `src/indicators.py` | Central indicator toolbox for momentum, volume, candle, and resampling computations.【F:src/indicators.py†L1-L120】 |
| `src/paper.py` | Paper-trading simulator handling SL/TP, trailing stops, and journaling.【F:src/paper.py†L1-L168】 |
| `src/strategy.py` | Strategy ensemble definitions, validation, and signal helpers.【F:src/strategy.py†L1-L120】 |
| `services/trading_service.py` | Execution, risk management, and auto-heal orchestration for live/paper modes.【F:services/trading_service.py†L1-L200】 |
| `services/auto_heal.py` | Snapshot-based recovery orchestrator for trading executors.【F:services/auto_heal.py†L1-L64】 |
| `routers/trading.py` | API endpoints for signals, paper trading, and equity exports.【F:routers/trading.py†L520-L656】 |
| `routers/metrics.py` | Monitoring endpoints for Prometheus metrics and SLO status.【F:routers/metrics.py†L1-L19】 |
| `tasks/news_ingest.py` | Background job refreshing RSS news feeds for sentiment inputs.【F:tasks/news_ingest.py†L1-L71】 |
| `news/rss_client.py` | RSS client that fetches, deduplicates, and scores headlines.【F:news/rss_client.py†L1-L120】 |
| `configs/strategy.yaml` | Default multi-strategy ensemble configuration.【F:configs/strategy.yaml†L1-L43】 |
| `doc/stage2_status.md`, `doc/stage3_status.md` | Readiness reports summarizing completed Stage 2/3 deliverables.【F:doc/stage2_status.md†L1-L17】【F:doc/stage3_status.md†L1-L21】 |
| `doc/runbooks.md` | Operational procedures for ingestion, auto-heal, monitoring, and paper workflows.【F:doc/runbooks.md†L1-L53】 |
| `tests/` | Regression suites covering indicators, strategy validation, and resilience features.【F:tests/test_indicators.py†L1-L44】【F:tests/test_strategy.py†L1-L44】【F:tests/test_resilience.py†L1-L34】 |
| `setup.py` | Packaging metadata and runtime dependencies ensuring ingestion and monitoring modules install cleanly.【F:setup.py†L1-L40】 |

## Readiness Summary
Stages 2 and 3 have no open functional or operational gaps; indicator coverage, news automation, ensemble validation, paper trading, risk controls, auto-healing, observability, and operational playbooks are complete and dependency-supported, marking the project as 100 % ready for these phases.【F:doc/stage2_status.md†L6-L17】【F:doc/stage3_status.md†L5-L21】
