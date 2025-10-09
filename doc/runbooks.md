# AI Trader Operational Runbooks

## 1. News Ingestion Worker
- Service: `tasks.news_ingest.background_loop` (launched from FastAPI when `ENABLE_NEWS_INGEST=1`).
- Manual refresh:
  ```bash
  poetry run python -m tasks.news_ingest
  ```
- Runtime knobs (env):
  - `NEWS_REFRESH_INTERVAL_SEC`: interval between RSS pulls (default 900s).
  - `NEWS_REFRESH_LIMIT`: maximum items per feed (default 50).
  - `NEWS_REFRESH_INITIAL_DELAY_SEC`: warm-up delay before first pull.
- Troubleshooting:
  1. Check logs tagged `ai_trader.news_ingest` for HTTP errors.
  2. Re-run manual refresh; verify `data/news.json` timestamp updates.
  3. If the worker is disabled, toggle `ENABLE_BG_TASKS`/`ENABLE_NEWS_INGEST` and restart API service.

## 2. Executor Auto-Healing
- Orchestrator: `services.auto_heal.AutoHealingOrchestrator` persists snapshots in `state/auto_heal/`.
- Recovery flow:
  1. On executor failure, a snapshot `trading_executor.json` is written containing mode, config, and error context.
  2. The orchestrator triggers `TradingService._restore_executor_from_snapshot`, swapping the primary executor after the snapshot is persisted.
  3. Snapshots replay automatically on service start; to force replay manually run:
     ```python
     await AutoHealingOrchestrator().replay("trading_executor")
     ```
- Manual intervention checklist:
  - Inspect snapshot contents to confirm expected mode/testnet flags.
  - Delete stale snapshot after recovery to avoid repetitive replays.
  - Use `TradingService.configure()` to hot-swap executors if automated restore fails.

## 3. Prometheus / SLO Monitoring
- Metrics endpoint: `GET /metrics` (Prometheus exposition format).
- SLO API: `GET /observability/slo` returns JSON with `target`, `actual`, `met` for tracked SLOs.
- Default tracked SLOs:
  - `order_success` ≥ 98%
  - `latency_under_2s` ≥ 95%
  - `executor_availability` ≥ 99%
- Grafana onboarding:
  1. Point Prometheus scrape job at `https://<host>/metrics`.
  2. Import dashboard using the provided metrics names (`ai_trader_order_latency_seconds`, `ai_trader_orders_total`, etc.).
  3. Set alerts on `ai_trader_slo_compliance_ratio{}` below targets.

## 4. Paper-Trading One-Click Cycle
- End-to-end validation:
  1. Run `/strategy/signals` to fetch ensemble signal.
  2. Execute `/trading/open` (market/limit) with the recommended signal.
  3. Confirm journal entry via `/paper/backtest` export (check equity curve and PnL with fees).
- Safety toggles:
  - Daily limits configured in `TradingConfig.risk`.
  - Use `/trading/close_all` for emergency position flattening.

Keep this document alongside deployment checklists to ensure Stage 2 and Stage 3 runbooks remain up to date.
