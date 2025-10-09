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
- Broker latency snapshot: `GET /observability/broker-latency` summarises recent Binance round-trips and SLO compliance.
- Thread health feed: `GET /observability/thread-health` exposes background task failures (`market_stream_consumer`, `ws_watchdog`, auto-heal, etc.).
- Market data quality: `GET /observability/market-data` returns freshness scores and missing bar counters per symbol/interval.
- Default tracked SLOs:
  - `order_success` ≥ 98%
  - `latency_under_2s` ≥ 95%
  - `executor_availability` ≥ 99%
  - `broker_latency_under_1s` ≥ 97%
  - `market_data_quality_score` ≥ 95%
  - `market_data_freshness` ≥ 98%
  - `stream_thread_recovery` ≥ 99%
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

## 5. Compliance: KYC / AML Governance
- **Counterparty due diligence:** Maintain an up-to-date registry of all brokers, exchanges, and liquidity venues. Collect legal entity identifiers, licensing status, and risk ratings. Re-certify annually and whenever onboarding a new venue.
- **Customer verification:** When activating client-facing features (signal distribution or managed accounts), enforce KYC tiering: identity verification, proof-of-funds, and sanctions screening via an approved provider (e.g., Refinitiv, ComplyAdvantage). Log verification artefacts in the compliance vault.
- **AML surveillance:** Stream trade and transfer data into the AML rules engine. Configure red-flag scenarios (structuring, rapid in/out, sanctioned jurisdiction traffic) and auto-escalate alerts to compliance within 24 hours. Retain alert dispositioning records for seven years.
- **Regulator reporting:** Produce monthly SAR/STR summaries and, where applicable, MiFID II/EMIR transaction reports. Schedule exports via `tasks/compliance/reporting.py` (cron) and archive encrypted copies in WORM storage.

## 6. Change-Management Workflow
- **Request intake:** All production changes (code, infrastructure, configuration, secret rotation) require a change request ticket with risk impact, rollback plan, and stakeholder approvals.
- **CAB review:** Convene a weekly Change Advisory Board (engineering lead, security officer, compliance) to review queued changes. Emergency fixes still require a retroactive CAB sign-off.
- **Implementation controls:** Use feature flags and blue/green deployments where possible. Pre-prod validation must include integration tests (`pytest tests/test_broker_gateway_live.py -q`) and reconciliation dry-runs.
- **Post-change verification:** Within 24 hours of deployment, confirm health dashboards, trade throughput, and reconciliation status. Document verification evidence in the change ticket and update the release log.
- **Audit trail:** Persist change metadata (ticket ID, approvers, git SHA, deployment timestamp) in the WORM trade journal alongside operational logs for full traceability.

## 7. HA Kubernetes Deployment Checklist
- **Terraform module:** `deploy/terraform` provisions the namespace, ConfigMap, Deployment, Service, HPA, and PodDisruptionBudget. Set `image_repository`, `image_tag`, and point `slack_webhook_url`/`pagerduty_routing_key` to secret-backed variables.
- **Helm chart:** `deploy/helm` mirrors the Terraform layout for GitOps pipelines. Override `values.yaml` to tune replicas, autoscaling, and resource limits. Health probes hit `/healthz`, `/_readyz`, and `/health/deep`.
- **Prometheus/Grafana:** Ensure the cluster scrape config watches the new namespace. The pod annotations expose `/metrics`; dashboards should track `ai_trader_broker_latency_seconds`, `ai_trader_market_data_quality`, and thread failure counters.
- **Alert routing:** Apply `monitoring/alertmanager.yaml` (or merge it into an existing Alertmanager stack) so PagerDuty handles `severity=critical` while Slack receives all notifications. Include `trace_id` in templates for log correlation.
- **Log shipping:** Deploy Fluent Bit/Vector per `monitoring/logging_stack.md` so JSON logs with `trace_id` flow into the central store (Loki/ELK). Validate that alerts carry matching IDs before enabling paging.

## 8. Live Trading Operator Control & Notifications
- **UI dashboards:** `/ui` now exposes dedicated widgets for real-time PnL, broker connectivity, live trade journal, and risk limits. The controls pull data from `/live/pnl`, `/live/broker`, `/live/trades`, and `/live/limits` respectively. Operators should keep the dashboard pinned during trading sessions and rely on the global refresh selector to throttle polling when bandwidth is constrained.
- **Strategy gating:** Each strategy row exposes enable/disable toggles and per-strategy caps. Updates flow through `PATCH /live/strategies/{name}` and are reflected instantly in the coordinator—attempted orders from disabled strategies will be blocked with `strategy_disabled`. Record every change (who/why) in the operations log, including the previous and new risk caps.
- **Daily guardrails:** The limits panel surfaces equity-at-start, current equity, realized PnL, and drawdown. When daily loss approaches configured thresholds (`REPORTS_INTERVAL_MINUTES`, `per_trade_cap`, `daily_max_loss_pct`), the on-call operator must confirm whether automatic halts fired and, if necessary, set the “enabled” toggle to false for affected strategies.
- **Report generation:** Scheduled CSV/PDF exports run via `tasks.reports.background_loop` (configurable with `ENABLE_REPORTS_BG` / `REPORTS_INTERVAL_MINUTES`). Manual regeneration is available from the UI button or `POST /reports/generate`. The latest artefacts can be downloaded at `/reports/download/{csv|pdf}` and archived alongside compliance evidence.
- **Notification hooks:** Wire Slack/PagerDuty alerts to trigger on:
  - Broker disconnected for ≥30s (`broker.connected == false` from `/live/broker`).
  - Daily drawdown ≥ configured stop or realized PnL below tolerance.
  - Strategy auto-blocks (watch for `strategy_disabled` and `strategy_daily_limit` errors in `/live/trades`).
  - Report task failures (logs tagged `ai_trader.reports`). Alerts should include direct links to the dashboard and the relevant report artefacts.
- **On-call checklist:** Upon alert, confirm broker health, review latest trades, adjust strategy toggles if needed, regenerate reports, and log the incident in the runbook tracker. Always validate that the UI refreshes successfully after mitigations (use the “Обновить” button above the trades table).

Keep this document alongside deployment checklists to ensure Stage 2 and Stage 3 runbooks remain up to date.
