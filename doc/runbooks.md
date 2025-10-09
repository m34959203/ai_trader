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

Keep this document alongside deployment checklists to ensure Stage 2 and Stage 3 runbooks remain up to date.
