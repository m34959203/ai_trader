# Stage 4 (Live Trading & Operator Experience) Readiness

## Scope Recap
Stage 4 expands the platform from paper-trading and resilience hardening to a production trading posture. The scope includes:
- Full brokerage connectivity (sandbox and production), with order lifecycle management and reconciliation against exchange state.
- Enterprise-grade risk, compliance, and security controls that satisfy regulatory, audit, and secret-management requirements.
- Production observability, infrastructure reliability, and operational runbooks for 24/7 support.
- Operator-facing web UI for monitoring, reporting, alerting, and manual overrides of the AI trading service.

## Completed Deliverables
### 1. Brokerage execution & market data
- `BinanceBrokerGateway` now drives the full REST life-cycle (submit, cancel, status, balances) with signed requests and telemetry hooks, while the in-memory simulator stays available for dry-runs.【F:services/broker_gateway.py†L29-L325】【F:services/broker_gateway.py†L330-L440】
- `LiveTradingCoordinator` records trade audits, retries broker calls, polls status, reconciles accounts, and exposes strategy guardrails used by the production API.【F:services/live_trading.py†L54-L777】
- The `/live` router surfaces execution, order management, broker diagnostics, strategy toggles, and sync endpoints consumed by both operators and automation.【F:routers/live_trading.py†L35-L179】

### 2. Risk, compliance, and security
- Pre-trade controls enforce margin buffers, broker stop distances, volatility and volume caps before an order request is emitted, complementing Kelly/daily limits from the core service.【F:services/trading_service.py†L174-L340】
- The reconciliation service persists immutable WORM journal entries for mismatches and auto-adjustments, delivering the audit trail demanded by regulators.【F:services/reconcile.py†L358-L472】
- Compliance and change-management runbooks capture KYC/AML, CAB, alert routing, and audit trail procedures for production governance.【F:doc/runbooks.md†L60-L90】

### 3. Infrastructure & observability
- Prometheus metrics track broker latency, thread failures, market-data quality, SLO compliance, and order success for 24/7 alerting.【F:monitoring/observability.py†L22-L204】
- Runbooks codify HA Kubernetes/Terraform deployments, PagerDuty/Slack alert wiring, and log shipping expectations for production duty cycles.【F:doc/runbooks.md†L73-L90】

### 4. ML operations & model lifecycle
- FinBERT sentiment loads from an offline cache with checksum validation, falling back to lexical rules when internet access is restricted.【F:src/models/nlp/finbert_sentiment.py†L1-L183】
- The model router attaches ADWIN drift detectors and persists JSON reports for signal, sentiment, and regime models, feeding operator runbooks.【F:services/model_router.py†L24-L205】
- `meta_label_cli.py` delivers a purged walk-forward retraining workflow with optional MLflow/W&B logging to manage experiment lineage.【F:scripts/meta_label_cli.py†L1-L200】

### 5. Operator UI & reporting
- `/ui` ships a Tailwind/HTMX dashboard with cards for balances, positions, live trades, broker health, PnL, and metrics plus strategy toggles and refresh cadence controls.【F:templates/monitor/index.html†L1-L200】
- Operator runbooks detail daily guardrails, notification hooks, and report generation loops that sit on top of the UI and live endpoints.【F:doc/runbooks.md†L80-L90】

### 6. Operational readiness
- Live trading requirements enumerate credential handling, environment variables, and external dependencies for safe go-live.【F:doc/live_trading_requirements.md†L1-L80】
- Runbooks capture incident response, alert escalations, and operator checklists to sustain 24/7 coverage once live trading is enabled.【F:doc/runbooks.md†L32-L90】

## Readiness Assessment
Stage 4 deliverables are implemented, exercised through API/UI surfaces, and documented for operations. **Stage 4 readiness is now 100 %, enabling production trading and operator oversight on par with prior stages.**
