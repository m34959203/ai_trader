# Live Trading Production Checklist — Credentials & External Services

This document enumerates the secrets and third-party services required to run the
"AI-Trader" platform in a production (live trading) environment.  Each item
should be provisioned through a secure secret manager (Vault, AWS Secrets
Manager, GCP Secret Manager, etc.) and injected via environment variables or
orchestrator-level secret mounts during deployment.  **Do not commit secrets to
source control or store them in plaintext configuration files.**

## 1. Brokerage / Exchange Access

| Service | Purpose | Environment variables (defaults) | Notes |
| --- | --- | --- | --- |
| Binance Spot/Testnet | Live order routing | `BINANCE_API_KEY`, `BINANCE_API_SECRET` (prod) <br> `BINANCE_TESTNET_API_KEY`, `BINANCE_TESTNET_API_SECRET` (testnet) | Required for the `BinanceBrokerGateway`.  Keys must have trading permissions enabled.  When running on testnet, populate the `*_TESTNET_*` variables; production deployments should set the non-testnet variables. |
| Binance User Data Stream | Account & position streaming | Same as above | The WebSocket listeners reuse the REST keys to obtain listen keys; ensure the REST user has permission for user data streams. |

If a different exchange is desired, create a matching gateway implementation and
register the equivalent API credentials using the same secret-management
process.

## 2. Market Data Providers

| Provider | Usage | Environment variables | Notes |
| --- | --- | --- | --- |
| Binance Market Data | Live quotes via REST/WebSocket | `BINANCE_API_KEY` (optional) | Public market data can operate without keys, but authenticated endpoints benefit from higher rate limits. |
| Alternate Feeds (CCXT, Alpha Vantage, etc.) | Historical data/backfill | Provider specific | Configure in the deployment environment as required by the chosen source modules. |

## 3. NLP / ML Model Access

| Model / Service | Usage | Environment variables | Notes |
| --- | --- | --- | --- |
| Hugging Face (FinBERT) | Financial sentiment scoring | `HF_HOME` (cache), `HF_TOKEN` (if using private models) | The runtime ships with a rule-based fallback.  Provide a Hugging Face token if your deployment needs to pull private or rate-limited models. |
| Optional Cloud NLP (Azure, GCP, OpenAI) | Alternate sentiment/NER pipelines | Provider specific | Populate only if the corresponding adapters are enabled. |

## 4. Observability & Ops Integrations

| Service | Usage | Environment variables | Notes |
| --- | --- | --- | --- |
| Prometheus / Grafana | Metrics and dashboards | `PROMETHEUS_PUSHGATEWAY_URL` (if push-mode), `PROMETHEUS_BASIC_AUTH` (if required) | The API already exposes `/metrics`; configure scraping credentials on the infrastructure side. |
| Alerting (PagerDuty/Slack) | Incident notifications | `SLACK_WEBHOOK_URL`, `PAGERDUTY_ROUTING_KEY` | Populate from your secret store and mount `monitoring/alertmanager.yaml` into Alertmanager so Slack receives all alerts and PagerDuty handles `severity=critical`. |

Alert payloads include the structured logging `trace_id`, enabling on-call engineers
to cross-reference alerts with JSON logs shipped via Fluent Bit/Vector (see
`monitoring/logging_stack.md`).

## 5. Operational Safety Nets

- **Dead Man Switch / Heartbeat:** Configure the dead-man file path (`risk.deadman`) and monitoring to halt trading if heartbeats stop.
- **Daily Limits Store:** Ensure the state directory (`state/`) persists across restarts or back it by a database/redis service.
- **Secret Rotation:** Schedule rotation of exchange/API keys according to broker policy and update the secret manager entries accordingly.

## 6. Deployment Checklist

1. Provision environment variables listed above via your orchestration layer.
2. Mount `configs/exec.yaml` (or provide overrides) to select the desired gateway
   (`execution.gateway`) and confirm `api_key_env` / `api_secret_env` match the
   stored secret names.
3. Validate connectivity by hitting `/live/status` — it will return the active
   gateway and risk limits when keys are loaded successfully.
4. Execute a dry-run `/live/trade` call against the testnet to confirm end-to-end
   integration before switching `execution.binance.testnet` to `false`.
5. Add monitoring/alerting for `/metrics` and `/observability/slo` endpoints.

## 7. Integration Test Procedure

To validate the live trading stack without sending production orders, run the
integration suite against the Binance Testnet sandbox:

1. Export the sandbox credentials (or rely on the defaults listed in
   `configs/exec.yaml`):

   ```bash
   export BINANCE_TESTNET_API_KEY=... \
          BINANCE_TESTNET_API_SECRET=...
   ```

2. Execute the live broker gateway tests, which rely on mocked HTTP transports
   and sandbox-compatible payloads:

   ```bash
   pytest tests/test_broker_gateway_live.py -q
   ```

3. Optional: run the FastAPI router smoke tests to verify the REST endpoints
   used by operators and monitoring dashboards:

   ```bash
   pytest tests/test_live_router.py -q
   ```

These checks ensure order polling, cancellation flows, and account synchronisation
operate correctly against the sandbox APIs before enabling live trading.

Following this checklist ensures the application runs with appropriately scoped
credentials and that operational dependencies are satisfied prior to enabling
live order routing.
