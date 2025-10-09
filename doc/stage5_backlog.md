# Stage 5 Backlog — Autonomous Fund Operations & Scale-Out

Stage 5 focuses on evolving the platform from a production-ready single-venue trader into an autonomous, multi-asset fund with institutional controls. The backlog below captures candidate epics and representative tasks that build on the current Stage 4 capabilities.

## 1. Multi-Venue & Instrument Expansion
- Add gateway adapters for at least two additional venues (e.g., Coinbase, Bybit) reusing the existing `BrokerGateway` protocol while abstracting venue-specific order flags, fees, and rate limits.【F:services/broker_gateway.py†L1-L200】
- Introduce a venue router in the live coordinator so strategies can target symbols per venue and aggregate fills, including FX conversion for non-USD quote assets.【F:services/live_trading.py†L1-L120】【F:services/live_trading.py†L200-L320】
- Extend execution configs to describe derivative instruments (perpetuals/options) including contract multipliers, margin mode, and funding schedules.【F:services/trading_service.py†L1-L120】【F:services/trading_service.py†L340-L520】
- Implement consolidated market data ingestion with latency monitoring for multi-venue order book snapshots, leveraging the existing websocket/watchdog infrastructure.【F:src/main.py†L493-L710】

## 2. Portfolio Allocation & Treasury Management
- Build a capital allocator that distributes risk budgets across strategies using portfolio risk metrics (VaR/ES) in addition to per-trade limits already enforced in the trading service.【F:services/trading_service.py†L120-L260】【F:risk/risk_manager.py†L45-L141】
- Track multi-currency balances and automate treasury rebalancing, including stablecoin conversions and withdrawals when drawdown rules trigger.【F:services/broker_gateway.py†L90-L182】【F:services/live_trading.py†L250-L340】
- Produce daily/weekly NAV calculations with cash flow reconciliation by enriching the reconciliation service and journal schema.【F:services/reconcile.py†L1-L200】【F:services/reconcile.py†L358-L472】

## 3. Advanced AI Research & Self-Tuning Strategies
- Integrate automated hyper-parameter sweeps and walk-forward validation pipelines that export results back into the model router configuration for deployment.【F:scripts/meta_label_cli.py†L1-L200】【F:services/model_router.py†L24-L140】
- Add reinforcement-learning or meta-optimization agents that consume market features alongside regime outputs, with sandbox evaluation loops before promotion.【F:src/models/base.py†L1-L70】【F:src/models/drl/consultant.py†L1-L33】
- Implement online feature importance and explainability reports for operator review, exposing them through the `/ai` diagnostics endpoints.【F:routers/autopilot.py†L1-L200】

## 4. Execution Quality & Market Microstructure
- Develop smart order routing policies with passive/active order selection, iceberg slicing, and venue selection based on live liquidity metrics.【F:services/broker_gateway.py†L1-L200】【F:services/trading_service.py†L520-L720】
- Introduce slippage, impact, and latency analytics using Prometheus histograms and store them for TCA dashboards and alerts.【F:monitoring/observability.py†L1-L120】【F:templates/monitor/index.html†L116-L200】
- Simulate stress scenarios (volatility spikes, connectivity loss) through the auto-heal framework to validate execution fallback behaviour before production rollout.【F:services/auto_heal.py†L1-L120】【F:tests/test_resilience.py†L1-L120】

## 5. Governance, Compliance & Investor Reporting
- Automate regulatory reporting packs (trade blotter, best-execution evidence, AML flags) by extending the reconciliation journal with immutable exports.【F:services/reconcile.py†L200-L360】
- Integrate secret rotation and hardware security module (HSM) workflows into the existing access controller for production key management.【F:services/trading_service.py†L20-L80】【F:services/security.py†L1-L200】
- Produce investor-facing performance reports (PnL attribution, risk metrics, exposure breakdown) and publish them through scheduled background jobs.【F:src/main.py†L713-L730】【F:templates/monitor/index.html†L116-L200】

## 6. Operator & Client Experience Enhancements
- Expand the HTMX dashboard into a full operator console with role-based access, change approvals, and embedded runbook links.【F:templates/monitor/index.html†L45-L155】【F:doc/runbooks.md†L60-L90】
- Deliver event-driven alerts (mobile, email, chat) for breaches detected in the trading coordinator and drift monitors, including acknowledgement workflows.【F:services/live_trading.py†L320-L420】【F:services/model_router.py†L156-L200】
- Provide a client portal exposing curated analytics (NAV, risk, compliance status) with audit logging aligned to the existing `/live` API surface.【F:routers/live_trading.py†L35-L179】【F:routers/trading.py†L520-L656】

## 7. Dashboard Functional System Management Blueprint
- **Operational Overview Layer.** Present a consolidated status banner with broker connectivity, strategy heartbeat, and automation toggles so operators can confirm live readiness at a glance before drilling into modules. Surface the existing heartbeat metrics from the trading coordinator and execution services with HTMX refresh hints for degraded states.【F:templates/monitor/index.html†L45-L155】【F:services/live_trading.py†L200-L340】
- **Strategy Lifecycle Console.** Provide controls to enable/disable strategies, adjust capital allocations, and schedule maintenance windows, with guardrails backed by the trading service and risk manager APIs. Include audit trails for each change so governance checks can be reconstructed later.【F:services/trading_service.py†L120-L340】【F:risk/risk_manager.py†L45-L141】
- **Risk Guardrail Panel.** Centralize circuit-breaker thresholds, drawdown rules, and exposure limits with real-time breach indicators. Integrate the risk manager’s telemetry and allow staged overrides that require multi-step confirmations to satisfy compliance workflows.【F:risk/risk_manager.py†L90-L200】【F:templates/monitor/index.html†L116-L200】
- **Treasury & Settlement Board.** Show multi-currency balances, pending transfers, and funding schedule reminders; expose quick actions for on-demand rebalancing that call into treasury automation endpoints once approvals are complete.【F:services/broker_gateway.py†L90-L182】【F:services/reconcile.py†L200-L360】
- **Incident Response Runbooks.** Embed contextual runbook snippets and escalation paths next to each widget, with deep links into the existing operator documentation so Stage 5 teams can resolve issues without leaving the console.【F:doc/runbooks.md†L1-L140】【F:templates/monitor/index.html†L155-L220】
- **Regulatory & Investor Reporting Hub.** Offer scheduled report generation, distribution status, and retention controls for trade blotters, AML alerts, and investor decks. Hook into reconciliation exports and the reporting jobs defined in Stage 5 governance epics to keep compliance artefacts synchronized.【F:services/reconcile.py†L200-L360】【F:src/main.py†L713-L730】
- **Customization & Access Control.** Allow role-based dashboards, widget layouts, and saved views so that compliance officers, portfolio managers, and client relations teams can tailor their workspace while respecting the security policies managed by the access controller.【F:services/security.py†L1-L200】【F:templates/monitor/index.html†L45-L120】

These initiatives collectively move the product toward Stage 5 readiness by scaling execution, deepening risk oversight, and operationalizing fund-level governance on top of the robust Stage 4 foundation.【F:doc/stage4_status.md†L1-L120】
