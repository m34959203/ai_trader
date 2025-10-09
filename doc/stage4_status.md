# Stage 4 (Live Trading & Operator Experience) Readiness

## Scope Recap
Stage 4 expands the platform from paper-trading and resilience hardening to a production trading posture. The scope includes:
- Full brokerage connectivity (sandbox and production), with order lifecycle management and reconciliation against exchange state.
- Enterprise-grade risk, compliance, and security controls that satisfy regulatory, audit, and secret-management requirements.
- Production observability, infrastructure reliability, and operational runbooks for 24/7 support.
- Operator-facing web UI for monitoring, reporting, alerting, and manual overrides of the AI trading service.

## Current Progress
- **Technical groundwork in place.** The codebase exposes broker-gateway abstractions, live trading coordinator surfaces, and diagnostic endpoints that Stage 4 can build upon.【F:services/broker_gateway.py†L1-L210】【F:services/live_trading.py†L1-L140】【F:routers/live_trading.py†L1-L60】
- **Credential guidance documented.** Deployment requirements for secrets, external services, and monitoring dependencies are catalogued to streamline provisioning.【F:doc/live_trading_requirements.md†L1-L80】
- **Roadmap defined.** A detailed live-trading roadmap enumerates the technical and organisational deliverables needed for Stage 4 completion.【F:doc/live_trading_roadmap.md†L1-L132】

## Outstanding Deliverables
The following work streams remain open and constitute the Stage 4 backlog:
1. **Brokerage execution & market data**
   - Finalise exchange selection, secure sandbox/production API keys, and configure rate-limit SLAs.
   - Implement and certify live order submission, cancellation, status polling, and position reconciliation against the chosen broker.
   - Connect authenticated real-time market-data feeds with redundancy and time synchronisation guarantees.
2. **Risk, compliance, and security**
   - Enforce broker-level SL/TP, volumetric limits, margin checks, and volatility-aware exposure controls.
   - Establish immutable trade journaling, WORM storage policies, and change-management workflows for audits.
   - Integrate KYC/AML, licensing reviews, and secure secret-management with rotation policies.
3. **Infrastructure & observability**
   - Deploy high-availability runtime (multi-zone Kubernetes/VMs) with health checks, auto-restarts, and backup failovers.
   - Extend Prometheus/Grafana with order-latency, broker failure, and data-quality metrics; integrate PagerDuty/Slack alerting.
   - Adopt structured logging (JSON + trace IDs) and centralised retention (ELK/OpenSearch) with RBAC.
4. **ML operations & model lifecycle**
   - Operationalise FinBERT downloads/caching, drift detection (ADWIN/alibi), and automated retraining with purged walk-forward CV.
   - Stand up experiment tracking (MLflow/W&B), meta-labelling pipelines, and canary deployments for strategy updates.
5. **Operator UI & reporting**
   - Deliver a secure web UI that surfaces live PnL, risk metrics, broker connectivity, trade ledgers, and health dashboards.
   - Implement controls for toggling strategies, adjusting limits, approving manual interventions, and acknowledging alerts.
   - Provide scheduled and on-demand reports (daily/weekly PnL, drawdown, compliance summaries) with export capabilities.
6. **Operational readiness**
   - Author incident playbooks, run fire-drill rehearsals, and define on-call rotations with clear escalation paths.
   - Complete end-to-end sandbox → pilot → production certification and commission external security/compliance audits.

## Readiness Assessment
Stage 4 has foundational code and documentation assets but lacks the production integrations, UI, risk/compliance hardening, and operational maturity required for go-live. **Stage 4 readiness is currently estimated at ~15%, with the outstanding tasks above blocking full production trading.**
