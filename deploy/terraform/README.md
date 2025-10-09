# AI Trader Kubernetes Deployment (Terraform)

This module provisions a highly-available deployment of the FastAPI service on an
existing Kubernetes cluster.  It creates the following resources:

* Namespace scoped to the workload (`var.namespace`).
* ConfigMap providing runtime environment variables and Prometheus scrape hints.
* Deployment with configurable replica count, structured logging environment,
  and health probes (`/healthz`, `/_readyz`, `/health/deep`).
* Service to expose the HTTP port (`var.service_port`).
* HorizontalPodAutoscaler configured for CPU utilization targets.
* PodDisruptionBudget limiting voluntary evictions to one replica at a time.

## Usage

```
module "ai_trader" {
  source = "./deploy/terraform"

  image_repository       = "registry.example.com/ai-trader"
  image_tag              = "v1.4.0"
  namespace              = "ai-trader-prod"
  replicas               = 3
  slack_webhook_url      = var.slack_webhook_url
  pagerduty_routing_key  = var.pagerduty_routing_key
  kube_context           = "prod-cluster"
}
```

After applying the module, the service will expose `/metrics` for Prometheus and
return structured JSON logs to stdout.  See `monitoring/logging_stack.md` for
forwarding guidance and `monitoring/alertmanager.yaml` for Slack/PagerDuty
routing examples.
