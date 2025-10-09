terraform {
  required_version = ">= 1.3.0"
  required_providers {
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.23.0"
    }
  }
}

provider "kubernetes" {
  config_path    = var.kubeconfig_path
  config_context = var.kube_context
}

locals {
  app_labels = {
    app.kubernetes.io/name       = "ai-trader"
    app.kubernetes.io/component  = "api"
    app.kubernetes.io/managed-by = "terraform"
  }
}

resource "kubernetes_namespace" "ai_trader" {
  metadata {
    name = var.namespace
    labels = {
      "app.kubernetes.io/part-of" = "ai-trader"
    }
  }
}

resource "kubernetes_config_map" "ai_trader_env" {
  metadata {
    name      = "ai-trader-env"
    namespace = kubernetes_namespace.ai_trader.metadata[0].name
    labels    = local.app_labels
  }

  data = {
    APP_ENV           = var.app_env
    LOG_LEVEL         = var.log_level
    FEATURE_LIVE      = tostring(var.enable_live_trading)
    PROMETHEUS_SCRAPE = "true"
  }
}

resource "kubernetes_deployment" "ai_trader" {
  metadata {
    name      = "ai-trader-api"
    namespace = kubernetes_namespace.ai_trader.metadata[0].name
    labels    = local.app_labels
  }

  spec {
    replicas = var.replicas

    selector {
      match_labels = local.app_labels
    }

    template {
      metadata {
        labels = merge(local.app_labels, {
          "app.kubernetes.io/version" = var.image_tag
        })
        annotations = {
          "prometheus.io/scrape" = "true"
          "prometheus.io/port"   = tostring(var.service_port)
          "prometheus.io/path"   = "/metrics"
        }
      }

      spec {
        service_account_name = var.service_account

        container {
          name  = "api"
          image = "${var.image_repository}:${var.image_tag}"

          image_pull_policy = var.image_pull_policy

          env_from {
            config_map_ref {
              name = kubernetes_config_map.ai_trader_env.metadata[0].name
            }
          }

          env {
            name  = "SLACK_WEBHOOK_URL"
            value = var.slack_webhook_url
          }

          env {
            name  = "PAGERDUTY_ROUTING_KEY"
            value = var.pagerduty_routing_key
          }

          resources {
            limits = {
              cpu    = var.cpu_limit
              memory = var.memory_limit
            }
            requests = {
              cpu    = var.cpu_request
              memory = var.memory_request
            }
          }

          port {
            name           = "http"
            container_port = var.service_port
            protocol       = "TCP"
          }

          liveness_probe {
            http_get {
              path = "/healthz"
              port = var.service_port
            }
            initial_delay_seconds = 10
            period_seconds        = 10
            failure_threshold     = 3
          }

          readiness_probe {
            http_get {
              path = "/_readyz"
              port = var.service_port
            }
            initial_delay_seconds = 5
            period_seconds        = 5
            failure_threshold     = 3
          }

          startup_probe {
            http_get {
              path = "/health/deep"
              port = var.service_port
            }
            failure_threshold = 10
            period_seconds    = 6
          }

          volume_mount {
            name       = "config"
            mount_path = "/etc/ai-trader"
          }
        }

        volume {
          name = "config"

          config_map {
            name = kubernetes_config_map.ai_trader_env.metadata[0].name
          }
        }

        termination_grace_period_seconds = 30
      }
    }
  }
}

resource "kubernetes_service" "ai_trader" {
  metadata {
    name      = "ai-trader-api"
    namespace = kubernetes_namespace.ai_trader.metadata[0].name
    labels    = local.app_labels
  }

  spec {
    selector = local.app_labels
    port {
      name        = "http"
      port        = var.service_port
      target_port = var.service_port
    }
    type = var.service_type
  }
}

resource "kubernetes_horizontal_pod_autoscaler_v2" "ai_trader" {
  metadata {
    name      = "ai-trader-api"
    namespace = kubernetes_namespace.ai_trader.metadata[0].name
  }

  spec {
    max_replicas = var.hpa_max_replicas
    min_replicas = var.hpa_min_replicas

    scale_target_ref {
      kind = "Deployment"
      name = kubernetes_deployment.ai_trader.metadata[0].name
      api_version = "apps/v1"
    }

    metric {
      type = "Resource"

      resource {
        name = "cpu"
        target {
          type               = "Utilization"
          average_utilization = var.hpa_cpu_utilization
        }
      }
    }
  }
}

resource "kubernetes_pod_disruption_budget" "ai_trader" {
  metadata {
    name      = "ai-trader-api"
    namespace = kubernetes_namespace.ai_trader.metadata[0].name
  }

  spec {
    max_unavailable = "1"
    selector {
      match_labels = local.app_labels
    }
  }
}
