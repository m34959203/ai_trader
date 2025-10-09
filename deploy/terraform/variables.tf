variable "namespace" {
  description = "Kubernetes namespace for AI Trader"
  type        = string
  default     = "ai-trader"
}

variable "kubeconfig_path" {
  description = "Path to the kubeconfig file"
  type        = string
  default     = "~/.kube/config"
}

variable "kube_context" {
  description = "Kubernetes context to use"
  type        = string
  default     = ""
}

variable "image_repository" {
  description = "Container image repository"
  type        = string
}

variable "image_tag" {
  description = "Container image tag"
  type        = string
  default     = "latest"
}

variable "image_pull_policy" {
  description = "Kubernetes image pull policy"
  type        = string
  default     = "IfNotPresent"
}

variable "replicas" {
  description = "Desired number of replicas"
  type        = number
  default     = 3
}

variable "service_port" {
  description = "Service port exposed by the container"
  type        = number
  default     = 8000
}

variable "service_type" {
  description = "Kubernetes service type"
  type        = string
  default     = "ClusterIP"
}

variable "cpu_request" {
  description = "CPU request for the API container"
  type        = string
  default     = "250m"
}

variable "cpu_limit" {
  description = "CPU limit for the API container"
  type        = string
  default     = "750m"
}

variable "memory_request" {
  description = "Memory request for the API container"
  type        = string
  default     = "512Mi"
}

variable "memory_limit" {
  description = "Memory limit for the API container"
  type        = string
  default     = "1Gi"
}

variable "hpa_min_replicas" {
  description = "Minimum replicas for the HPA"
  type        = number
  default     = 2
}

variable "hpa_max_replicas" {
  description = "Maximum replicas for the HPA"
  type        = number
  default     = 6
}

variable "hpa_cpu_utilization" {
  description = "Target CPU utilization for the HPA"
  type        = number
  default     = 60
}

variable "service_account" {
  description = "Service account name used by the pods"
  type        = string
  default     = "default"
}

variable "app_env" {
  description = "Application environment (prod/staging/dev)"
  type        = string
  default     = "prod"
}

variable "log_level" {
  description = "Log level for the API"
  type        = string
  default     = "INFO"
}

variable "enable_live_trading" {
  description = "Toggle live trading features"
  type        = bool
  default     = true
}

variable "slack_webhook_url" {
  description = "Slack webhook URL for alert forwarding"
  type        = string
  default     = ""
}

variable "pagerduty_routing_key" {
  description = "PagerDuty routing key used by alertmanager"
  type        = string
  default     = ""
}
