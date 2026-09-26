variable "region" {
  description = "AWS region to deploy into. App Runner must be available there."
  type        = string
  default     = "us-east-1"
}

variable "name" {
  description = "Name used for the App Runner service, ECR repo, IAM roles and secret prefix."
  type        = string
  default     = "agentroutecomparison"
}

variable "cpu" {
  description = "App Runner vCPU units: 256 (0.25), 512 (0.5), 1024 (1), 2048 (2) or 4096 (4)."
  type        = string
  default     = "512"
}

variable "memory" {
  description = "App Runner memory in MB. Must be a valid pairing with cpu, e.g. 256/512, 512/1024, 1024/2048."
  type        = string
  default     = "1024"
}

variable "budget_email" {
  description = "If set, creates a monthly AWS Budget that emails this address at 80% (forecast) and 100% (actual)."
  type        = string
  default     = ""
}

variable "monthly_budget_usd" {
  description = "Monthly budget limit in USD for the budget alert. Covers AWS charges only, not LLM API bills."
  type        = number
  default     = 30
}

variable "enable_rate_limit" {
  description = "Attach an AWS WAF rate limit per client IP (about $6-10/month). Recommended because the page has no login."
  type        = bool
  default     = false
}

variable "rate_limit_per_5min" {
  description = "Requests allowed per client IP in any 5-minute window when enable_rate_limit is true (minimum 100)."
  type        = number
  default     = 300
}

variable "audit_user_headers" {
  description = "Comma-separated request headers carrying the signed-in user from your SSO layer, checked in order. ALB/Cognito OIDC sets x-amzn-oidc-data (a JWT with email) and x-amzn-oidc-identity; oauth2-proxy sets x-forwarded-email."
  type        = string
  default     = "x-amzn-oidc-data,x-amzn-oidc-identity,x-forwarded-email,x-auth-request-email,x-forwarded-user"
}

variable "compare_limit_per_day" {
  description = "Full model comparisons allowed per client IP per rolling 24 hours. Behind a shared office IP everyone shares this limit."
  type        = number
  default     = 5
}

variable "architect_model" {
  description = "Claude model that reads the goal and writes test cases. claude-sonnet-5 matched claude-opus-5 on 42 of 44 decisions at ~43% of the cost."
  type        = string
  default     = "claude-sonnet-5"
}
