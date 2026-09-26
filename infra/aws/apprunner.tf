# Exactly one instance: caches, the similarity index and rate limits are kept in process memory.
resource "aws_apprunner_auto_scaling_configuration_version" "single" {
  auto_scaling_configuration_name = substr("${var.name}-single", 0, 32)
  min_size                        = 1
  max_size                        = 1
  max_concurrency                 = 100
}

resource "aws_apprunner_service" "app" {
  service_name                   = var.name
  auto_scaling_configuration_arn = aws_apprunner_auto_scaling_configuration_version.single.arn

  source_configuration {
    auto_deployments_enabled = false

    authentication_configuration {
      access_role_arn = aws_iam_role.ecr_access.arn
    }

    image_repository {
      image_identifier      = local.image_uri
      image_repository_type = "ECR"

      image_configuration {
        port = "8080"
        runtime_environment_variables = {
          PORT     = "8080"
          SELF_URL = "http://127.0.0.1:8080"
          # Headers the SSO layer sets with the signed-in user (first match wins), and whether to redact
          # personal data from goals and prompts before they're written to the audit log.
          AUDIT_USER_HEADERS = var.audit_user_headers
          AUDIT_REDACT_PII   = "true"
          # Full model comparisons allowed per client IP per rolling 24 hours (first runs test a shortlist).
          COMPARE_LIMIT_PER_DAY = tostring(var.compare_limit_per_day)
          TRUSTED_PROXY_HOPS    = "1"
          ARCHITECT_MODEL       = var.architect_model
        }
        # Each env var is filled from its secret when a deployment starts.
        runtime_environment_secrets = { for k, s in aws_secretsmanager_secret.keys : k => s.arn }
      }
    }
  }

  instance_configuration {
    cpu               = var.cpu
    memory            = var.memory
    instance_role_arn = aws_iam_role.instance.arn
  }

  health_check_configuration {
    protocol            = "HTTP"
    path                = "/"
    interval            = 10
    timeout             = 5
    healthy_threshold   = 1
    unhealthy_threshold = 5
  }

  depends_on = [
    terraform_data.image,
    aws_iam_role_policy_attachment.ecr_access,
    aws_iam_role_policy.read_secrets,
    aws_secretsmanager_secret_version.placeholder,
  ]
}
