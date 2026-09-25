# Audit trail in CloudWatch. The app writes one JSON line per user action (log_type = "audit") to stdout;
# App Runner ships it to the service's application log group. These resources add saved queries, a
# metric and a dashboard on top.
locals {
  app_log_group = "/aws/apprunner/${aws_apprunner_service.app.service_name}/${aws_apprunner_service.app.service_id}/application"
  audit_queries = {
    "who-asked-what"            = <<-Q
      fields @timestamp, user.email, data.goal, data.top3.0.architecture as recommended, data.top3.0.model as model, data.top3.0.monthly_usd as monthly_usd
      | filter log_type = "audit" and event = "advise_completed"
      | sort @timestamp desc
      | limit 200
    Q
    "recommended-architectures" = <<-Q
      filter log_type = "audit" and event = "advise_completed"
      | stats count(*) as recommendations by data.top3.0.architecture, data.top3.0.model
      | sort recommendations desc
    Q
    "activity-by-user"          = <<-Q
      filter log_type = "audit"
      | stats count(*) as actions, count_distinct(session_id) as sessions, max(@timestamp) as last_seen by user.email, event
      | sort actions desc
    Q
    "all-actions"               = <<-Q
      fields @timestamp, user.email, event, path, session_id, request_id, data
      | filter log_type = "audit"
      | sort @timestamp desc
      | limit 500
    Q
    "failures"                  = <<-Q
      fields @timestamp, user.email, data.goal, data.error
      | filter log_type = "audit" and event = "advise_failed"
      | sort @timestamp desc
    Q
  }
}

resource "aws_cloudwatch_query_definition" "audit" {
  for_each        = local.audit_queries
  name            = "${var.name}/audit/${each.key}"
  log_group_names = [local.app_log_group]
  query_string    = each.value
}

resource "aws_cloudwatch_log_metric_filter" "advice" {
  name           = "${var.name}-advice-completed"
  log_group_name = local.app_log_group
  pattern        = "{ ($.log_type = \"audit\") && ($.event = \"advise_completed\") }"
  metric_transformation {
    name      = "AdviceCompleted"
    namespace = var.name
    value     = "1"
    unit      = "Count"
  }
  depends_on = [aws_apprunner_service.app] # App Runner creates the log group when the service starts
}

resource "aws_cloudwatch_log_metric_filter" "advice_failed" {
  name           = "${var.name}-advice-failed"
  log_group_name = local.app_log_group
  pattern        = "{ ($.log_type = \"audit\") && ($.event = \"advise_failed\") }"
  metric_transformation {
    name      = "AdviceFailed"
    namespace = var.name
    value     = "1"
    unit      = "Count"
  }
  depends_on = [aws_apprunner_service.app]
}

resource "aws_cloudwatch_metric_alarm" "advice_failures" {
  alarm_name          = "${var.name}-advice-failures"
  alarm_description   = "Architecture advice requests are failing."
  namespace           = var.name
  metric_name         = "AdviceFailed"
  statistic           = "Sum"
  period              = 900
  evaluation_periods  = 1
  threshold           = 3
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  depends_on          = [aws_cloudwatch_log_metric_filter.advice_failed]
}

resource "aws_cloudwatch_dashboard" "audit" {
  dashboard_name = "${var.name}-usage"
  dashboard_body = jsonencode({
    widgets = [
      {
        type = "metric", x = 0, y = 0, width = 12, height = 6
        properties = {
          title   = "Architecture advice per hour"
          region  = var.region
          stat    = "Sum"
          period  = 3600
          view    = "timeSeries"
          metrics = [[var.name, "AdviceCompleted"], [var.name, "AdviceFailed"]]
        }
      },
      {
        type = "log", x = 12, y = 0, width = 12, height = 6
        properties = {
          title  = "Recommended architectures"
          region = var.region
          view   = "table"
          query  = "SOURCE '${local.app_log_group}' | ${replace(trimspace(local.audit_queries["recommended-architectures"]), "\n", " ")}"
        }
      },
      {
        type = "log", x = 0, y = 6, width = 24, height = 8
        properties = {
          title  = "Who asked what (latest)"
          region = var.region
          view   = "table"
          query  = "SOURCE '${local.app_log_group}' | ${replace(trimspace(local.audit_queries["who-asked-what"]), "\n", " ")}"
        }
      },
      {
        type = "log", x = 0, y = 14, width = 24, height = 6
        properties = {
          title  = "Activity by user"
          region = var.region
          view   = "table"
          query  = "SOURCE '${local.app_log_group}' | ${replace(trimspace(local.audit_queries["activity-by-user"]), "\n", " ")}"
        }
      },
    ]
  })
  depends_on = [aws_apprunner_service.app]
}
