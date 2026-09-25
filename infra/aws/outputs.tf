output "url" {
  description = "Public URL of the app."
  value       = "https://${aws_apprunner_service.app.service_url}"
}

output "ecr_repository" {
  value = aws_ecr_repository.app.repository_url
}

output "image" {
  value = local.image_uri
}

output "secret_names" {
  description = "Secrets Manager secrets to fill with your API keys."
  value       = [for s in aws_secretsmanager_secret.keys : s.name]
}

output "next_steps" {
  value = <<-EOT
    1. Store the keys you have (leave the rest as NOT_SET to keep that provider simulated):
         aws secretsmanager put-secret-value --region ${var.region} --secret-id ${var.name}/ANTHROPIC_API_KEY --secret-string '<key>'
    2. Redeploy so App Runner reads the new secret values:
         aws apprunner start-deployment --region ${var.region} --service-arn ${aws_apprunner_service.app.arn}
    3. Open https://${aws_apprunner_service.app.service_url}
  EOT
}
