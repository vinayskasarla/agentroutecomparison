# One Secrets Manager secret per key. Terraform creates each with the placeholder "NOT_SET" and then
# ignores the value, so your real keys never end up in Terraform state. The app treats NOT_SET as
# "no key" and runs that provider in simulated mode.
#
# Store a real value (then redeploy, see outputs.next_steps):
#   aws secretsmanager put-secret-value --secret-id <name>/ANTHROPIC_API_KEY --secret-string 'sk-ant-...'
locals {
  secret_names = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "XAI_API_KEY",
    "GEMINI_API_KEY",
    "TYPESAFE_API_KEY",
    "REDIS_URL", # optional, e.g. an Upstash rediss:// URL; leave NOT_SET to use the in-process cache
  ]
}

resource "aws_secretsmanager_secret" "keys" {
  for_each = toset(local.secret_names)

  name                    = "${var.name}/${each.key}"
  description             = "${each.key} for ${var.name}. Set to NOT_SET to disable."
  recovery_window_in_days = 0 # delete immediately on destroy so the name can be reused
}

resource "aws_secretsmanager_secret_version" "placeholder" {
  for_each = aws_secretsmanager_secret.keys

  secret_id     = each.value.id
  secret_string = "NOT_SET"

  lifecycle {
    ignore_changes = [secret_string]
  }
}
