resource "aws_ecr_repository" "app" {
  name                 = var.name
  image_tag_mutability = "IMMUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "app" {
  repository = aws_ecr_repository.app.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep the 10 most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

# Build and push the image from your machine. The tag is a hash of the app's source, so the image is
# rebuilt (and App Runner redeployed) only when the code changes. Needs Docker and the AWS CLI locally.
locals {
  app_root = abspath("${path.module}/../..")
  src_files = sort(setunion(
    fileset(local.app_root, "*.py"),
    fileset(local.app_root, "static/**"),
    fileset(local.app_root, "knowledge/**"),
    toset(["requirements.txt", "Dockerfile"]),
  ))
  image_tag = substr(sha1(join("", [for f in local.src_files : filesha1("${local.app_root}/${f}")])), 0, 12)
  image_uri = "${aws_ecr_repository.app.repository_url}:${local.image_tag}"
  registry  = split("/", aws_ecr_repository.app.repository_url)[0]
}

resource "terraform_data" "image" {
  triggers_replace = [local.image_uri]

  provisioner "local-exec" {
    working_dir = local.app_root
    interpreter = ["/bin/bash", "-c"]
    command     = <<-EOT
      set -euo pipefail
      aws ecr get-login-password --region ${var.region} | docker login --username AWS --password-stdin ${local.registry}
      docker build --platform linux/amd64 -t ${local.image_uri} .
      docker push ${local.image_uri}
    EOT
  }
}
